"""Build the VERL WildSci training and GPQA-Diamond validation Parquet files.

Inputs are pinned upstream snapshots stored in the drafter domain data bundle.
GPQA's official archive is password-protected; its password is published in
the official GPQA repository README.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import tempfile
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


WILDSCI_SHA256 = "de09e66d9c74399e22b5d942be55c7b1d3c778d3a0b7878da345bc42978d75b5"
GPQA_SHA256 = "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc"
GPQA_PASSWORD = b"deserted-untie-orchid"
GPQA_SHUFFLE_SEED = 20260804


def check_sha256(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {digest}")


def question_prompt(question: str, options: dict[str, str]) -> str:
    choices = "\n".join(f"{letter}. {answer}" for letter, answer in options.items())
    return (
        "Answer the following multiple-choice science question. Think step by step, "
        "then end with your final choice in the form 'Answer: X', where X is the "
        "option letter.\n\n"
        f"Question: {question.strip()}\n\nOptions:\n{choices}"
    )


def verl_row(source: str, question: str, options: dict[str, str], answer: str, extra: dict) -> dict:
    if (
        not question.strip()
        or answer not in options
        or any(not text.strip() for text in options.values())
    ):
        raise ValueError(f"Invalid multiple-choice item at {source} index {extra['index']}")
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": question_prompt(question, options)}],
        "ability": "science",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": extra,
    }


def wildsci_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for source_index, line in enumerate(stream):
            item = json.loads(line)
            if item["voting_type"] != "all_aligned":
                continue
            options = item["options"]
            if set(options) != set("ABCDEFGHIJ"):
                raise ValueError(f"WildSci options are not A-J at source row {source_index}")
            rows.append(
                verl_row(
                    "wildsci_all_aligned",
                    item["question"],
                    options,
                    item["answer"],
                    {
                        "index": len(rows),
                        "source_index": source_index,
                        "split": "train",
                        "source_dataset": "JustinTX/WildSci",
                        "paper_id": item["paper_id"],
                        "discipline": item["discipline"],
                    },
                )
            )
    if len(rows) != 13566:
        raise ValueError(f"Expected 13,566 WildSci All-Aligned rows, found {len(rows)}")
    return rows


def gpqa_rows(path: Path) -> list[dict]:
    rng = random.Random(GPQA_SHUFFLE_SEED)
    rows = []
    with zipfile.ZipFile(path) as archive:
        with archive.open("dataset/gpqa_diamond.csv", pwd=GPQA_PASSWORD) as binary:
            reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig"))
            for item in reader:
                choices = [
                    (item["Correct Answer"], True),
                    (item["Incorrect Answer 1"], False),
                    (item["Incorrect Answer 2"], False),
                    (item["Incorrect Answer 3"], False),
                ]
                rng.shuffle(choices)
                options = {letter: text for letter, (text, _) in zip("ABCD", choices)}
                answer = next(letter for letter, (_, correct) in zip("ABCD", choices) if correct)
                rows.append(
                    verl_row(
                        "gpqa_diamond",
                        item["Question"],
                        options,
                        answer,
                        {
                            "index": len(rows),
                            "split": "test",
                            "source_dataset": "idavidrein/gpqa",
                            "record_id": item["Record ID"],
                            "domain": item["High-level domain"],
                        },
                    )
                )
    if len(rows) != 198 or len({row["extra_info"]["record_id"] for row in rows}) != 198:
        raise ValueError("GPQA-Diamond must contain 198 distinct records")
    return rows


def write_parquet(rows: list[dict], path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Will not overwrite existing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as tmp:
        temporary_path = Path(tmp.name)
    try:
        # PyArrow 25 can fail to read dictionary-encoded nested prompt fields.
        pq.write_table(pa.Table.from_pylist(rows), temporary_path, use_dictionary=False)
        if len(pq.read_table(temporary_path)) != len(rows):
            raise ValueError(f"Parquet row count mismatch: {temporary_path}")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"{path}: {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path("/workspace/data/drafter_domains"))
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing generated Parquet files"
    )
    args = parser.parse_args()
    root = args.bundle_root
    wildsci = root / "raw/wildsci/wildsci.jsonl"
    gpqa = root / "raw/gpqa/dataset.zip"
    check_sha256(wildsci, WILDSCI_SHA256)
    check_sha256(gpqa, GPQA_SHA256)
    train = wildsci_rows(wildsci)
    test = gpqa_rows(gpqa)
    write_parquet(
        train, root / "verl/science/wildsci_all_aligned/train.parquet", overwrite=args.overwrite
    )
    write_parquet(
        test, root / "verl/science/eval/gpqa_diamond/test.parquet", overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
