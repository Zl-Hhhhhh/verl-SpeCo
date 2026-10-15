"""Prepare pinned complete MATH-500 and GSM8K test splits for DAPO-style evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SOURCES = {
    "math500": (
        "HuggingFaceH4/MATH-500",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "test.jsonl",
        500,
    ),
    "gsm8k": (
        "openai/grade-school-math",
        "3101c7d5072418e28b9008a6636bde82a006892c",
        "grade_school_math/data/test.jsonl",
        1319,
    ),
}
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def make_row(question: str, answer: str, *, index: int, source: str) -> dict:
    if not question.strip() or not answer.strip():
        raise ValueError(f"Blank {source} question or answer at {index}")
    return {
        "data_source": "math_dapo",
        "prompt": [{"role": "user", "content": PROMPT_PREFIX + question.strip() + PROMPT_SUFFIX}],
        "ability": "MATH",
        "reward_model": {"style": "rule-lighteval/MATH_v2", "ground_truth": answer.strip()},
        "extra_info": {"index": index, "raw_problem": question.strip(), "source_dataset": source},
    }


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/data/math_eval_dapo"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (repo, revision, filename, expected_count) in SOURCES.items():
        if name == "math500":
            raw = Path(hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision,
                filename=filename,
                local_dir=args.output_dir / "raw" / name,
            ))
        else:
            raw = args.output_dir / "raw" / name / "test.jsonl"
            if not raw.exists():
                raw.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://raw.githubusercontent.com/{repo}/{revision}/{filename}"
                with urlopen(url, timeout=60) as response:
                    content = response.read()
                raw.write_bytes(content)
        if name == "math500":
            records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
            rows = [make_row(item["problem"], item["answer"], index=i, source=name)
                    for i, item in enumerate(records)]
            ids = [item["unique_id"] for item in records]
        else:
            records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
            rows = [make_row(
                item["question"], item["answer"].rsplit("####", 1)[1].strip().replace(",", ""),
                index=i, source=name,
            ) for i, item in enumerate(records)]
            ids = [item["question"] for item in records]
        if len(rows) != expected_count or len(set(ids)) != expected_count:
            raise ValueError(f"{name}: expected {expected_count} unique records, got {len(rows)} / {len(set(ids))}")
        destination = args.output_dir / name / "test.parquet"
        if destination.exists():
            existing = pq.read_table(destination)
            if existing.to_pylist() != rows:
                raise FileExistsError(f"Existing evaluation data differs: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), destination, use_dictionary=False)
        if len(pq.read_table(destination)) != expected_count:
            raise ValueError(f"Parquet count mismatch: {destination}")
        manifest[name] = {
            "source": repo,
            "revision": revision,
            "filename": filename,
            "raw_sha256": sha256(raw),
            "parquet_sha256": sha256(destination),
            "count": expected_count,
            "parquet": str(destination),
            "scorer": "verl.utils.reward_score.math_dapo (Answer: format)",
        }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
