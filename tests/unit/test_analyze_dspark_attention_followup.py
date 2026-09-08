from __future__ import annotations

import json
from pathlib import Path

from tools.analyze_dspark_attention_followup import analyze


def _write_run(
    log_root: Path, analysis_root: Path, prefix: str, arm: str, token_hash: str
) -> None:
    (log_root / f"{prefix}_{arm}.log").write_text(
        "step:0 - val-aux/drafter/spec_decode/mean_acceptance_length:4"
        " - val-aux/drafter/spec_decode/verify_steps:10"
        " - val-aux/drafter/spec_decode/accepted_tokens:30"
        " - val-aux/drafter/spec_decode/generation_time_s:2"
        " - val-aux/drafter/spec_decode/response_tokens:64\n",
        encoding="utf-8",
    )
    diagnostics = analysis_root / prefix / "validation" / arm / "diagnostics"
    text_dir = analysis_root / prefix / "validation" / arm / "text"
    diagnostics.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    (diagnostics / "spec_decode.step-000000.json").write_text(
        json.dumps(
            {
                "ordered_response_token_sha256": token_hash,
                "sample_count": 1,
                "trace_event_count": 2,
                "acceptance_by_position": [],
            }
        ),
        encoding="utf-8",
    )
    (text_dir / "0.jsonl").write_text(
        json.dumps({"output": "same output"}) + "\n", encoding="utf-8"
    )


def test_followup_analyzer_aggregates_runs_and_checks_hashes(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    analysis_root = tmp_path / "analysis"
    output_dir = tmp_path / "combined"
    log_root.mkdir()
    for prefix in ("repeat1", "repeat2"):
        for arm in ("full", "swa1024"):
            _write_run(log_root, analysis_root, prefix, arm, "same-token-hash")

    payload = analyze(
        prefixes=("repeat1", "repeat2"),
        arms=("full", "swa1024"),
        log_root=log_root,
        analysis_root=analysis_root,
        output_dir=output_dir,
    )

    assert all(row["completed_runs"] == 2 for row in payload["aggregate"])
    assert payload["output_equivalence"]["token_outputs_all_equal"] is True
    assert payload["output_equivalence"]["text_outputs_all_equal"] is True
    assert (output_dir / "analysis.json").is_file()
    assert (output_dir / "runs.csv").is_file()
    assert (output_dir / "aggregate.csv").is_file()
