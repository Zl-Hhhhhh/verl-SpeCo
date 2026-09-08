from __future__ import annotations

import json
from pathlib import Path

from tools.summarize_dspark_attention_ablation import summarize


def test_summarize_attention_ablation_writes_requested_metrics(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    output_dir = tmp_path / "analysis"
    log_root.mkdir()
    (log_root / "trial_full.log").write_text(
        "(Task pid=1) step:0 - val-aux/drafter/spec_decode/mean_acceptance_length:3.25"
        " - val-aux/drafter/spec_decode/verify_steps:100"
        " - val-aux/drafter/spec_decode/accepted_tokens:225"
        " - val-aux/drafter/spec_decode/generation_time_s:10.5"
        " - val-aux/drafter/spec_decode/response_tokens:640\n",
        encoding="utf-8",
    )

    records = summarize("trial", log_root, output_dir)

    assert records[0]["status"] == "completed"
    assert records[0]["mean_acceptance_length"] == 3.25
    assert records[0]["verify_steps"] == 100.0
    assert records[0]["accepted_tokens"] == 225.0
    assert records[0]["generation_time_s"] == 10.5
    assert records[0]["response_tokens"] == 640.0
    assert records[1]["status"] == "pending"
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["ablation_prefix"] == "trial"
    assert (output_dir / "summary.csv").is_file()


def test_summarize_attention_ablation_includes_quality_and_position_metrics(
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "logs"
    output_dir = tmp_path / "analysis"
    log_root.mkdir()
    (log_root / "quality_swa1024.log").write_text(
        "step:0 - val-aux/math_dapo/reward/mean@1:0.25"
        " - val-core/math_dapo/acc/mean@1:0.5"
        " - val-aux/drafter/spec_decode/mean_acceptance_length:4.0"
        " - val-aux/drafter/spec_decode/verify_steps:10"
        " - val-aux/drafter/spec_decode/accepted_tokens:30"
        " - val-aux/drafter/spec_decode/generation_time_s:2"
        " - val-aux/drafter/spec_decode/response_tokens:64"
        " - val-aux/drafter/spec_decode/response_tokens_per_second:32"
        " - val-aux/drafter/spec_decode/position_1_attempted:10"
        " - val-aux/drafter/spec_decode/position_1_accepted:9"
        " - val-aux/drafter/spec_decode/position_1_acceptance_rate:0.9\n",
        encoding="utf-8",
    )

    records = summarize(
        "quality", log_root, output_dir, arms=("swa1024",)
    )

    assert records[0]["status"] == "completed"
    assert records[0]["reward_mean"] == 0.25
    assert records[0]["accuracy_mean"] == 0.5
    assert records[0]["position_1_acceptance_rate"] == 0.9
    assert len(records) == 1
