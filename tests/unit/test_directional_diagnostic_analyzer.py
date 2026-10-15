import importlib.util
import json
from pathlib import Path

import torch


def test_diagnostic_analyzer_writes_expected_artifacts(tmp_path):
    script = Path(__file__).resolve().parents[2] / "outputs/experiment_setup/directional_repair_diagnostic_20261015/analyze.py"
    spec = importlib.util.spec_from_file_location("directional_diagnostic_analyzer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = {
        "candidate_count": 1, "selected_count": 1, "selected_sample_files": 1,
        "proposal_method": "greedy",
        "deploy_sampling": {"temperature": 1.0, "top_k": -1, "top_p": 1.0,
                            "repetition_penalty": 1.0},
    }
    (tmp_path / "paired_panel_manifest.json").write_text(json.dumps(manifest))
    row = {"sample_id": "sample", "anchor": 2, "absolute_position": 12,
           "surplus": .2, "deficit": .2, "m": .2, "current_tv": .2,
           "actor_tv": .2, "actor_entropy": .9, "selected": True,
           "trace_accepted_count": 3, "trace_draft_count": 7}
    (tmp_path / "candidates_rank00.jsonl").write_text(json.dumps(row) + "\n")
    torch.save({"rows": [{key: value for key, value in row.items() if key != "selected"}],
                "old": torch.tensor([[.3, .1, .6]]),
                "new": torch.tensor([[.1, .3, .6]]),
                "draft": torch.tensor([[.3, .1, .6]]),
                "repeat_draft": torch.tensor([[.3, .1, .6]]),
                "old_feature_draft": torch.tensor([[.3, .1, .6]])},
               tmp_path / "paired_rank00_sample.pt")
    report = module.diagnose(tmp_path)
    assert report["initial_checked_positions"] == 1
    assert report["all_checked_positions"] == 1
    assert report["eta0.5_direction"]["top1_change_fraction"] == 0
    for name in ("candidate_scores.parquet", "selected_target_metrics.parquet",
                 "target_diagnostics.json", "candidate_selected_distributions.png",
                 "position_and_mismatch.png"):
        assert (tmp_path / name).is_file()
