#!/usr/bin/env python3
"""Analyze paired full-vocabulary targets; never updates model weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from verl_speco.trainer.directional_sampling import build_repair_targets


def summary(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {"count": len(array), "mean": float(array.mean()),
            "p10": float(np.quantile(array, .10)), "p50": float(np.quantile(array, .5)),
            "p90": float(np.quantile(array, .9)), "p99": float(np.quantile(array, .99)),
            "max": float(array.max())}


def write_json_lines(frame, path):
    # Python's encoder preserves tiny nonzero surplus/deficit values; decimal
    # rounding here would change the reported two-sided condition counts.
    with path.open("w", encoding="utf-8") as stream:
        for row in frame.to_dict(orient="records"):
            cleaned = {key: None if pd.isna(value) else value for key, value in row.items()}
            stream.write(json.dumps(cleaned, allow_nan=False) + "\n")


def cohort_statistics(frame, noise_threshold):
    """Keep every conditional denominator next to its reported statistic."""
    count = len(frame)
    positive_tv = frame.current_tv > 0
    two_sided = (frame.surplus > 0) & (frame.deficit > 0)
    above_noise = frame.m > noise_threshold
    observed_trace = frame.trace_draft_count > 0
    total_tv = float(frame.current_tv.sum())
    return {
        "position_count": count,
        "independent_response_count": int(frame.sample_id.nunique()),
        "m_over_tv": summary(frame.loc[positive_tv, "m_over_tv"]),
        "m_over_tv_denominator": int(positive_tv.sum()),
        "zero_tv_positions_excluded_from_ratio": int((~positive_tv).sum()),
        "sum_m": float(frame.m.sum()), "sum_tv": total_tv,
        "sum_m_over_sum_tv": float(frame.m.sum() / total_tv) if total_tv else None,
        "sum_ratio_position_denominator": count,
        "above_noise_count": int(above_noise.sum()),
        "above_noise_fraction": float(above_noise.mean()),
        "above_noise_denominator": count,
        "two_sided_count": int(two_sided.sum()),
        "above_noise_given_two_sided_count": int((above_noise & two_sided).sum()),
        "above_noise_given_two_sided_fraction": float(above_noise[two_sided].mean())
        if two_sided.any() else None,
        "above_noise_given_two_sided_denominator": int(two_sided.sum()),
        "trace_observed_position_count": int(observed_trace.sum()),
        "trace_accept_fraction": summary(frame.loc[observed_trace, "trace_accept_fraction"]),
        "trace_first_proposal_accepted_count": int((frame.loc[observed_trace, "trace_accepted_count"] > 0).sum()),
        "trace_first_proposal_accepted_fraction": float((frame.loc[observed_trace, "trace_accepted_count"] > 0).mean())
        if observed_trace.any() else None,
        "trace_accept_denominator": int(observed_trace.sum()),
    }


def plot_distributions(root, candidates, selected):
    image = Image.new("RGB", (1500, 450), "white")
    draw = ImageDraw.Draw(image)
    for panel, field in enumerate(("m", "current_tv", "actor_entropy")):
        x0 = 50 + panel * 500
        draw.text((x0, 12), field, fill="black")
        low = min(float(candidates[field].min()), float(selected[field].min()))
        high = max(float(candidates[field].max()), float(selected[field].max()))
        for frame, color in ((candidates, "#2065a1"), (selected, "#d55532")):
            counts, _ = np.histogram(frame[field], bins=60, range=(low, high + 1e-9))
            counts = counts / max(len(frame), 1)
            scale = max(counts.max(), 1e-9)
            for index, count in enumerate(counts):
                left = x0 + int(index * 420 / 60)
                height = int(count / scale * 320)
                draw.line((left, 390, left, 390 - height), fill=color, width=2)
        draw.text((x0, 415), "all candidates: blue  |  selected top-m: red", fill="black")
    image.save(root / "candidate_selected_distributions.png")
    image = Image.new("RGB", (1100, 450), "white")
    draw = ImageDraw.Draw(image)
    for panel, field in enumerate(("absolute_position", "current_tv")):
        x0 = 55 + 540 * panel
        draw.text((x0, 12), f"m vs {field}", fill="black")
        xmin = float(candidates[field].min())
        xmax = float(candidates[field].max())
        ymax = float(candidates.m.max())
        for frame, color in ((candidates, "#2065a1"), (selected, "#d55532")):
            for x, y in zip(frame[field], frame.m, strict=True):
                px = x0 + int((float(x) - xmin) / max(xmax - xmin, 1e-9) * 430)
                py = 400 - int(float(y) / max(ymax, 1e-9) * 350)
                draw.point((px, py), fill=color)
    image.save(root / "position_and_mismatch.png")


def diagnose(root):
    manifest = json.loads((root / "paired_panel_manifest.json").read_text())
    if manifest["deploy_sampling"] != {
        "temperature": 1.0, "top_k": -1, "top_p": 1.0, "repetition_penalty": 1.0
    } or manifest["proposal_method"] != "greedy":
        raise ValueError("Deployment proposal/actor sampling differs from this diagnostic")
    candidates = pd.concat(
        [pd.read_json(path, lines=True, precise_float=True) for path in sorted(root.glob("candidates_rank*.jsonl"))],
        ignore_index=True,
    )
    candidates["selected"] = candidates["selected"].fillna(False).astype(bool)
    if len(candidates) != manifest["candidate_count"] or int(candidates.selected.sum()) != manifest["selected_count"]:
        raise ValueError("Candidate counts disagree with paired panel manifest")
    if candidates.duplicated(["sample_id", "anchor"]).any():
        raise ValueError("Duplicate paired candidate positions")
    candidates["m_over_tv"] = candidates.m / candidates.current_tv.replace(0, np.nan)
    candidates["trace_accept_fraction"] = (
        candidates.trace_accepted_count / candidates.trace_draft_count.replace(0, np.nan)
    )
    # This host's Arrow build can crash on dictionary indices and misread
    # bit-packed booleans. Store the selection flag explicitly as integer 0/1.
    candidates.assign(selected=candidates.selected.astype("int8")).to_parquet(
        root / "candidate_scores.parquet", index=False, use_dictionary=False
    )
    write_json_lines(candidates, root / "candidate_scores.jsonl")
    files = sorted(root.glob("paired_rank*.pt"))
    if len(files) != manifest["selected_sample_files"]:
        raise ValueError("Incomplete selected distribution files")
    metrics = []
    preliminary = []
    checks = {f"eta{eta:g}_{name}": {
        "checked_positions": 0, "minimum_probability": 1.0,
        "max_normalization_error": 0.0, "max_tv_improvement_error": 0.0,
    } for eta in (.5, 1.0) for name in ("direction", "all_mismatch")}
    for path in files:
        item = torch.load(path, map_location="cpu", weights_only=False)
        rows = item["rows"]
        if any(item[key].shape[0] != len(rows) for key in ("old", "new", "draft", "repeat_draft")):
            raise ValueError(f"Distribution row mismatch: {path}")
        for first in (True, False):
            index = list(range(min(8, len(rows)))) if first else list(range(len(rows)))
            if not index:
                continue
            old, new, draft = (item[key][index].float() for key in ("old", "new", "draft"))
            built = {eta: build_repair_targets(old, new, draft, eta) for eta in (.5, 1.0)}
            if not torch.allclose(built[1.0]["m"], torch.tensor([
                rows[i]["m"] for i in index
            ]), atol=1e-5, rtol=0):
                raise ValueError(f"Stored m differs from full distributions: {path}")
            if first:
                preliminary.extend(built[1.0]["m"].tolist())
                continue
            for eta, targets in built.items():
                for name in ("direction", "all_mismatch"):
                    target = targets[name]
                    check = checks[f"eta{eta:g}_{name}"]
                    check["checked_positions"] += len(index)
                    check["minimum_probability"] = min(check["minimum_probability"], float(target.min()))
                    check["max_normalization_error"] = max(check["max_normalization_error"], float((target.sum(-1) - 1).abs().max()))
                    actual_improvement = targets["current_tv"] - .5 * (new - target).abs().sum(-1)
                    check["max_tv_improvement_error"] = max(check["max_tv_improvement_error"], float((actual_improvement - eta * targets["m"]).abs().max()))
            q_repeat = item["repeat_draft"].float()
            repeat_targets = build_repair_targets(old, new, q_repeat, 1.0)
            null_targets = build_repair_targets(new, new, draft, 1.0)
            q_old_feature = item["old_feature_draft"]
            for i, row in enumerate(rows):
                original = int(draft[i].argmax())
                original_top_two = torch.topk(draft[i], 2).values
                datum = {"sample_id": path.stem, **row,
                         "repeat_q_max_abs": float((draft[i] - q_repeat[i]).abs().max()),
                         "repeat_m_abs": float((built[1.0]["m"][i] - repeat_targets["m"][i]).abs()),
                         "null_actor_change_m": float(null_targets["m"][i]),
                         "q0_top1_token": original,
                         "q0_margin": float(original_top_two[0] - original_top_two[1])}
                if q_old_feature is not None:
                    datum["feature_drift_tv"] = float(.5 * (
                        draft[i] - q_old_feature[i].float()
                    ).abs().sum())
                datum["greedy_p_before"] = float(new[i, original])
                for eta, targets in built.items():
                    for name in ("direction", "all_mismatch"):
                        target = targets[name][i]
                        token = int(target.argmax())
                        top_two = torch.topk(target, 2).values
                        prefix = f"eta{eta:g}_{name}"
                        datum[prefix + "_top1_token"] = token
                        datum[prefix + "_top1_changed"] = int(token != original)
                        datum[prefix + "_greedy_p_gain"] = float(new[i, token] - new[i, original])
                        datum[prefix + "_margin"] = float(top_two[0] - top_two[1])
                        datum[prefix + "_tv_improvement"] = float(
                            .5 * (new[i] - draft[i]).abs().sum()
                            - .5 * (new[i] - target).abs().sum()
                        )
                metrics.append(datum)
    if len(metrics) != manifest["selected_count"]:
        raise ValueError("Selected target count is incomplete")
    selected = pd.DataFrame(metrics)
    if "feature_drift_tv" not in selected or not selected.feature_drift_tv.notna().any():
        raise ValueError("No paired q(F-) / q(F+) feature-drift diagnostics were recorded")
    selected["m_over_tv"] = selected.m / selected.current_tv.replace(0, np.nan)
    selected["trace_accept_fraction"] = (
        selected.trace_accepted_count / selected.trace_draft_count.replace(0, np.nan)
    )
    selected.to_parquet(root / "selected_target_metrics.parquet", index=False, use_dictionary=False)
    write_json_lines(selected, root / "selected_target_metrics.jsonl")
    noise = selected.repeat_m_abs.to_numpy()
    threshold = max(1e-6, float(np.quantile(noise, .99)) * 10)
    report = {
        "initial_checked_positions": len(preliminary),
        "all_checked_positions": len(metrics),
        "numeric_tolerance": 1e-5,
        "repeat_m_noise": summary(noise),
        "repeat_q_max_abs": summary(selected.repeat_q_max_abs),
        "noise_threshold": threshold,
        "noise_threshold_definition": "max(1e-6, 10 * p99(abs(m(q_repeat)-m(q0)))) on selected positions",
        "probability_definitions": {
            "old": "p-: actor0 full-vocabulary FP32 softmax",
            "new": "p+: actor5 full-vocabulary FP32 softmax; deployment sampling is identical",
            "draft": "q0(F+): frozen original drafter with actor5 features and output head",
            "old_feature_draft": "q0(F-): same frozen drafter and actor5 output head, actor0 features on identical prefixes",
            "current_tv": "TV(p+, q0(F+))", "actor_tv": "TV(p-, p+)",
            "surplus": "S", "deficit": "U", "feature_drift_tv": "TV(q0(F-), q0(F+))",
        },
        "cohorts": {"all_candidates": cohort_statistics(candidates, threshold),
                    "selected_top_m": cohort_statistics(selected, threshold)},
        "target_probability_checks": {"scope": "selected top-m positions",
                                      "passed": True, "results": checks},
        "feature_drift_denominator": int(selected.feature_drift_tv.notna().sum()),
        "null_actor_change_m_max": float(selected.null_actor_change_m.max()),
        "candidate": {field: summary(candidates[field]) for field in
                      ("surplus", "deficit", "m", "current_tv", "actor_tv", "actor_entropy",
                       "absolute_position")},
        "selected": {field: summary(selected[field]) for field in
                     ("surplus", "deficit", "m", "current_tv", "actor_tv", "actor_entropy",
                      "absolute_position",
                      "repeat_q_max_abs", "repeat_m_abs")},
        "candidate_two_sided_fraction": float(((candidates.surplus > 0) & (candidates.deficit > 0)).mean()),
        "candidate_m_above_noise_fraction": float((candidates.m > threshold).mean()),
        "candidate_m_over_tv": summary(candidates.m_over_tv.dropna()),
        "selected_m_over_tv": summary(selected.m_over_tv.dropna()),
        "candidate_trace_accept_fraction": summary(candidates.trace_accept_fraction.dropna()),
        "selected_trace_accept_fraction": summary(selected.trace_accept_fraction.dropna()),
        "selected_m_above_noise_fraction": float((selected.m > threshold).mean()),
        "selected_sum_m_over_sum_tv": float(selected.m.sum() / selected.current_tv.sum()),
        "candidate_sum_m_over_sum_tv": float(candidates.m.sum() / candidates.current_tv.sum())
        if candidates.current_tv.sum() > 0 else None,
        "selected_feature_drift_available_fraction": float(selected.feature_drift_tv.notna().mean())
        if "feature_drift_tv" in selected else 0.0,
        "selected_feature_drift_tv": summary(selected.feature_drift_tv.dropna())
        if "feature_drift_tv" in selected else {"count": 0},
    }
    for eta in (.5, 1.0):
        for name in ("direction", "all_mismatch"):
            prefix = f"eta{eta:g}_{name}"
            report[prefix] = {
                "top1_change_fraction": float(selected[prefix + "_top1_changed"].mean()),
                "greedy_p_gain": summary(selected[prefix + "_greedy_p_gain"]),
                "beneficial_change_fraction": float((selected[prefix + "_greedy_p_gain"] > 0).mean()),
                "harmful_change_fraction": float((selected[prefix + "_greedy_p_gain"] < 0).mean()),
                "tv_improvement": summary(selected[prefix + "_tv_improvement"]),
            }
    (root / "target_diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_distributions(root, candidates, selected)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(json.dumps(diagnose(parser.parse_args().root), indent=2))
