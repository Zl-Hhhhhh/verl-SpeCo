"""Strict JSON summary of replay evidence (missing ranks never count as pass)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(root):
    cases = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        ranks = {}
        replica_events = defaultdict(dict)
        for path in sorted(folder.glob("rank_*/events.jsonl")):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            counts = defaultdict(int)
            rank = int(path.parent.name.split("_")[-1])
            for event in events:
                if event["event"] in ("baseline", "updated", "rolled_back", "save_complete") and "parameters" in event:
                    counts[event["event"]] += 1
                    replica_events[(event["event"], counts[event["event"]], rank % 4)][rank // 4] = event
            failures = [e for e in events if e.get("ok") is False or e.get("sp_equal") is False]
            ranks[path.parent.name] = {
                "last_event": events[-1]["event"],
                "passed": events[-1]["event"] == "PASS" and not failures,
                "completed_cycles": sum(e["event"] == "cycle_complete" for e in events),
                "first_failure": ({k: v for k, v in failures[0].items() if k not in ("parameters", "buffers", "optimizer", "output")}
                                  if failures else None),
                "sp_checks": sum("sp_equal" in e for e in events),
                "file_reloads": sum(e["event"] == "fresh_process_reload" for e in events),
            }
        replica_failures = []
        for (event_name, occurrence, sp_rank), replicas in replica_events.items():
            if len(replicas) != 4:
                continue
            first = replicas[0]
            for dp_rank, other in replicas.items():
                changed = [k for k, v in first["parameters"].items()
                           if v["sha256"] != other["parameters"][k]["sha256"]]
                optimizer_equal = first["optimizer"] == other["optimizer"]
                if changed or not optimizer_equal:
                    replica_failures.append({"event": event_name, "occurrence": occurrence, "sp_rank": sp_rank,
                                             "dp_ranks": [0, dp_rank], "changed_parameters": changed,
                                             "optimizer_equal": optimizer_equal})
        cases[folder.name] = {"passed": len(ranks) == 16 and all(r["passed"] for r in ranks.values()) and not replica_failures,
                              "ranks": ranks, "replica_failures": replica_failures}
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = summarize(args.root)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {"passed": v["passed"], "ranks": len(v["ranks"]), "replica_failures": len(v["replica_failures"]),
                         "last": v["ranks"].get("rank_00", {})} for k, v in results.items()}, indent=2))
