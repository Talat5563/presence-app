"""Compares harness runs side by side, on the households every run enrolled.

Usage: python -I tools/experiments/compare_runs.py MANIFEST_DIR PROBES_DIR RUN_DIR [RUN_DIR ...]

A setting that enrols more children also adds their (harder) episodes, so the runs' own
results.csv files are not comparable. This replays each run's observations.csv through the
engine's gates and accumulator (EvidenceAccumulator, MatcherConfig.applyGates without negatives,
Metrics.scoreEvents) restricted to the households enrolled in every run, then reports the best
detection reachable at no more than 2 and 0.5 false alarms per hour — spec §4.3's ITERATE and
SHIP false-alarm bars — and the operating point that reached it.

Recording durations come from the vadProbes index (frames x 32 ms), which drops at most one
partial frame per file against the harness's own exposure hours.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

WINDOW_MS, COOLDOWN_MS, BUDGET_MS = 20_000, 45_000, 10_000
MIN_Z, MAX_ABS_LLR = 0.5, 4.0
GATES = (0.35, 0.3, 0.25, 0.2)
THRESHOLDS = (2, 4, 6, 8, 10)


def gated(raw, similarity, z, gate):
    llr = raw
    if z < MIN_Z:
        llr = min(llr, 0.0)
    if similarity < gate:
        llr = min(llr, 0.0)
    return max(-MAX_ABS_LLR, min(MAX_ABS_LLR, llr))


def detections(observations, gate, threshold):
    window, cooldown_from, fired = [], None, []
    for ts, raw, similarity, z in observations:
        if cooldown_from is not None and ts - cooldown_from < COOLDOWN_MS:
            continue
        window.append((ts, gated(raw, similarity, z, gate)))
        window = [(t, v) for t, v in window if t >= ts - WINDOW_MS]
        if sum(v for _, v in window) >= threshold:
            fired.append(ts)
            cooldown_from, window = ts, []
    return fired


def main():
    manifest, probes, runs = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
    with open(manifest / "recordings.csv", encoding="utf-8") as f:
        recordings = list(csv.DictReader(f))
    with open(manifest / "events.csv", encoding="utf-8") as f:
        events = defaultdict(list)
        for e in csv.DictReader(f):
            events[e["recording_path"]].append(int(e["start_ms"]))
    with open(probes / "index.csv", encoding="utf-8") as f:
        hours = {r["path"]: int(r["frames"]) * 0.032 / 3600 for r in csv.DictReader(f)}

    def skipped(run):
        text = (run / "report.md").read_text(encoding="utf-8")
        return {line.split("SKIP household ")[1].split(":")[0] for line in text.splitlines() if "SKIP household" in line}

    households = {r["household_id"] for r in recordings} - set().union(*(skipped(run) for run in runs))
    evaluated = [r["path"] for r in recordings
                 if r["household_id"] in households and r["content_type"] not in ("enrollment", "read")]
    exposure = sum(hours[p] for p in evaluated)
    total_events = sum(len(events[p]) for p in evaluated)
    print(f"{len(households)} households in every run, {total_events} events, {exposure:.1f} h\n")

    for run in runs:
        by_path, count, positive = defaultdict(list), Counter(), Counter()
        with open(run / "observations.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["household_id"] not in households:
                    continue
                by_path[r["path"]].append((int(r["timestamp_ms"]), float(r["raw_llr"]),
                                           float(r["target_similarity"]), float(r["cohort_z"])))
                count[r["role"]] += 1
                positive[r["role"]] += float(r["llr"]) > 0
        points = []
        for gate in GATES:
            for threshold in THRESHOLDS:
                hits = fired = 0
                for path in evaluated:
                    fires = detections(by_path.get(path, []), gate, threshold)
                    unused = sorted(fires)
                    for start in sorted(events[path]):
                        match = next((t for t in unused if start <= t <= start + BUDGET_MS), None)
                        if match is not None:
                            unused.remove(match)
                            hits += 1
                    fired += len(fires)
                points.append((hits / total_events, (fired - hits) / exposure, gate, threshold))

        def best(max_fa):
            ok = [p for p in points if p[1] <= max_fa]
            if not ok:
                return "—"
            rate, fa, gate, threshold = max(ok, key=lambda p: (p[0], -p[1]))
            return f"{rate:6.1%} (gate {gate}, thr {threshold}, {fa:.2f}/h)"

        print(f"== {run.name}")
        print(f"   best at <=2 FA/h:   {best(2.0)}")
        print(f"   best at <=0.5 FA/h: {best(0.5)}")
        print("   observations: " + ", ".join(
            f"{role} {count[role]} ({positive[role] / count[role]:.0%} positive)" for role in sorted(count)))


if __name__ == "__main__":
    main()
