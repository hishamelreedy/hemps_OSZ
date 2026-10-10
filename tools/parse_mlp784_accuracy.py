#!/usr/bin/env python3
"""Parse MLP784 batch accuracy from HEMPS PE UART logs (log0x0.txt)."""

import argparse
import glob
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MANIFEST = os.path.join(os.path.dirname(__file__), "data", "mlp784_batch.json")


def extract_log_value(line):
    """Return integer payload from PE log line, or None."""
    line = line.strip()
    m = re.search(r"_0x0_0_0_(-?\d+)\s*$", line)
    if m:
        return int(m.group(1))
    if re.match(r"^-?\d+\s*$", line):
        return int(line.strip())
    return None


def extract_log_tick(line):
    """Return leading tick counter from PE log line, or None."""
    m = re.match(r"^(\d+)_", line.strip())
    if m:
        return int(m.group(1))
    return None


def next_value(lines, i):
    j = i + 1
    while j < len(lines):
        val = extract_log_value(lines[j])
        if val is not None:
            return val, j
        j += 1
    return None, j


def parse_master_log(text):
    """Parse Echo split lines from master PE log."""
    records = []
    lines = text.splitlines()
    i = 0
    batch_correct = None
    batch_size = None

    while i < len(lines):
        line = lines[i]
        payload = line
        if "_" in line and line.count("_") >= 3:
            payload = line.split("_", 4)[-1]

        if "MLP batch size:" in payload or payload.endswith("MLP batch size:"):
            batch_size, i = next_value(lines, i)
            i += 1
            continue

        if "MLP batch correct:" in payload or payload.endswith("MLP batch correct:"):
            batch_correct, i = next_value(lines, i)
            i += 1
            continue

        if "MLP sample:" in payload or payload.endswith("MLP sample:"):
            rec = {
                "sample": None, "mnist_idx": None, "expected": None, "predicted": None,
                "start_tick": None, "end_tick": None, "input_sum": None,
            }
            rec["sample"], i = next_value(lines, i)
            i += 1
            while i < len(lines):
                pl = lines[i]
                if "_" in pl and pl.count("_") >= 3:
                    pl = pl.split("_", 4)[-1]
                if "MLP sample start tick:" in pl or pl.endswith("MLP sample start tick:"):
                    tick, i = next_value(lines, i)
                    rec["start_tick"] = tick
                elif "MLP input sum:" in pl or pl.endswith("MLP input sum:"):
                    rec["input_sum"], i = next_value(lines, i)
                elif "MLP mnist idx:" in pl or pl.endswith("MLP mnist idx:"):
                    rec["mnist_idx"], i = next_value(lines, i)
                elif "MLP expected class:" in pl or pl.endswith("MLP expected class:"):
                    rec["expected"], i = next_value(lines, i)
                elif "MLP predicted class:" in pl or pl.endswith("MLP predicted class:"):
                    rec["predicted"], i = next_value(lines, i)
                elif "MLP sample end tick:" in pl or pl.endswith("MLP sample end tick:"):
                    tick, i = next_value(lines, i)
                    rec["end_tick"] = tick
                    if rec["sample"] is not None:
                        records.append(rec)
                    break
                elif "MLP sample:" in pl or pl.endswith("MLP sample:"):
                    if rec["sample"] is not None and rec["predicted"] is not None:
                        records.append(rec)
                    break
                i += 1
            i += 1
            continue
        i += 1

    return records, batch_correct, batch_size


def dedupe_records(records):
    """Keep last record per batch sample index (log may contain reruns)."""
    by_sample = {}
    for rec in records:
        if rec["sample"] is not None:
            by_sample[rec["sample"]] = rec
    return [by_sample[k] for k in sorted(by_sample.keys())]


def print_latency_summary(records, tick_to_ms=1e-5):
    """Print per-sample latency and throughput from master tick stamps.

    PE GetTick() uses 10 ns per tick (see pe.cpp Simulation time conversion).
    """
    latencies = []
    for rec in records:
        if rec.get("start_tick") is None or rec.get("end_tick") is None:
            continue
        dt = rec["end_tick"] - rec["start_tick"]
        if dt < 0:
            continue
        lat_ms = dt * tick_to_ms
        latencies.append((rec["sample"], dt, lat_ms))
        print("    latency sample=%s ticks=%d ~%.3f ms (start=%s end=%s)" % (
            rec["sample"], dt, lat_ms, rec["start_tick"], rec["end_tick"],
        ))
    if len(latencies) >= 2:
        total_ticks = latencies[-1][1]  # per-sample only; rough
        spans = [latencies[i][1] for i in range(len(latencies))]
        avg_ticks = sum(spans) / float(len(spans))
        avg_ms = avg_ticks * tick_to_ms
        throughput = len(latencies) / (sum(spans) * tick_to_ms) if sum(spans) else 0
        print("  latency avg: %.3f ms/image (%d ticks @ 1 tick=1ns est.)" % (avg_ms, int(avg_ticks)))
        print("  throughput (sequential): %.4f images/ms (~%.1f images/s at 1ns/tick)" % (
            throughput, throughput * 1000.0,
        ))
        print("  note: 1 tick = 10 ns (HEMPS PE time slice)")
    elif len(latencies) == 1:
        print("  latency: %.3f ms for 1 sample" % latencies[0][2])


def parse_sim_time_ms(text):
    m = re.search(r"Simulation time:\s*([\d.]+)ms", text)
    if m:
        return float(m.group(1))
    return None


def load_manifest(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(records, batch_correct, batch_size):
    if not records:
        return None
    correct = sum(
        1 for r in records
        if r["expected"] is not None and r["predicted"] == r["expected"]
    )
    total = len(records)
    if batch_correct is not None and batch_correct != correct:
        print("note: parsed correct=%d differs from logged batch correct=%d" % (
            correct, batch_correct,
        ))
    if batch_size is not None and batch_size != total:
        print("note: parsed samples=%d differs from logged batch size=%d" % (
            total, batch_size,
        ))
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / float(total) if total else 0.0,
        "records": records,
    }


def find_master_logs(paths):
    out = []
    for path in paths:
        if os.path.isdir(path):
            for hit in glob.glob(os.path.join(path, "**", "log0x0.txt"), recursive=True):
                out.append(hit)
            for hit in glob.glob(os.path.join(path, "log0x0.txt")):
                out.append(hit)
        elif os.path.isfile(path):
            out.append(path)
    return sorted(set(out))


def main():
    parser = argparse.ArgumentParser(description="MLP784 batch accuracy from sim logs")
    parser.add_argument(
        "logs",
        nargs="*",
        help="log0x0.txt or testcase log/ directory (default: testcases/mlp784/mlp784_batch/log)",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Offline batch manifest JSON")
    parser.add_argument("--latency", action="store_true", help="Print per-sample latency from tick stamps")
    parser.add_argument("--sim-log", default=None, help="vsim transcript/sim txt for Simulation time line")
    parser.add_argument("--csv", default=None, help="Optional CSV output")
    args = parser.parse_args()

    if args.logs:
        log_paths = find_master_logs(args.logs)
    else:
        default_dir = os.path.join(REPO, "testcases", "mlp784", "mlp784_batch", "log")
        log_paths = find_master_logs([default_dir])

    if not log_paths:
        print("No log0x0.txt found", file=sys.stderr)
        return 1

    manifest = load_manifest(args.manifest)
    overall_records = []

    for path in log_paths:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
        records, batch_correct, batch_size = parse_master_log(text)
        records = dedupe_records(records)
        summary = summarize(records, batch_correct, batch_size)
        print("%s" % path)
        if not summary:
            print("  no batch records found")
            continue
        print("  samples=%d correct=%d accuracy=%.4f" % (
            summary["total"], summary["correct"], summary["accuracy"],
        ))
        for rec in summary["records"]:
            ok = rec["expected"] == rec["predicted"]
            print("    sample=%d mnist_idx=%s expected=%s predicted=%s %s" % (
                rec["sample"],
                rec["mnist_idx"],
                rec["expected"],
                rec["predicted"],
                "OK" if ok else "MISS",
            ))
        if args.latency:
            print("  --- latency / throughput (sequential batch) ---")
            print_latency_summary(summary["records"])
        overall_records.extend(summary["records"])

    if len(log_paths) > 1 and overall_records:
        total = len(overall_records)
        correct = sum(
            1 for r in overall_records
            if r["expected"] is not None and r["predicted"] == r["expected"]
        )
        print("")
        print("overall: samples=%d correct=%d accuracy=%.4f" % (
            total, correct, correct / float(total),
        ))

    if manifest and overall_records:
        offline = manifest.get("offline_preds", [])
        indices = manifest.get("indices", [])
        mism = 0
        for rec in overall_records:
            s = rec["sample"]
            if s is None or s >= len(offline):
                continue
            if rec["predicted"] != offline[s]:
                mism += 1
                print("offline/sim pred mismatch sample=%d mnist=%s offline=%s sim=%s" % (
                    s, indices[s] if s < len(indices) else "?",
                    offline[s], rec["predicted"],
                ))
        if mism == 0 and len(overall_records) == len(offline):
            print("offline Q8 preds match all sim preds")

    if args.sim_log and os.path.isfile(args.sim_log):
        sim_text = open(args.sim_log, "r", encoding="utf-8", errors="ignore").read()
        sim_ms = parse_sim_time_ms(sim_text)
        if sim_ms is not None and overall_records:
            n = len(overall_records)
            print("")
            print("simulation time: %.3f ms total" % sim_ms)
            print("throughput (end-to-end): %.4f images/ms (%.2f images/s)" % (
                n / sim_ms, (n / sim_ms) * 1000.0,
            ))
            accel_hits = len(re.findall(r"\[ACCEL\]", sim_text))
            if accel_hits:
                print("HW accel invocations (transcript): %d" % accel_hits)

    if args.csv and overall_records:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("sample,mnist_idx,expected,predicted,correct\n")
            for rec in overall_records:
                ok = int(rec["expected"] == rec["predicted"])
                f.write("%s,%s,%s,%s,%d\n" % (
                    rec["sample"], rec["mnist_idx"], rec["expected"], rec["predicted"], ok,
                ))
        print("Wrote %s" % args.csv)

    return 0 if overall_records else 1


if __name__ == "__main__":
    sys.exit(main())
