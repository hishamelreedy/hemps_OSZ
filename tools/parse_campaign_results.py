#!/usr/bin/env python3
"""Parse HEMPS log output for MLP784 IC + probe campaign metrics."""

import argparse
import glob
import os
import re
import sys

SCENARIO_HINTS = (
    ("ht_payload", "ht_payload"),
    ("ht", "ht_payload"),
    ("secure", "secure"),
    ("vhdl", "vhdl"),
    ("accel", "accel"),
    ("clean", "clean"),
)


def scenario_name(path):
    # Prefer parent dirs (e.g. .../mlp784_clean/log/log0x0.txt) over basename.
    norm = path.replace("\\", "/").lower()
    for tag, label in SCENARIO_HINTS:
        if tag in norm:
            return label
    base = os.path.basename(path).lower()
    parts = base.replace("_sim.txt", "").replace(".txt", "").split("_")
    if parts and parts[0] == "mlp784" and len(parts) > 1:
        return "_".join(parts[1:])
    return os.path.splitext(base)[0]


def extract_log_value(line):
    line = line.strip()
    m = re.search(r"_[0-9a-fx]+_\d+_\d+_(-?\d+)\s*$", line, re.I)
    if m:
        return int(m.group(1))
    if re.match(r"^-?\d+\s*$", line):
        return int(line.strip())
    return None


def log_payload(line):
    if "_" in line and line.count("_") >= 3:
        return line.split("_", 4)[-1]
    return line


def next_value(lines, i):
    j = i + 1
    while j < len(lines):
        val = extract_log_value(lines[j])
        if val is not None:
            return val, j
        j += 1
    return None, i


def scan_text(text):
    lines = text.splitlines()
    expected = []
    predicted = []
    start_tick = None
    end_tick = None

    for i, line in enumerate(lines):
        pl = log_payload(line)
        if "MLP expected class:" in pl:
            val, _ = next_value(lines, i)
            if val is not None:
                expected.append(str(val))
        elif "MLP predicted class:" in pl:
            val, _ = next_value(lines, i)
            if val is not None:
                predicted.append(str(val))
        elif "MLP sample start tick:" in pl and start_tick is None:
            start_tick, _ = next_value(lines, i)
        elif "MLP sample end tick:" in pl and end_tick is None:
            end_tick, _ = next_value(lines, i)

    # Also accept compact same-line forms from sim transcripts.
    expected.extend(re.findall(r"MLP expected class:(\d+)", text))
    predicted.extend(re.findall(r"MLP predicted class:(\d+)", text))

    layers = re.findall(r"\[IC-ALERT\] layer=(-?\d+)", text)
    return {
        "ic_alerts": len(re.findall(r"\[IC-ALERT\]", text)),
        "ic_kernel": len(re.findall(r"\[IC-ALERT\] layer=", text)),
        "ic_layers": layers,
        "probe_success": len(re.findall(r"\bSUCCESS\b", text)),
        "probe_failure": len(re.findall(r"\bFAILURE\b", text)),
        "mlp_done": len(re.findall(r"MLP master done", text)),
        "predicted": predicted,
        "expected": expected,
        "accel_1cyc": len(re.findall(r"backend=hw_fc_1cyc", text)),
        "start_tick": start_tick,
        "end_tick": end_tick,
        "e2e_ticks": (end_tick - start_tick)
        if start_tick is not None and end_tick is not None
        else None,
    }


def scan_file(path):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    return scan_text(text)


def merge_stats(stats_list):
    out = {
        "ic_alerts": 0,
        "ic_kernel": 0,
        "probe_success": 0,
        "probe_failure": 0,
        "mlp_done": 0,
        "predicted": [],
        "expected": [],
        "accel_1cyc": 0,
        "ic_layers": [],
        "start_tick": None,
        "end_tick": None,
        "e2e_ticks": None,
    }
    for stats in stats_list:
        for key in ("ic_alerts", "ic_kernel", "probe_success", "probe_failure", "mlp_done", "accel_1cyc"):
            out[key] += stats[key]
        out["predicted"].extend(stats["predicted"])
        out["expected"].extend(stats["expected"])
        out["ic_layers"].extend(stats.get("ic_layers") or [])
        if out["start_tick"] is None and stats.get("start_tick") is not None:
            out["start_tick"] = stats["start_tick"]
        if out["end_tick"] is None and stats.get("end_tick") is not None:
            out["end_tick"] = stats["end_tick"]
        if out["e2e_ticks"] is None and stats.get("e2e_ticks") is not None:
            out["e2e_ticks"] = stats["e2e_ticks"]
    return out


def collect_paths(items):
    paths = []
    for item in items or ["log"]:
        if os.path.isdir(item):
            paths.extend(glob.glob(os.path.join(item, "**", "*.txt"), recursive=True))
        elif os.path.isfile(item):
            paths.append(item)
        else:
            paths.extend(glob.glob(item))
    return sorted(set(paths))


def aggregate_by_scenario(paths):
    groups = {}
    for path in paths:
        scen = scenario_name(path)
        groups.setdefault(scen, []).append(path)
    rows = []
    for scen in sorted(groups.keys()):
        stats = merge_stats(scan_file(p) for p in groups[scen])
        rows.append((scen, groups[scen], stats))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize IC/probe results from log files")
    parser.add_argument("logs", nargs="*", help="Log files or directories")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    parser.add_argument("--grid", action="store_true", help="Print Figure 7.4-style scenario grid")
    args = parser.parse_args()

    paths = collect_paths(args.logs)
    if not paths:
        print("No log files found", file=sys.stderr)
        return 1

    if args.grid:
        rows = aggregate_by_scenario(paths)
        print("Figure 7.4 grid (scenario x metrics)")
        print("%-14s %8s %8s %8s %8s %8s %8s %10s %10s" % (
            "scenario", "ic_alert", "probe_ok", "probe_fail", "mlp_done",
            "class_ok", "accel_1c", "e2e_ticks", "ic_layers",
        ))
        for scen, _, stats in rows:
            class_ok = "-"
            if stats["expected"] and stats["predicted"]:
                class_ok = "yes" if stats["expected"][-1] == stats["predicted"][-1] else "no"
            e2e = "-" if stats.get("e2e_ticks") is None else str(stats["e2e_ticks"])
            layers = ",".join(stats.get("ic_layers") or []) or "-"
            print("%-14s %8d %8d %8d %8d %8s %8d %10s %10s" % (
                scen,
                stats["ic_alerts"],
                stats["probe_success"],
                stats["probe_failure"],
                stats["mlp_done"],
                class_ok,
                stats["accel_1cyc"],
                e2e,
                layers,
            ))
        print("")

    for path in paths:
        stats = scan_file(path)
        if (
            stats["ic_alerts"] == 0
            and stats["mlp_done"] == 0
            and not stats["expected"]
            and not stats["predicted"]
            and stats["probe_success"] == 0
            and stats["probe_failure"] == 0
        ):
            continue
        print("%s" % path)
        print("  ic_alerts=%d probe_ok=%d probe_fail=%d mlp_done=%d accel_1cyc=%d"
              % (stats["ic_alerts"], stats["probe_success"], stats["probe_failure"],
                 stats["mlp_done"], stats["accel_1cyc"]))
        if stats["expected"] and stats["predicted"]:
            print("  expected=%s predicted=%s" % (stats["expected"][-1], stats["predicted"][-1]))
        if stats.get("e2e_ticks") is not None:
            print("  start_tick=%s end_tick=%s e2e_ticks=%s" % (
                stats["start_tick"], stats["end_tick"], stats["e2e_ticks"]))
        if stats.get("ic_layers"):
            print("  ic_layers=%s" % ",".join(stats["ic_layers"]))

    if args.csv:
        rows = aggregate_by_scenario(paths)
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write(
                "scenario,ic_alerts,probe_success,probe_failure,mlp_done,"
                "accel_1cyc,class_ok,e2e_ticks,ic_layers\n"
            )
            for scen, _, stats in rows:
                class_ok = ""
                if stats["expected"] and stats["predicted"]:
                    class_ok = "yes" if stats["expected"][-1] == stats["predicted"][-1] else "no"
                e2e = "" if stats.get("e2e_ticks") is None else str(stats["e2e_ticks"])
                layers = ",".join(stats.get("ic_layers") or [])
                f.write("%s,%d,%d,%d,%d,%d,%s,%s,%s\n" % (
                    scen,
                    stats["ic_alerts"],
                    stats["probe_success"],
                    stats["probe_failure"],
                    stats["mlp_done"],
                    stats["accel_1cyc"],
                    class_ok,
                    e2e,
                    layers,
                ))
        print("Wrote %s" % args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
