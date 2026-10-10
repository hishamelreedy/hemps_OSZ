#!/usr/bin/env python3
"""Parse distributed LeNet-5 PE logs for per-task and critical-path timing."""
import os
import re
import sys
from pathlib import Path

LAYOUTS = {
    "4x4": [
        ("master", "log0x0.txt", "[0,0]"),
        ("c1p1", "log1x0.txt", "[1,0]"),
        ("c2p2", "log2x0.txt", "[2,0]"),
        ("fc1", "log3x0.txt", "[3,0]"),
        ("fc2", "log0x1.txt", "[0,1]"),
        ("fc3", "log1x1.txt", "[1,1]"),
    ],
    "6x3": [
        ("master", "log0x0.txt", "[0,0]"),
        ("c1p1", "log1x0.txt", "[1,0]"),
        ("c2p2", "log2x0.txt", "[2,0]"),
        ("fc1", "log3x0.txt", "[3,0]"),
        ("fc2", "log4x0.txt", "[4,0]"),
        ("fc3", "log5x0.txt", "[5,0]"),
    ],
    "6x3_spatial": [
        ("p0-master", "log0x0.txt", "[0,0]"),
        ("p0-c1p1", "log1x0.txt", "[1,0]"),
        ("p0-c2p2", "log2x0.txt", "[2,0]"),
        ("p0-fc1", "log3x0.txt", "[3,0]"),
        ("p0-fc2", "log4x0.txt", "[4,0]"),
        ("p0-fc3", "log5x0.txt", "[5,0]"),
        ("p1-master", "log0x1.txt", "[0,1]"),
        ("p1-c1p1", "log1x1.txt", "[1,1]"),
        ("p1-c2p2", "log2x1.txt", "[2,1]"),
        ("p1-fc1", "log3x1.txt", "[3,1]"),
        ("p1-fc2", "log4x1.txt", "[4,1]"),
        ("p1-fc3", "log5x1.txt", "[5,1]"),
    ],
}


def parse_log(path):
    data = {
        "alloc": None,
        "start": None,
        "done": None,
        "term": None,
        "recv": [],
        "send": [],
        "accel": [],
    }
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"Task id: \d+ allocated at (\d+)", line)
            if m:
                data["alloc"] = int(m.group(1))
                continue
            m = re.match(r"Task id: \d+ terminated at (\d+)", line)
            if m:
                data["term"] = int(m.group(1))
                continue
            m = re.match(r"(\d+)_[^\s]+_ (send|recv)\b", line)
            if m:
                tick, kind = int(m.group(1)), m.group(2)
                (data["send"] if kind == "send" else data["recv"]).append(tick)
                continue
            m = re.match(r"(\d+)_[^\s]+_\[ACCEL\]", line)
            if m:
                data["accel"].append(int(m.group(1)))
                continue
            if "LENET master start" in line:
                data["start"] = int(line.split("_")[0])
            elif "LENET master done" in line:
                data["done"] = int(line.split("_")[0])
            elif "LENET c1p1 start" in line:
                data["start"] = int(line.split("_")[0])
            elif "LENET c1p1 done" in line:
                data["done"] = int(line.split("_")[0])
            elif "LENET c2p2 start" in line:
                data["start"] = int(line.split("_")[0])
            elif "LENET c2p2 done" in line:
                data["done"] = int(line.split("_")[0])
            elif "LENET fc1 start" in line:
                data["start"] = int(line.split("_")[0])
            elif "LENET fc1 done" in line:
                data["done"] = int(line.split("_")[0])
            elif "LENET fc2 start" in line:
                data["start"] = int(line.split("_")[0])
            elif "LENET fc2 done" in line:
                data["done"] = int(line.split("_")[0])
            elif "LENET fc3 start" in line:
                data["start"] = int(line.split("_")[0])
            elif "LENET fc3 done" in line:
                data["done"] = int(line.split("_")[0])
    return data


def main():
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("log")
    sim_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 5.83502
    total_ticks = int(sys.argv[3]) if len(sys.argv) > 3 else 581231
    layout = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("LENET5_LAYOUT", "4x4")
    if layout not in LAYOUTS:
        sys.exit("Unknown layout %r (use 4x4, 6x3, or 6x3_spatial)" % layout)

    def ms(ticks):
        return ticks * sim_ms / total_ticks

    tasks = LAYOUTS[layout]
    print("Layout: %s" % layout)

    parsed = {}
    print("=== Per-task timing (PE wall time: start -> done) ===")
    print(f"{'Task':<8} {'PE':<6} {'Start':>8} {'Done':>8} {'Active':>8} {'ms':>8} {'Recv':>5} {'Send':>5} {'Accel@':>20}")
    print("-" * 90)
    for name, fname, pe in tasks:
        d = parse_log(log_dir / fname)
        parsed[name] = d
        start = d["start"] if d["start"] is not None else d["alloc"]
        done = d["done"] if d["done"] is not None else d["term"]
        active = done - start
        accel_s = ",".join(str(t) for t in d["accel"][:3])
        print(
            f"{name:<8} {pe:<6} {start:8d} {done:8d} {active:8d} {ms(active):8.3f} "
            f"{len(d['recv']):5d} {len(d['send']):5d} {accel_s}"
        )

    # Critical path from master markers (single-pipeline layouts only)
    if layout not in ("4x4", "6x3"):
        print("\n(Skip critical-path summary for multi-pipeline layout %s)" % layout)
        return

    m = parsed["master"]
    master_start = m["start"]
    master_done = m["done"]
    send_done = m["send"][-1] if m["send"] else None

    c1_done = parsed["c1p1"]["done"]
    c2_done = parsed["c2p2"]["done"]
    fc1_done = parsed["fc1"]["done"]
    fc2_done = parsed["fc2"]["done"]
    fc3_done = parsed["fc3"]["done"]

    segs = [
        ("Master send input (784 x7 chunks)", master_start, send_done),
        ("c1p1 stage", send_done, c1_done),
        ("c2p2 stage", c1_done, c2_done),
        ("fc1 stage", c2_done, fc1_done),
        ("fc2 stage", fc1_done, fc2_done),
        ("fc3 stage", fc2_done, fc3_done),
        ("Master recv output", fc3_done, master_done),
    ]

    total = master_done - master_start
    print(f"\n=== Critical path (master start -> master done) ===")
    print(f"Simulation time: {sim_ms} ms over {total_ticks} ticks")
    print(f"End-to-end: {total} ticks = {ms(total):.3f} ms\n")
    print(f"{'Segment':<36} {'Ticks':>8} {'ms':>8} {'Share':>7}")
    print("-" * 62)
    bottleneck = ("", 0)
    for label, a, b in segs:
        if a is None or b is None:
            continue
        dt = b - a
        if dt > bottleneck[1]:
            bottleneck = (label, dt)
        print(f"{label:<36} {dt:8d} {ms(dt):8.3f} {100 * dt / total:6.1f}%")
    print("-" * 62)
    print(f"Bottleneck: {bottleneck[0]} ({ms(bottleneck[1]):.3f} ms, {100 * bottleneck[1] / total:.1f}%)")

    # Boot overlap: time from alloc to start
    print("\n=== Boot delay (allocated -> LENET start) ===")
    for name, _, pe in tasks:
        if name == "master":
            continue
        d = parsed[name]
        if d["alloc"] and d["start"]:
            delay = d["start"] - d["alloc"]
            print(f"  {name:<6} {pe}: {delay:6d} ticks = {ms(delay):.3f} ms")


if __name__ == "__main__":
    main()
