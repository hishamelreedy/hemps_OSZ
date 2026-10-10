#!/usr/bin/env python3
"""Parse lenet_accel_unit PE log on slave [1,0]: SW vs HW per layer + batch summary."""

import argparse
import glob
import os
import re
import sys


def extract_value(line):
    line = line.strip()
    m = re.search(r"_\d+x\d+_\d+_\d+_(-?\d+)\s*$", line)
    if m:
        return int(m.group(1))
    m = re.search(r"_0x\d+_0_\d+_(-?\d+)\s*$", line)
    if m:
        return int(m.group(1))
    if re.match(r"^-?\d+\s*$", line):
        return int(line.strip())
    return None


def extract_field(line):
    line = line.strip()
    m = re.search(r"_\d+x\d+_\d+_\d+_(.+)\s*$", line)
    if m:
        return m.group(1)
    return None


def next_field(lines, i):
    j = i + 1
    while j < len(lines):
        field = extract_field(lines[j])
        if field is not None:
            if re.match(r"^-?\d+$", field):
                return int(field), j
            return field, j
        v = extract_value(lines[j])
        if v is not None:
            return v, j
        j += 1
    return None, j


def parse_log(text):
    lines = text.splitlines()
    samples = []
    summary = {}
    cur = None
    i = 0
    while i < len(lines):
        payload = lines[i]
        if "_" in payload and payload.count("_") >= 3:
            payload = payload.split("_", 4)[-1]

        if "ACCEL unit sample:" in payload:
            if cur is not None:
                samples.append(cur)
            cur = {"sample": None, "layers": {}, "pred_sw": None, "pred_hw": None,
                   "expected": None, "sw_hw": None}
            cur["sample"], i = next_field(lines, i)
            i += 1
            continue

        if cur is not None:
            if "ACCEL unit layer:" in payload:
                layer, i = next_field(lines, i)
                rec = {}
                j = i + 1
                while j < len(lines):
                    pl = lines[j]
                    if "_" in pl and pl.count("_") >= 3:
                        pl = pl.split("_", 4)[-1]
                    if "ACCEL unit max diff:" in pl:
                        rec["max_diff"], j = next_field(lines, j)
                    elif "ACCEL unit sw argmax:" in pl:
                        rec["sw_argmax"], j = next_field(lines, j)
                    elif "ACCEL unit hw argmax:" in pl:
                        rec["hw_argmax"], j = next_field(lines, j)
                    elif "ACCEL unit layer:" in pl or "ACCEL unit pred sw:" in pl:
                        break
                    j += 1
                if layer:
                    cur["layers"][layer] = rec
                i = j
                continue
            elif "ACCEL unit pred sw:" in payload:
                cur["pred_sw"], i = next_field(lines, i)
            elif "ACCEL unit pred hw:" in payload:
                cur["pred_hw"], i = next_field(lines, i)
            elif "ACCEL unit expected:" in payload:
                cur["expected"], i = next_field(lines, i)
            elif "ACCEL unit sample sw_hw:" in payload:
                status, i = next_field(lines, i)
                cur["sw_hw"] = status

        if "ACCEL unit sw_hw_match:" in payload:
            summary["sw_hw_match"], i = next_field(lines, i)
        elif "ACCEL unit pred_correct:" in payload:
            summary["pred_correct"], i = next_field(lines, i)
        elif "ACCEL unit batch total:" in payload:
            summary["batch_total"], i = next_field(lines, i)
        i += 1

    if cur is not None:
        samples.append(cur)
    return samples, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", help="log1x0.txt or log directory")
    args = parser.parse_args()

    if args.log and os.path.isfile(args.log):
        paths = [args.log]
    elif args.log and os.path.isdir(args.log):
        paths = glob.glob(os.path.join(args.log, "log1x0.txt"))
    else:
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        paths = glob.glob(os.path.join(
            repo, "testcases", "lenet5", "lenet5_accel_unit", "log", "log1x0.txt",
        ))

    if not paths:
        print("no log1x0.txt found", file=sys.stderr)
        return 1

    ok = True
    for path in paths:
        text = open(path, encoding="utf-8", errors="ignore").read()
        samples, summary = parse_log(text)
        print(path)
        if not samples:
            print("  no ACCEL unit records")
            ok = False
            continue
        for s in samples:
            print("  sample=%s expected=%s pred_sw=%s pred_hw=%s sw_hw=%s" % (
                s["sample"], s["expected"], s["pred_sw"], s["pred_hw"], s.get("sw_hw"),
            ))
            for layer in ("c1", "p1", "c2", "p2", "fc1", "fc2", "fc3"):
                rec = s.get("layers", {}).get(layer, {})
                md = rec.get("max_diff")
                if md is None:
                    continue
                status = "OK" if md == 0 else "FAIL"
                if md != 0:
                    ok = False
                print("    %s max_diff=%s %s" % (layer, md, status))
            if s.get("sw_hw") != "OK":
                ok = False
        if summary:
            total = summary.get("batch_total")
            match = summary.get("sw_hw_match")
            correct = summary.get("pred_correct")
            print("  batch: sw_hw_match=%s/%s pred_correct=%s/%s" % (
                match, total, correct, total,
            ))
            if total and match != total:
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
