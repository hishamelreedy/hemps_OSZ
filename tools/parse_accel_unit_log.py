#!/usr/bin/env python3
"""Parse mlp_accel_unit PE log: SW vs HW max diff per layer."""

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
    cur = None
    i = 0
    while i < len(lines):
        payload = lines[i]
        if "_" in payload and payload.count("_") >= 3:
            payload = payload.split("_", 4)[-1]

        if "ACCEL unit sample:" in payload:
            if cur is not None:
                samples.append(cur)
            cur = {"sample": None, "layers": {}, "in_sum": None,
                   "pred_sw": None, "pred_hw": None, "expected": None}
            cur["sample"], i = next_field(lines, i)
            i += 1
            continue

        if cur is not None:
            if "ACCEL unit in sum:" in payload:
                cur["in_sum"], i = next_field(lines, i)
            elif "ACCEL unit fc1 sum:" in payload:
                cur["layers"].setdefault("fc1", {})["sum"], i = next_field(lines, i)
            elif "ACCEL unit fc2 sum:" in payload:
                cur["layers"].setdefault("fc2", {})["sum"], i = next_field(lines, i)
            elif "ACCEL unit fc3 sum:" in payload:
                cur["layers"].setdefault("fc3", {})["sum"], i = next_field(lines, i)
            elif "ACCEL unit pred:" in payload:
                cur["pred_hw"], i = next_field(lines, i)
            elif "ACCEL unit layer:" in payload:
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
        i += 1
    if cur is not None:
        samples.append(cur)
    return samples


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
            repo, "testcases", "mlp784", "mlp784_accel_unit", "log", "log1x0.txt",
        ))

    if not paths:
        print("no log1x0.txt found", file=sys.stderr)
        return 1

    ok = True
    for path in paths:
        text = open(path, encoding="utf-8", errors="ignore").read()
        samples = parse_log(text)
        print(path)
        if not samples:
            print("  no ACCEL unit records")
            ok = False
            continue
        for s in samples:
            print("  sample=%s in_sum=%s expected=%s pred_sw=%s pred_hw=%s" % (
                s["sample"], s["in_sum"], s["expected"], s["pred_sw"], s["pred_hw"],
            ))
            for layer, rec in s.get("layers", {}).items():
                md = rec.get("max_diff")
                status = "OK" if md == 0 else "FAIL"
                if md != 0:
                    ok = False
                if rec.get("sum") is not None:
                    print("    %s sum=%s" % (layer, rec["sum"]))
                if md is not None:
                    print("    %s max_diff=%s sw_ix=%s hw_ix=%s %s" % (
                        layer, md, rec.get("sw_argmax"), rec.get("hw_argmax"), status,
                    ))
            if s["expected"] is not None:
                if s["pred_sw"] != s["pred_hw"]:
                    print("    fc3 SW/HW mismatch")
                    ok = False
                if s["pred_hw"] == s["expected"]:
                    print("    HW pipeline pred OK")
                else:
                    print("    HW pipeline pred MISS (pred_hw=%s expected=%s)" % (
                        s["pred_hw"], s["expected"],
                    ))
                    ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
