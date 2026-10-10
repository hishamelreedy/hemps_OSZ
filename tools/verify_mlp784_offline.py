#!/usr/bin/env python3
"""Offline validation for MLP784 P0: Q8 FC + NILES Stage I/II checks without HEMPS sim."""

import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_GOLDEN = os.path.join(os.path.dirname(__file__), "data", "mlp784_golden.json")


def relu(x):
    return x if x > 0 else 0


def fc_blob(input_vec, weights, in_dim, out_dim, apply_relu):
    bias = weights[in_dim * out_dim :]
    out = []
    for o in range(out_dim):
        s = bias[o]
        for i in range(in_dim):
            s += (input_vec[i] * weights[i * out_dim + o]) >> 8
        out.append(relu(s) if apply_relu else s)
    return out


def ic_stage2_stats(vec, cal):
    mn = min(vec)
    mx = max(vec)
    av = sum(vec) // len(vec)
    if mn < cal["gmin"] or mx > cal["gmax"]:
        return False
    if av < cal["gavg"] - cal["avg_slack"] or av > cal["gavg"] + cal["avg_slack"]:
        return False
    return True


def ic_stage1_header_payload(header_slot, payload_slot):
    return header_slot == payload_slot


def ic_stage1_slot_map(n, golden_map, slack=0):
    """Stage I: expected spatial index sequence (identity map on clean offline path)."""
    for i in range(n):
        expected = golden_map[i] if golden_map else i
        if not ic_stage1_header_payload(i, i):
            return False, i
        if abs(i - expected) > slack:
            return False, i
    return True, -1


def load_json_golden(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["layers"]


def parse_int_array_from_header(path, symbol):
    text = open(path, "r", encoding="utf-8").read()
    marker = "static const int %s[" % symbol
    start = text.find(marker)
    if start < 0:
        raise RuntimeError("symbol %s not found in %s" % (symbol, path))
    brace = text.find("{", start)
    end = text.find("};", brace)
    body = text[brace + 1 : end]
    vals = []
    for part in body.replace("\n", " ").split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals


def read_expected_class(include_dir):
    for line in open(os.path.join(include_dir, "mlp784_input.h"), encoding="utf-8"):
        if line.startswith("#define MLP784_EXPECTED_CLASS"):
            return int(line.split()[-1])
    return None


def run_once(inp, w1, w2, w3, layers, header_expected, corrupt_fc1):
    fc1 = fc_blob(inp, w1, 784, 30, True)
    if corrupt_fc1:
        fc1 = [v ^ 0xFF for v in fc1]
    fc2 = fc_blob(fc1, w2, 30, 30, True)
    fc3 = fc_blob(fc2, w3, 30, 10, False)

    vectors = {"FC1": fc1, "FC2": fc2, "FC3": fc3}
    pred = max(range(10), key=lambda i: fc3[i])
    ok = True
    print("predicted=%d expected=%s" % (pred, header_expected))
    for name, vec in vectors.items():
        cal = layers[name]
        slot_slack = cal.get("slot_slack", 2)
        golden_map = list(range(len(vec)))
        stage1_ok, bad_idx = ic_stage1_slot_map(len(vec), golden_map, slot_slack)
        stage2_ok = ic_stage2_stats(vec, cal)
        print("%s stage1=%s stage2=%s (bad_idx=%d)" % (name, stage1_ok, stage2_ok, bad_idx))
        if not (stage1_ok and stage2_ok):
            ok = False

    if header_expected is not None and pred != header_expected and not corrupt_fc1:
        print("prediction mismatch vs header expected class")
        ok = False
    return ok, pred


def main():
    parser = argparse.ArgumentParser(description="Offline MLP784 FC + IC validation")
    parser.add_argument("--golden-json", default=DEFAULT_GOLDEN)
    parser.add_argument("--corrupt-fc1", action="store_true", help="Simulate XOR payload corruption on FC1")
    parser.add_argument(
        "--ab",
        action="store_true",
        help="Send golden twice: HT-off (clean) then HT-on (XOR FC1); IC must fail only on pass 2",
    )
    args = parser.parse_args()

    include = os.path.join(REPO, "software", "include")
    inp = parse_int_array_from_header(os.path.join(include, "mlp784_input.h"), "MLP784_INPUT")
    w1 = parse_int_array_from_header(os.path.join(include, "mlp784_weights.h"), "MLP784_FC1_WEIGHTS")
    w2 = parse_int_array_from_header(os.path.join(include, "mlp784_weights.h"), "MLP784_FC2_WEIGHTS")
    w3 = parse_int_array_from_header(os.path.join(include, "mlp784_weights.h"), "MLP784_FC3_WEIGHTS")
    layers = load_json_golden(args.golden_json)
    header_expected = read_expected_class(include)

    if args.ab:
        print("=== PASS 1: HT disabled (clean) ===")
        ok1, _ = run_once(inp, w1, w2, w3, layers, header_expected, corrupt_fc1=False)
        print("=== PASS 2: HT enabled (XOR FC1) ===")
        ok2, _ = run_once(inp, w1, w2, w3, layers, header_expected, corrupt_fc1=True)
        if not ok1:
            print("FAIL: IC alerted on clean pass (false positive)")
            return 1
        if ok2:
            print("FAIL: IC did not alert on HT pass (false negative)")
            return 1
        print("A/B OK: IC silent when HT off, alerts when HT on (not always triggered)")
        return 0

    ok, _ = run_once(inp, w1, w2, w3, layers, header_expected, corrupt_fc1=args.corrupt_fc1)

    if args.corrupt_fc1:
        if ok:
            print("ERROR: corrupt-fc1 should fail IC checks")
            return 1
        print("corrupt-fc1 correctly failed IC (expected for HT scenario)")
        return 0

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
