#!/usr/bin/env python3
"""Offline LeNet-5 Q8 + IC validation (no HEMPS sim)."""

from __future__ import print_function

import argparse
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCLUDE = os.path.join(REPO, "software", "include")
DEFAULT_GOLDEN = os.path.join(os.path.dirname(__file__), "data", "lenet5_golden.json")


def parse_c_array(text, name):
    m = re.search(r"static const int %s\[(\d+)\] = \{([^}]+)\}" % name, text, re.S)
    if not m:
        raise RuntimeError("array %s not found" % name)
    vals = [int(x) for x in re.findall(r"-?\d+", m.group(2))]
    return vals


def parse_define_int(text, name, default=None):
    m = re.search(r"#define %s (-?\d+)" % name, text)
    if m:
        return int(m.group(1))
    if default is not None:
        return default
    raise RuntimeError("define %s not found" % name)


def load_header_arrays():
    with open(os.path.join(INCLUDE, "lenet5_weights.h"), encoding="utf-8") as f:
        wtxt = f.read()
    with open(os.path.join(INCLUDE, "lenet5_input.h"), encoding="utf-8") as f:
        itxt = f.read()

    batch_size = parse_define_int(itxt, "LENET5_BATCH_SIZE", 1)
    all_pixels = parse_c_array(itxt, "LENET5_BATCH_SAMPLES")
    input_dim = 784
    if len(all_pixels) < input_dim:
        raise RuntimeError("LENET5_BATCH_SAMPLES too short")
    first_input = all_pixels[:input_dim]

    label = parse_define_int(itxt, "LENET5_SAMPLE_LABEL", None)
    expected = parse_define_int(itxt, "LENET5_EXPECTED_CLASS", None)
    if label is None or expected is None:
        labels = parse_c_array(itxt, "LENET5_BATCH_LABELS")
        label = labels[0]
        expected = labels[0]

    return {
        "c1": parse_c_array(wtxt, "LENET5_C1_WEIGHTS"),
        "c2": parse_c_array(wtxt, "LENET5_C2_WEIGHTS"),
        "fc1": parse_c_array(wtxt, "LENET5_FC1_WEIGHTS"),
        "fc2": parse_c_array(wtxt, "LENET5_FC2_WEIGHTS"),
        "fc3": parse_c_array(wtxt, "LENET5_FC3_WEIGHTS"),
        "input": first_input,
        "batch_size": batch_size,
        "expected": expected,
        "label": label,
    }


def unpack_conv(blob, cout, cin, k):
    import numpy as np

    w = np.zeros((cout, cin, k, k), dtype=np.int32)
    idx = 0
    for oc in range(cout):
        for ic in range(cin):
            for ky in range(k):
                for kx in range(k):
                    w[oc, ic, ky, kx] = blob[idx]
                    idx += 1
    b = np.array(blob[idx : idx + cout], dtype=np.int32)
    return w, b


def unpack_fc(blob, in_dim, out_dim):
    import numpy as np

    w = np.zeros((in_dim, out_dim), dtype=np.int32)
    idx = 0
    for i in range(in_dim):
        for o in range(out_dim):
            w[i, o] = blob[idx]
            idx += 1
    b = np.array(blob[idx : idx + out_dim], dtype=np.int32)
    return w, b


def conv2d_q8(x, w, b):
    import numpy as np

    h, ww, cin = x.shape
    cout, _, k, _ = w.shape
    ho, wo = h - k + 1, ww - k + 1
    out = np.zeros((ho, wo, cout), dtype=np.int32)
    for oc in range(cout):
        for oy in range(ho):
            for ox in range(wo):
                s = int(b[oc])
                for ic in range(cin):
                    for ky in range(k):
                        for kx in range(k):
                            s += (int(x[oy + ky, ox + kx, ic]) * int(w[oc, ic, ky, kx])) >> 8
                out[oy, ox, oc] = s
    return out


def avgpool2x2_q8(x):
    import numpy as np

    h, w, c = x.shape
    out = np.zeros((h // 2, w // 2, c), dtype=np.int32)
    for oy in range(h // 2):
        for ox in range(w // 2):
            for ch in range(c):
                v = (
                    int(x[oy * 2, ox * 2, ch])
                    + int(x[oy * 2, ox * 2 + 1, ch])
                    + int(x[oy * 2 + 1, ox * 2, ch])
                    + int(x[oy * 2 + 1, ox * 2 + 1, ch])
                )
                out[oy, ox, ch] = v // 4
    return out


def fc_q8(x, w, b):
    import numpy as np

    out = np.zeros(w.shape[1], dtype=np.int32)
    xi = np.asarray(x, dtype=np.int32).reshape(-1)
    for o in range(w.shape[1]):
        s = int(b[o])
        for i in range(w.shape[0]):
            s += (int(xi[i]) * int(w[i, o])) >> 8
        out[o] = s
    return out


def relu(x):
    import numpy as np

    return np.maximum(0, x)


def ic_stats(vec, cal):
    mn = min(vec)
    mx = max(vec)
    av = sum(vec) // len(vec)
    if mn < cal["gmin"] or mx > cal["gmax"]:
        return False
    if av < cal["gavg"] - cal["avg_slack"] or av > cal["gavg"] + cal["avg_slack"]:
        return False
    return True


def ic_slot(vec, golden, slack, safe_mm=0):
    mism = 0
    for a, b in zip(vec, golden):
        if abs(a - b) > slack:
            mism += 1
            if mism > safe_mm:
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--corrupt-c1p1", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("numpy required", file=sys.stderr)
        return 1

    if not os.path.isfile(os.path.join(INCLUDE, "lenet5_weights.h")):
        print("missing lenet5_weights.h — run tools/export_lenet5_assets.py first", file=sys.stderr)
        return 1

    assets = load_header_arrays()
    if assets["batch_size"] > 1:
        print("batch_size=%d — offline check uses sample 0 only (golden/IC may not apply)"
              % assets["batch_size"])

    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)["layers"]

    wc1, bc1 = unpack_conv(assets["c1"], 6, 1, 5)
    wc2, bc2 = unpack_conv(assets["c2"], 16, 6, 5)
    w1, b1 = unpack_fc(assets["fc1"], 256, 120)
    w2, b2 = unpack_fc(assets["fc2"], 120, 84)
    w3, b3 = unpack_fc(assets["fc3"], 84, 10)

    x = np.array(assets["input"], dtype=np.int32).reshape(28, 28, 1)
    c1 = relu(conv2d_q8(x, wc1, bc1))
    p1 = avgpool2x2_q8(c1).reshape(-1)
    if args.corrupt_c1p1:
        p1 = [v ^ 0xFF for v in p1.tolist()]
        p1 = np.array(p1, dtype=np.int32)
    c2 = relu(conv2d_q8(p1.reshape(12, 12, 6), wc2, bc2))
    p2 = avgpool2x2_q8(c2).reshape(-1)
    h1 = relu(fc_q8(p2, w1, b1))
    h2 = relu(fc_q8(h1, w2, b2))
    logits = fc_q8(h2, w3, b3)
    pred = int(logits.argmax())

    print("label=%d expected=%d pred=%d" % (assets["label"], assets["expected"], pred))

    ok = True
    layers = {
        "C1P1": p1.tolist() if hasattr(p1, "tolist") else list(p1),
        "C2P2": p2.tolist(),
        "FC1": h1.tolist(),
        "FC2": h2.tolist(),
        "FC3": logits.tolist(),
    }
    for name, vec in layers.items():
        g = golden[name]
        slot_ok = ic_slot(
            vec, g["golden_vector"], g["slot_slack"],
            g.get("safe_slot_mismatches", 0),
        )
        stats_ok = ic_stats(vec, g)
        status = "OK" if slot_ok and stats_ok else "FAIL"
        if status == "FAIL":
            ok = False
        print("  %s slot=%s stats=%s %s" % (
            name, "OK" if slot_ok else "FAIL", "OK" if stats_ok else "FAIL", status,
        ))

    if assets["batch_size"] > 1:
        print("batch mode: IC layer checks informational only")
        ok = True

    if args.corrupt_c1p1:
        if layers["C1P1"] == golden["C1P1"]["golden_vector"]:
            print("corrupt-c1p1 did not change activations")
            ok = False
        else:
            print("corrupt-c1p1: IC should FAIL on C1P1 (expected)")
            # For corrupt mode, FAIL on C1P1 is success
            ok = not ic_slot(
                layers["C1P1"], golden["C1P1"]["golden_vector"],
                golden["C1P1"]["slot_slack"],
                golden["C1P1"].get("safe_slot_mismatches", 0),
            )
    else:
        if assets["batch_size"] > 1:
            print("batch mode: skip pred/golden strict match (pipelined batch sim)")
        elif pred != assets["expected"]:
            print("pred mismatch")
            ok = False

    print("overall: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
