#!/usr/bin/env python3
"""Offline VGG16-lite Q8 + IC validation (no HEMPS sim)."""
from __future__ import print_function
import json, os, re, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from export_vgg16_assets import infer_q8

INCLUDE = os.path.join(REPO, "software", "include")
GOLDEN = os.path.join(os.path.dirname(__file__), "data", "vgg16_golden.json")


def parse_array(text, name):
    m = re.search(r"static const int %s\[\d+\] = \{([^}]+)\}" % name, text, re.S)
    if not m:
        raise RuntimeError("missing %s" % name)
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))]


def unpack_conv(blob, cout, cin, k):
    import numpy as np
    n = cout * cin * k * k
    w = np.array(blob[:n], dtype=np.int32).reshape(cout, cin, k, k)
    b = np.array(blob[n : n + cout], dtype=np.int32)
    return w, b


def unpack_fc(blob, din, dout):
    import numpy as np
    n = din * dout
    w = np.array(blob[:n], dtype=np.int32).reshape(din, dout)
    b = np.array(blob[n : n + dout], dtype=np.int32)
    return w, b


def ic_ok(vec, cal):
    golden = cal["golden_vector"]
    slack = cal.get("slot_slack", 2)
    if len(vec) != len(golden):
        return False, "len"
    for a, b in zip(vec, golden):
        if abs(a - b) > slack:
            return False, "slot"
    mn, mx, av = min(vec), max(vec), sum(vec) // len(vec)
    if mn < cal["gmin"] or mx > cal["gmax"]:
        return False, "stats"
    if av < cal["gavg"] - cal["avg_slack"] or av > cal["gavg"] + cal["avg_slack"]:
        return False, "avg"
    return True, "ok"


def main():
    import numpy as np
    wtxt = open(os.path.join(INCLUDE, "vgg16_weights.h"), encoding="utf-8").read()
    itxt = open(os.path.join(INCLUDE, "vgg16_input.h"), encoding="utf-8").read()
    img = np.array(parse_array(itxt, "VGG16_BATCH_SAMPLES"), dtype=np.int32)
    expected = int(re.search(r"#define VGG16_EXPECTED_CLASS (-?\d+)", itxt).group(1))
    w1, b1 = unpack_conv(parse_array(wtxt, "VGG16_B1_WEIGHTS"), 8, 1, 5)
    w2, b2 = unpack_conv(parse_array(wtxt, "VGG16_B2_WEIGHTS"), 16, 8, 5)
    w3, b3 = unpack_conv(parse_array(wtxt, "VGG16_B3_WEIGHTS"), 24, 16, 3)
    wf1, bf1 = unpack_fc(parse_array(wtxt, "VGG16_FC1_WEIGHTS"), 216, 64)
    wf2, bf2 = unpack_fc(parse_array(wtxt, "VGG16_FC2_WEIGHTS"), 64, 10)
    r = infer_q8(img, w1, b1, w2, b2, w3, b3, wf1, bf1, wf2, bf2)
    layers = json.load(open(GOLDEN, encoding="utf-8"))["layers"]
    print("predicted=%d expected=%d" % (r["pred"], expected))
    ok = r["pred"] == expected
    for name in ["B1", "B2", "B3", "FC1", "FC2"]:
        good, why = ic_ok(r[name], layers[name])
        print("%s %s" % (name, "OK" if good else "FAIL(%s)" % why))
        ok = ok and good
    print("overall: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
