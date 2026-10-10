#!/usr/bin/env python3
"""Quick offline check: golden sample pred matches expected for AlexNet/VGG exports."""

from __future__ import print_function

import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCLUDE = os.path.join(REPO, "software", "include")


def check(prefix, expected_macro="EXPECTED_CLASS"):
    path = os.path.join(INCLUDE, "%s_input.h" % prefix.lower())
    if not os.path.isfile(path):
        print("%s: MISSING %s" % (prefix, path))
        return False
    text = open(path, encoding="utf-8").read()
    exp = int(re.search(r"#define %s_%s (-?\d+)" % (prefix, expected_macro), text).group(1))
    lab = int(re.search(r"#define %s_SAMPLE_LABEL (-?\d+)" % prefix, text).group(1))
    wpath = os.path.join(INCLUDE, "%s_weights.h" % prefix.lower())
    gpath = os.path.join(INCLUDE, "ic_golden_%s.h" % prefix.lower())
    ok = os.path.isfile(wpath) and os.path.isfile(gpath)
    print("%s: label=%d expected=%d weights=%s golden=%s %s" % (
        prefix, lab, exp,
        "OK" if os.path.isfile(wpath) else "MISSING",
        "OK" if os.path.isfile(gpath) else "MISSING",
        "PASS" if ok and exp == lab else "FAIL",
    ))
    return ok and exp == lab


def main():
    ok = True
    for p in ("LENET5", "ALEXNET", "VGG16"):
        if not check(p):
            ok = False
    print("overall: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
