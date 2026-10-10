#!/usr/bin/env python3
"""Generate TB-only LeNet batch input header (no numpy required)."""

from __future__ import print_function

import argparse
import gzip
import os
import struct
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DEFAULT = os.path.join(REPO, "software", "include", "lenet5_tb_batch.h")


def load_mnist(split="test"):
    if split == "train":
        img_name, lbl_name = "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
    else:
        img_name, lbl_name = "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"

    img_path = os.path.join(DATA_DIR, img_name)
    lbl_path = os.path.join(DATA_DIR, lbl_name)
    if not os.path.isfile(img_path) or not os.path.isfile(lbl_path):
        raise RuntimeError("MNIST files missing under %s (run export once)" % DATA_DIR)

    with gzip.open(img_path, "rb") as f:
        raw = f.read()
    with gzip.open(lbl_path, "rb") as f:
        lraw = f.read()

    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise RuntimeError("bad MNIST magic %d" % magic)
    images = raw[16:]
    labels = lraw[8:]
    return n, rows, cols, images, labels


def emit_array(name, values, cols=8):
    lines = ["static const int %s[%d] = {" % (name, len(values))]
    row = []
    for i, v in enumerate(values):
        row.append(str(int(v)))
        if len(row) == cols or i == len(values) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    lines.append("};")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("-o", "--output", default=OUT_DEFAULT)
    args = parser.parse_args()

    n, rows, cols, images, labels = load_mnist(args.split)
    dim = rows * cols
    if args.start + args.batch > n:
        print("not enough samples (have %d)" % n, file=sys.stderr)
        return 1

    pixels = []
    labs = []
    idxs = []
    for i in range(args.start, args.start + args.batch):
        off = i * dim
        pixels.extend(images[off : off + dim])
        labs.append(labels[i])
        idxs.append(i)

    lines = [
        "#ifndef LENET5_TB_BATCH_H_",
        "#define LENET5_TB_BATCH_H_",
        "",
        "/* TB-only batch (%s MNIST[%d:%d]) — not for PE apps */" % (
            args.split, args.start, args.start + args.batch),
        "",
        "#define LENET5_TB_BATCH_SIZE %d" % args.batch,
        "#define LENET5_TB_INPUT_DIM %d" % dim,
        "",
    ]
    lines.extend(emit_array("LENET5_TB_BATCH_SAMPLES", pixels))
    lines.append("")
    lines.extend(emit_array("LENET5_TB_BATCH_LABELS", labs))
    lines.append("")
    lines.extend(emit_array("LENET5_TB_BATCH_INDICES", idxs))
    lines.append("")
    lines.append("#endif")
    lines.append("")

    out = args.output
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote %s (%d samples)" % (out, args.batch))
    return 0


if __name__ == "__main__":
    sys.exit(main())
