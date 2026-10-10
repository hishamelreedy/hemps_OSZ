#!/usr/bin/env python3
"""Generate software/include/lenet5_input.h with N MNIST samples for pipelined batch runs."""

from __future__ import print_function

import argparse
import gzip
import os
import re
import struct
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INCLUDE_DIR = os.path.join(REPO, "software", "include")
TB_BATCH_H = os.path.join(INCLUDE_DIR, "lenet5_tb_batch.h")
OUT_DEFAULT = os.path.join(INCLUDE_DIR, "lenet5_input.h")


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


def load_mnist(split="test"):
    if split == "train":
        img_name, lbl_name = "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
    else:
        img_name, lbl_name = "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"

    img_path = os.path.join(DATA_DIR, img_name)
    lbl_path = os.path.join(DATA_DIR, lbl_name)
    if not os.path.isfile(img_path) or not os.path.isfile(lbl_path):
        return None

    with gzip.open(img_path, "rb") as f:
        raw = f.read()
    with gzip.open(lbl_path, "rb") as f:
        lraw = f.read()

    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise RuntimeError("bad MNIST magic %d" % magic)
    return n, rows * cols, raw[16:], lraw[8:]


def load_from_tb_batch(batch, start=0):
    if not os.path.isfile(TB_BATCH_H):
        return None
    with open(TB_BATCH_H, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"static const int LENET5_TB_BATCH_SAMPLES\[(\d+)\] = \{([^}]+)\}", text, re.S)
    if not m:
        return None
    all_pixels = [int(x) for x in re.findall(r"-?\d+", m.group(2))]
    dim = 784
    need = (start + batch) * dim
    if len(all_pixels) < need:
        return None
    labs = [int(x) for x in re.findall(r"-?\d+", re.search(
        r"LENET5_TB_BATCH_LABELS\[(\d+)\] = \{([^}]+)\}", text, re.S).group(2))]
    idxs = [int(x) for x in re.findall(r"-?\d+", re.search(
        r"LENET5_TB_BATCH_INDICES\[(\d+)\] = \{([^}]+)\}", text, re.S).group(2))]
    pixels = all_pixels[start * dim : (start + batch) * dim]
    return pixels, labs[start : start + batch], idxs[start : start + batch]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument(
        "--spatial-pipelines",
        type=int,
        default=0,
        help="If >0, N parallel pipelines (one MNIST sample each); sets LENET5_BATCH_SIZE=1",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("-o", "--output", default=OUT_DEFAULT)
    parser.add_argument(
        "--no-tensor-debug",
        action="store_true",
        help="Set CNN_TENSOR_DEBUG=0 (default is 1: log [CNN] send/recv chunks in PE logs)",
    )
    args = parser.parse_args()

    spatial = args.spatial_pipelines
    if spatial > 0:
        args.batch = spatial
    if args.batch < 1:
        print("batch must be >= 1", file=sys.stderr)
        return 1

    pixels = labs = idxs = None
    mnist = load_mnist(args.split)
    if mnist is not None:
        n, dim, images, labels = mnist
        if args.start + args.batch > n:
            print("not enough MNIST samples", file=sys.stderr)
            return 1
        pixels = []
        for i in range(args.start, args.start + args.batch):
            off = i * dim
            pixels.extend(images[off : off + dim])
        labs = [labels[i] for i in range(args.start, args.start + args.batch)]
        idxs = list(range(args.start, args.start + args.batch))
        source = "MNIST %s[%d:%d]" % (args.split, args.start, args.start + args.batch)
    else:
        tb = load_from_tb_batch(args.batch, args.start)
        if tb is None:
            print("need MNIST under tools/data or lenet5_tb_batch.h", file=sys.stderr)
            return 1
        pixels, labs, idxs = tb
        source = "lenet5_tb_batch.h[%d:%d]" % (args.start, args.start + args.batch)

    lines = [
        "#ifndef LENET5_INPUT_H_",
        "#define LENET5_INPUT_H_",
        "",
        "/* %s — distributed LeNet batch */" % source,
        "",
        "#define LENET5_SPATIAL_PIPELINES %d" % spatial,
        "#define LENET5_BATCH_SIZE %d" % (1 if spatial > 0 else args.batch),
        "#define LENET5_TOTAL_SAMPLES %d" % args.batch,
        "#define LENET5_ENABLE_IC 0",
        "#define LENET5_ACCEL_UNIT 0",
        "#define CNN_TENSOR_DEBUG %d" % (0 if args.no_tensor_debug else 1),
        "",
    ]
    lines.extend(emit_array("LENET5_BATCH_SAMPLES", pixels))
    lines.append("")
    lines.append("static const int LENET5_BATCH_LABELS[%d] = { %s };" % (
        args.batch, ", ".join(str(int(x)) for x in labs)))
    lines.append("static const int LENET5_BATCH_INDICES[%d] = { %s };" % (
        args.batch, ", ".join(str(int(x)) for x in idxs)))
    lines.append("")
    lines.append("#define LENET5_INPUT LENET5_BATCH_SAMPLES")
    lines.append("#define LENET5_EXPECTED_CLASS %d" % int(labs[0]))
    lines.append("#define LENET5_SAMPLE_LABEL %d" % int(labs[0]))
    lines.append("")
    lines.append("#endif")
    lines.append("")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote %s (%d samples, labels=%s)" % (args.output, args.batch, labs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
