#!/usr/bin/env python3
"""
Export VGG16-lite Q8 assets for HEMPS (MNIST padded 32x32).

Reduced-channel VGG-style blocks (valid conv, PE-friendly):
  32x32x1
  B1: 5x5x8 + pool -> 14x14x8
  B2: 5x5x16 + pool -> 5x5x16
  B3: 3x3x24 -> 3x3x24 (=216)
  FC: 216->64->10
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import cnn_export_common as C

SEED = 11


def infer_q8(img32, w1, b1, w2, b2, w3, b3, wf1, bf1, wf2, bf2):
    import numpy as np

    x = img32.astype(np.int32).reshape(32, 32, 1)
    x = C.relu(C.conv2d_q8(x, w1, b1))
    b1o = C.avgpool2x2_q8(x)
    x = C.relu(C.conv2d_q8(b1o, w2, b2))
    b2o = C.avgpool2x2_q8(x)
    b3o = C.relu(C.conv2d_q8(b2o, w3, b3))
    flat = b3o.reshape(-1)
    h1 = C.relu(C.fc_q8_vec(flat, wf1, bf1))
    logits = C.fc_q8_vec(h1, wf2, bf2)
    return {
        "B1": b1o.reshape(-1).astype(int).tolist(),
        "B2": b2o.reshape(-1).astype(int).tolist(),
        "B3": flat.astype(int).tolist(),
        "FC1": h1.astype(int).tolist(),
        "FC2": logits.astype(int).tolist(),
        "pred": int(logits.argmax()),
    }


def extract_features(images32, w1, b1, w2, b2, w3, b3):
    import numpy as np

    n = len(images32)
    feats = np.zeros((n, 216), dtype=np.float32)
    for start in range(0, n, 64):
        end = min(start + 64, n)
        xb = images32[start:end].astype(np.int32).reshape(end - start, 32, 32, 1)
        c1 = np.maximum(0, C.conv2d_q8_batch(xb, w1, b1))
        p1 = C.avgpool2x2_batch(c1)
        c2 = np.maximum(0, C.conv2d_q8_batch(p1, w2, b2))
        p2 = C.avgpool2x2_batch(c2)
        c3 = np.maximum(0, C.conv2d_q8_batch(p2, w3, b3))
        feats[start:end] = c3.reshape(end - start, -1).astype(np.float32)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--train-limit", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--skip-ic-headers", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("numpy required", file=sys.stderr)
        return 1

    np.random.seed(SEED)
    train_images, train_labels = C.load_mnist("train")
    test_images, test_labels = C.load_mnist("test")
    train32 = C.pad_mnist_32(train_images)
    test32 = C.pad_mnist_32(test_images)
    n = min(args.train_limit, len(train32))

    w1 = C.q8_clip(np.random.randn(8, 1, 5, 5).astype(np.float32) * 34)
    b1 = C.q8_clip(np.zeros(8, dtype=np.float32))
    w2 = C.q8_clip(np.random.randn(16, 8, 5, 5).astype(np.float32) * 26)
    b2 = C.q8_clip(np.zeros(16, dtype=np.float32))
    w3 = C.q8_clip(np.random.randn(24, 16, 3, 3).astype(np.float32) * 22)
    b3 = C.q8_clip(np.zeros(24, dtype=np.float32))

    print("Extracting VGG16-lite Q8 features (%d) ..." % n)
    feats = extract_features(train32[:n], w1, b1, w2, b2, w3, b3) / 256.0
    print("Training FC ...")
    packed = C.train_fc_on_features(
        feats, train_labels[:n], [64, 10], args.epochs, args.lr, seed=SEED
    )
    wf1, bf1, wf2, bf2 = packed

    correct = 0
    n_q8 = min(50, len(test32))
    for i in range(n_q8):
        r = infer_q8(test32[i], w1, b1, w2, b2, w3, b3, wf1, bf1, wf2, bf2)
        if r["pred"] == int(test_labels[i]):
            correct += 1
    print("Q8 test_acc(%d)=%.4f" % (n_q8, correct / float(n_q8)))

    sample_idx = args.sample_idx or 0
    if args.sample_idx is None:
        for i in range(min(400, len(train32))):
            r = infer_q8(train32[i], w1, b1, w2, b2, w3, b3, wf1, bf1, wf2, bf2)
            if r["pred"] == int(train_labels[i]):
                sample_idx = i
                break

    img = train32[sample_idx]
    label = int(train_labels[sample_idx])
    result = infer_q8(img, w1, b1, w2, b2, w3, b3, wf1, bf1, wf2, bf2)
    print("golden idx=%d label=%d pred=%d" % (sample_idx, label, result["pred"]))

    C.write_weights_header(
        os.path.join(C.INCLUDE_DIR, "vgg16_weights.h"),
        "VGG16_WEIGHTS_H_",
        [
            ("VGG16_B1_WEIGHTS_WORDS", "VGG16_B1_WEIGHTS", C.pack_conv(w1, b1)),
            ("VGG16_B2_WEIGHTS_WORDS", "VGG16_B2_WEIGHTS", C.pack_conv(w2, b2)),
            ("VGG16_B3_WEIGHTS_WORDS", "VGG16_B3_WEIGHTS", C.pack_conv(w3, b3)),
            ("VGG16_FC1_WEIGHTS_WORDS", "VGG16_FC1_WEIGHTS", C.pack_fc(wf1, bf1)),
            ("VGG16_FC2_WEIGHTS_WORDS", "VGG16_FC2_WEIGHTS", C.pack_fc(wf2, bf2)),
        ],
    )
    C.write_input_header(
        os.path.join(C.INCLUDE_DIR, "vgg16_input.h"),
        "VGG16_INPUT_H_",
        "VGG16",
        [int(v) for v in img.reshape(-1).astype(int).tolist()],
        label,
        result["pred"],
        sample_idx,
    )
    golden = C.build_golden_json({
        "B1": result["B1"],
        "B2": result["B2"],
        "B3": result["B3"],
        "FC1": result["FC1"],
        "FC2": result["FC2"],
    })
    golden_path = os.path.join(C.DATA_DIR, "vgg16_golden.json")
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)
    print("Wrote %s" % golden_path)
    if not args.skip_ic_headers:
        C.run_gen_ic_headers(golden_path, "vgg16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
