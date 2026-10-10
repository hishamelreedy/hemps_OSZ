#!/usr/bin/env python3
"""
Export AlexNet-mini Q8 assets for HEMPS (MNIST 28x28, PE-memory friendly).

Paper-aligned block names; channels reduced for 256KB pages:
  28x28x1
  -> C1 5x5x8 + pool -> 12x12x8
  -> C2 5x5x16 + pool -> 4x4x16
  -> C3 3x3x24 -> 2x2x24 (=96)
  -> FC1 96->64 -> FC2 64->10
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import cnn_export_common as C

SEED = 7


def infer_q8(img, wc1, bc1, wc2, bc2, wc3, bc3, w1, b1, w2, b2):
    import numpy as np

    x = img.astype(np.int32).reshape(28, 28, 1)
    c1 = C.relu(C.conv2d_q8(x, wc1, bc1))
    p1 = C.avgpool2x2_q8(c1)
    c2 = C.relu(C.conv2d_q8(p1, wc2, bc2))
    p2 = C.avgpool2x2_q8(c2)
    c3 = C.relu(C.conv2d_q8(p2, wc3, bc3))
    flat = c3.reshape(-1)
    h1 = C.relu(C.fc_q8_vec(flat, w1, b1))
    logits = C.fc_q8_vec(h1, w2, b2)
    return {
        "C1P": p1.reshape(-1).astype(int).tolist(),
        "C2P": p2.reshape(-1).astype(int).tolist(),
        "C3": flat.astype(int).tolist(),
        "FC1": h1.astype(int).tolist(),
        "FC2": logits.astype(int).tolist(),
        "pred": int(logits.argmax()),
    }


def extract_features(images, wc1, bc1, wc2, bc2, wc3, bc3):
    import numpy as np

    n = len(images)
    feats = np.zeros((n, 96), dtype=np.float32)
    for start in range(0, n, 64):
        end = min(start + 64, n)
        xb = images[start:end].astype(np.int32).reshape(end - start, 28, 28, 1)
        c1 = np.maximum(0, C.conv2d_q8_batch(xb, wc1, bc1))
        p1 = C.avgpool2x2_batch(c1)
        c2 = np.maximum(0, C.conv2d_q8_batch(p1, wc2, bc2))
        p2 = C.avgpool2x2_batch(c2)
        c3 = np.maximum(0, C.conv2d_q8_batch(p2, wc3, bc3))
        feats[start:end] = c3.reshape(end - start, -1).astype(np.float32)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--train-limit", type=int, default=10000)
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
    n = min(args.train_limit, len(train_images))

    wc1 = C.q8_clip(np.random.randn(8, 1, 5, 5).astype(np.float32) * 36)
    bc1 = C.q8_clip(np.zeros(8, dtype=np.float32))
    wc2 = C.q8_clip(np.random.randn(16, 8, 5, 5).astype(np.float32) * 28)
    bc2 = C.q8_clip(np.zeros(16, dtype=np.float32))
    wc3 = C.q8_clip(np.random.randn(24, 16, 3, 3).astype(np.float32) * 24)
    bc3 = C.q8_clip(np.zeros(24, dtype=np.float32))

    print("Extracting AlexNet-mini Q8 features (%d) ..." % n)
    feats = extract_features(train_images[:n], wc1, bc1, wc2, bc2, wc3, bc3) / 256.0
    print("Training FC ...")
    packed = C.train_fc_on_features(
        feats, train_labels[:n], [64, 10], args.epochs, args.lr, seed=SEED
    )
    w1, b1, w2, b2 = packed

    correct = 0
    n_q8 = min(60, len(test_images))
    for i in range(n_q8):
        r = infer_q8(test_images[i], wc1, bc1, wc2, bc2, wc3, bc3, w1, b1, w2, b2)
        if r["pred"] == int(test_labels[i]):
            correct += 1
    print("Q8 test_acc(%d)=%.4f" % (n_q8, correct / float(n_q8)))

    sample_idx = args.sample_idx or 0
    if args.sample_idx is None:
        for i in range(min(400, len(train_images))):
            r = infer_q8(train_images[i], wc1, bc1, wc2, bc2, wc3, bc3, w1, b1, w2, b2)
            if r["pred"] == int(train_labels[i]):
                sample_idx = i
                break

    img = train_images[sample_idx]
    label = int(train_labels[sample_idx])
    result = infer_q8(img, wc1, bc1, wc2, bc2, wc3, bc3, w1, b1, w2, b2)
    print("golden idx=%d label=%d pred=%d" % (sample_idx, label, result["pred"]))

    C.write_weights_header(
        os.path.join(C.INCLUDE_DIR, "alexnet_weights.h"),
        "ALEXNET_WEIGHTS_H_",
        [
            ("ALEXNET_C1_WEIGHTS_WORDS", "ALEXNET_C1_WEIGHTS", C.pack_conv(wc1, bc1)),
            ("ALEXNET_C2_WEIGHTS_WORDS", "ALEXNET_C2_WEIGHTS", C.pack_conv(wc2, bc2)),
            ("ALEXNET_C3_WEIGHTS_WORDS", "ALEXNET_C3_WEIGHTS", C.pack_conv(wc3, bc3)),
            ("ALEXNET_FC1_WEIGHTS_WORDS", "ALEXNET_FC1_WEIGHTS", C.pack_fc(w1, b1)),
            ("ALEXNET_FC2_WEIGHTS_WORDS", "ALEXNET_FC2_WEIGHTS", C.pack_fc(w2, b2)),
        ],
    )
    C.write_input_header(
        os.path.join(C.INCLUDE_DIR, "alexnet_input.h"),
        "ALEXNET_INPUT_H_",
        "ALEXNET",
        [int(v) for v in img.reshape(-1).astype(int).tolist()],
        label,
        result["pred"],
        sample_idx,
    )
    golden = C.build_golden_json({
        "C1P": result["C1P"],
        "C2P": result["C2P"],
        "C3": result["C3"],
        "FC1": result["FC1"],
        "FC2": result["FC2"],
    })
    golden_path = os.path.join(C.DATA_DIR, "alexnet_golden.json")
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)
    print("Wrote %s" % golden_path)
    if not args.skip_ic_headers:
        C.run_gen_ic_headers(golden_path, "alexnet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
