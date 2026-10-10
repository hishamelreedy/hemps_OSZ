#!/usr/bin/env python3
"""Export LeNet-5 Q8 assets for HEMPS (uses cnn_export_common).

Conv filters are hand-scaled so Q8 >>8 does not kill C2/P2 (previous random
C2 init produced all-zero P2 and constant class-0 logits).
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import cnn_export_common as C

SEED = 42


def init_c1(cout=6, k=5, scale=48):
    """Edge + blob detectors on single-channel MNIST."""
    import numpy as np

    np.random.seed(SEED)
    w = np.zeros((cout, 1, k, k), dtype=np.float32)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    blur = np.ones((3, 3), dtype=np.float32)
    patterns = [sobel_x, sobel_y, lap, blur, -sobel_x, -sobel_y]
    for i in range(cout):
        ker = patterns[i % len(patterns)]
        w[i, 0, 1:4, 1:4] = ker * scale
        # small noise so filters are not identical after clip
        w[i] += np.random.randn(1, k, k).astype(np.float32) * 2.0
    b = np.zeros(cout, dtype=np.float32)
    return C.q8_clip(w), C.q8_clip(b)


def init_c2(cout=16, cin=6, k=5, scale=36):
    """
    Positive mixing filters: each output channel averages a few input channels.
    Random signed Q8 C2 previously underflowed to all-zero after ReLU+pool.
    """
    import numpy as np

    np.random.seed(SEED + 1)
    w = np.zeros((cout, cin, k, k), dtype=np.float32)
    # 3x3 box in the center, positive
    box = np.ones((3, 3), dtype=np.float32) * (scale / 9.0)
    for oc in range(cout):
        # cycle which input channels this filter looks at
        ics = [oc % cin, (oc + 1) % cin, (oc + 2) % cin]
        for ic in ics:
            w[oc, ic, 1:4, 1:4] += box
        # slight variation
        w[oc] += np.random.randn(cin, k, k).astype(np.float32) * 1.5
    b = np.full(cout, 4.0, dtype=np.float32)  # small positive bias
    return C.q8_clip(w), C.q8_clip(b)


def extract_features(images, wc1, bc1, wc2, bc2):
    import numpy as np

    n = len(images)
    feats = np.zeros((n, 256), dtype=np.float32)
    for start in range(0, n, 128):
        end = min(start + 128, n)
        xb = images[start:end].astype(np.int32).reshape(end - start, 28, 28, 1)
        c1 = np.maximum(0, C.conv2d_q8_batch(xb, wc1, bc1))
        p1 = C.avgpool2x2_batch(c1)
        c2 = np.maximum(0, C.conv2d_q8_batch(p1, wc2, bc2))
        p2 = C.avgpool2x2_batch(c2)
        feats[start:end] = p2.reshape(end - start, -1).astype(np.float32)
    return feats


def fc_q8_np(x, w, b):
    """Bit-exact-ish vectorized Q8 FC: sum((x*w)>>8)+b (matches cnn_export_common loops)."""
    import numpy as np

    xi = np.asarray(x, dtype=np.int64).reshape(-1)
    wi = np.asarray(w, dtype=np.int64)
    # (in, out) : sum_i (xi[i]*wi[i,o]) >> 8
    prod = (xi[:, None] * wi) >> 8
    return (prod.sum(axis=0) + np.asarray(b, dtype=np.int64)).astype(np.int32)


def fc_q8_forward(feat_i, w1, b1, w2, b2, w3, b3):
    import numpy as np

    h1 = np.maximum(0, fc_q8_np(feat_i, w1, b1))
    h2 = np.maximum(0, fc_q8_np(h1, w2, b2))
    logits = fc_q8_np(h2, w3, b3)
    return int(np.argmax(logits))


def train_fc_q8_tracked(feats_i, labels, sizes, epochs, lr, seed=42):
    """
    Train float FC on feats/256, pack to Q8 each epoch, keep best Q8 weights
    by accuracy on a Q8 forward pass with integer features.
    """
    import numpy as np

    np.random.seed(seed)
    n, d = feats_i.shape
    feats_f = feats_i.astype(np.float32) / 256.0
    layers_w = []
    layers_b = []
    prev = d
    for out_dim in sizes:
        layers_w.append(np.random.randn(prev, out_dim).astype(np.float32) * 0.05)
        layers_b.append(np.zeros(out_dim, dtype=np.float32))
        prev = out_dim

    bs = 64
    best_acc = -1.0
    best_packed = None
    eval_n = min(800, n)

    for epoch in range(epochs):
        order = np.random.permutation(n)
        correct = 0
        diverged = False
        for start in range(0, n, bs):
            idx = order[start : start + bs]
            x = feats_f[idx]
            yb = labels[idx]
            bsz = len(idx)
            acts = [x]
            h = x
            for i, (w, b) in enumerate(zip(layers_w, layers_b)):
                z = h @ w + b
                if i < len(layers_w) - 1:
                    h = C.relu(z)
                else:
                    h = z
                acts.append(h)

            logits = acts[-1]
            if not np.isfinite(logits).all():
                diverged = True
                break
            correct += int(np.sum(np.argmax(logits, axis=1) == yb))

            t = np.zeros((bsz, sizes[-1]), dtype=np.float32)
            t[np.arange(bsz), yb] = 1.0
            m = np.max(logits, axis=1, keepdims=True)
            e = np.exp(np.clip(logits - m, -40.0, 40.0))
            p = e / (np.sum(e, axis=1, keepdims=True) + 1e-8)
            g = (p - t) / bsz

            for i in range(len(layers_w) - 1, -1, -1):
                w = layers_w[i]
                prev_act = acts[i]
                dw = np.clip(prev_act.T @ g, -1.0, 1.0)
                db = np.clip(np.sum(g, axis=0), -1.0, 1.0)
                layers_w[i] = np.clip(w - lr * dw, -4.0, 4.0).astype(np.float32)
                layers_b[i] = np.clip(layers_b[i] - lr * db, -4.0, 4.0).astype(np.float32)
                if i > 0:
                    g = np.clip((g @ w.T) * (acts[i] > 0), -1.0, 1.0)

        if diverged:
            print("  fc epoch %d DIVERGED" % (epoch + 1))
            break

        packed = []
        q_ws, q_bs = [], []
        for w, b in zip(layers_w, layers_b):
            qw = C.q8_clip(w * 256.0)
            qb = C.q8_clip(b * 256.0)
            q_ws.append(qw)
            q_bs.append(qb)
            packed.extend([qw, qb])

        q_ok = 0
        for i in range(eval_n):
            if fc_q8_forward(feats_i[i], q_ws[0], q_bs[0], q_ws[1], q_bs[1], q_ws[2], q_bs[2]) == int(labels[i]):
                q_ok += 1
        q_acc = q_ok / float(eval_n)
        print("  fc epoch %d float_acc=%.4f  q8_acc(%d)=%.4f" % (
            epoch + 1, correct / float(n), eval_n, q_acc))
        if q_acc >= best_acc:
            best_acc = q_acc
            best_packed = packed

    if best_packed is None:
        raise RuntimeError("FC training produced no Q8 checkpoint")
    print("  best q8_acc=%.4f" % best_acc)
    return best_packed


def infer_q8(img, wc1, bc1, wc2, bc2, w1, b1, w2, b2, w3, b3):
    import numpy as np

    x = img.astype(np.int32).reshape(28, 28, 1)
    c1 = C.relu(C.conv2d_q8(x, wc1, bc1))
    p1 = C.avgpool2x2_q8(c1)
    c2 = C.relu(C.conv2d_q8(p1, wc2, bc2))
    p2 = C.avgpool2x2_q8(c2)
    flat = p2.reshape(-1)
    h1 = np.maximum(0, fc_q8_np(flat, w1, b1))
    h2 = np.maximum(0, fc_q8_np(h1, w2, b2))
    logits = fc_q8_np(h2, w3, b3)
    return {
        "C1P1": p1.reshape(-1).astype(int).tolist(),
        "C2P2": flat.astype(int).tolist(),
        "FC1": h1.astype(int).tolist(),
        "FC2": h2.astype(int).tolist(),
        "FC3": logits.astype(int).tolist(),
        "pred": int(logits.argmax()),
        "p2_max": int(flat.max()),
        "p2_sum": int(flat.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-limit", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--skip-ic-headers", action="store_true")
    parser.add_argument("--min-test-acc", type=float, default=0.50)
    args = parser.parse_args()

    try:
        import numpy as np  # noqa
    except ImportError:
        print("numpy required", file=sys.stderr)
        return 1

    train_images, train_labels = C.load_mnist("train")
    test_images, test_labels = C.load_mnist("test")
    n = min(args.train_limit, len(train_images))

    print("Init LeNet conv filters (C2 energy-preserving) ...")
    wc1, bc1 = init_c1(6, 5, scale=48)
    wc2, bc2 = init_c2(16, 6, 5, scale=40)

    print("Extracting Q8 features (%d) ..." % n)
    feats_raw = extract_features(train_images[:n], wc1, bc1, wc2, bc2)
    print("  feature max=%.1f mean=%.2f nonzero_frac=%.3f" % (
        float(feats_raw.max()),
        float(feats_raw.mean()),
        float((feats_raw != 0).mean()),
    ))
    if float(feats_raw.max()) <= 0:
        print("ERROR: P2 features are all zero — aborting export", file=sys.stderr)
        return 1

    # Same convention as MLP784: train on x/256, pack w*256 for (x*w)>>8 inference.
    feats = feats_raw.astype(np.float32) / 256.0

    print("Training FC (track best Q8 accuracy) ...")
    packed = train_fc_q8_tracked(
        feats_raw, train_labels[:n], [120, 84, 10], args.epochs, args.lr, seed=SEED
    )
    w1, b1, w2, b2, w3, b3 = packed

    correct = 0
    n_q8 = min(200, len(test_images))
    pred_hist = [0] * 10
    for i in range(n_q8):
        r = infer_q8(test_images[i], wc1, bc1, wc2, bc2, w1, b1, w2, b2, w3, b3)
        pred_hist[r["pred"]] += 1
        if r["pred"] == int(test_labels[i]):
            correct += 1
    acc = correct / float(n_q8)
    print("Q8 test_acc(%d)=%.4f  pred_hist=%s" % (n_q8, acc, pred_hist))
    if acc < args.min_test_acc:
        print("ERROR: test accuracy %.3f < min %.3f" % (acc, args.min_test_acc),
              file=sys.stderr)
        return 1
    if pred_hist.count(0) >= 9 and pred_hist[0] == n_q8:
        print("ERROR: model collapsed to always predict 0", file=sys.stderr)
        return 1

    # Pick golden: correct pred, non-zero P2, prefer non-zero label if possible
    sample_idx = args.sample_idx
    if sample_idx is None:
        sample_idx = None
        for prefer_nonzero_label in (True, False):
            for i in range(min(2000, len(train_images))):
                r = infer_q8(train_images[i], wc1, bc1, wc2, bc2, w1, b1, w2, b2, w3, b3)
                lab = int(train_labels[i])
                if r["pred"] != lab or r["p2_max"] <= 0:
                    continue
                if prefer_nonzero_label and lab == 0:
                    continue
                sample_idx = i
                break
            if sample_idx is not None:
                break
        if sample_idx is None:
            print("ERROR: no suitable golden sample found", file=sys.stderr)
            return 1

    img = train_images[sample_idx]
    label = int(train_labels[sample_idx])
    result = infer_q8(img, wc1, bc1, wc2, bc2, w1, b1, w2, b2, w3, b3)
    print("golden idx=%d label=%d pred=%d p2_max=%d p2_sum=%d" % (
        sample_idx, label, result["pred"], result["p2_max"], result["p2_sum"]))
    if result["pred"] != label or result["p2_max"] <= 0:
        print("ERROR: golden sample invalid", file=sys.stderr)
        return 1

    C.write_weights_header(
        os.path.join(C.INCLUDE_DIR, "lenet5_weights.h"),
        "LENET5_WEIGHTS_H_",
        [
            ("LENET5_C1_WEIGHTS_WORDS", "LENET5_C1_WEIGHTS", C.pack_conv(wc1, bc1)),
            ("LENET5_C2_WEIGHTS_WORDS", "LENET5_C2_WEIGHTS", C.pack_conv(wc2, bc2)),
            ("LENET5_FC1_WEIGHTS_WORDS", "LENET5_FC1_WEIGHTS", C.pack_fc(w1, b1)),
            ("LENET5_FC2_WEIGHTS_WORDS", "LENET5_FC2_WEIGHTS", C.pack_fc(w2, b2)),
            ("LENET5_FC3_WEIGHTS_WORDS", "LENET5_FC3_WEIGHTS", C.pack_fc(w3, b3)),
        ],
    )
    pixels = [int(v) for v in img.reshape(-1).astype(int).tolist()]
    C.write_input_header(
        os.path.join(C.INCLUDE_DIR, "lenet5_input.h"),
        "LENET5_INPUT_H_",
        "LENET5",
        pixels,
        label,
        result["pred"],
        sample_idx,
    )

    golden = C.build_golden_json({
        "C1P1": result["C1P1"],
        "C2P2": result["C2P2"],
        "FC1": result["FC1"],
        "FC2": result["FC2"],
        "FC3": result["FC3"],
    })
    golden_path = os.path.join(C.DATA_DIR, "lenet5_golden.json")
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)
    print("Wrote %s" % golden_path)
    if not args.skip_ic_headers:
        C.run_gen_ic_headers(golden_path, "lenet5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
