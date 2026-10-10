#!/usr/bin/env python3
"""Shared helpers for HEMPS CNN Q8 asset export (LeNet / AlexNet / VGG)."""

from __future__ import print_function

import gzip
import json
import os
import struct
import subprocess
import sys
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INCLUDE_DIR = os.path.join(REPO, "software", "include")
TOOLS_DIR = os.path.dirname(__file__)

MNIST_TRAIN_IMG = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
MNIST_TRAIN_LBL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
MNIST_TEST_IMG = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
MNIST_TEST_LBL = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"


def fetch_gz(url, dest):
    if not os.path.isfile(dest):
        print("Downloading %s ..." % url)
        urllib.request.urlretrieve(url, dest)


def load_mnist(split="train"):
    import numpy as np

    os.makedirs(DATA_DIR, exist_ok=True)
    if split == "train":
        img_url, lbl_url = MNIST_TRAIN_IMG, MNIST_TRAIN_LBL
        img_name, lbl_name = "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
    else:
        img_url, lbl_url = MNIST_TEST_IMG, MNIST_TEST_LBL
        img_name, lbl_name = "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"

    img_path = os.path.join(DATA_DIR, img_name)
    lbl_path = os.path.join(DATA_DIR, lbl_name)
    fetch_gz(img_url, img_path)
    fetch_gz(lbl_url, lbl_path)

    with gzip.open(img_path, "rb") as f:
        raw = f.read()
    with gzip.open(lbl_path, "rb") as f:
        lraw = f.read()

    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise RuntimeError("bad MNIST magic %d" % magic)
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows, cols)
    labels = np.frombuffer(lraw, dtype=np.uint8, offset=8)
    return images.astype(np.float32), labels


def pad_mnist_32(images):
    """Center 28x28 into 32x32."""
    import numpy as np

    n = len(images)
    out = np.zeros((n, 32, 32), dtype=np.float32)
    out[:, 2:30, 2:30] = images
    return out


def relu(x):
    import numpy as np

    return np.maximum(0, x)


def q8_clip(v):
    import numpy as np

    return np.clip(np.round(v), -128, 127).astype(np.int32)


def emit_array(name, values, per_line=8):
    lines = ["static const int %s[%d] = {" % (name, len(values))]
    for i in range(0, len(values), per_line):
        chunk = ", ".join(str(int(v)) for v in values[i : i + per_line])
        lines.append("    " + chunk + ",")
    lines.append("};")
    return lines


def stats(vec):
    return min(vec), max(vec), sum(vec) // len(vec)


def build_golden_json(layers_dict, conv_slack=True):
    layers = {}
    for name, vec in layers_dict.items():
        gmin, gmax, gavg = stats(vec)
        is_conv = conv_slack and (name.startswith("C") or name.startswith("B") or name.startswith("P"))
        layers[name] = {
            "golden_vector": vec,
            "gmin": gmin,
            "gmax": gmax,
            "gavg": gavg,
            "avg_slack": 8 if is_conv else 4,
            "slot_slack": 4 if is_conv else 2,
            "safe_slot_mismatches": 2 if is_conv else 0,
        }
    return {"layers": layers}


def run_gen_ic_headers(json_path, network):
    script = os.path.join(TOOLS_DIR, "gen_ic_headers.py")
    cmd = [sys.executable, script, json_path, "--network", network]
    print("Running: %s" % " ".join(cmd))
    subprocess.check_call(cmd)


def pack_conv(w, b):
    cout, cin, k, _ = w.shape
    out = []
    for oc in range(cout):
        for ic in range(cin):
            for ky in range(k):
                for kx in range(k):
                    out.append(int(w[oc, ic, ky, kx]))
    for oc in range(cout):
        out.append(int(b[oc]))
    return out


def pack_fc(w, b):
    in_dim, out_dim = w.shape
    out = []
    for i in range(in_dim):
        for o in range(out_dim):
            out.append(int(w[i, o]))
    for o in range(out_dim):
        out.append(int(b[o]))
    return out


def conv2d_q8(x, w, b):
    import numpy as np

    h, ww, cin = x.shape
    cout, _, k, _ = w.shape
    ho, wo = h - k + 1, ww - k + 1
    out = np.zeros((ho, wo, cout), dtype=np.int32)
    xi = x.astype(np.int32)
    wi = w.astype(np.int32)
    bi = b.astype(np.int32)
    for oc in range(cout):
        for oy in range(ho):
            for ox in range(wo):
                s = int(bi[oc])
                for ic in range(cin):
                    for ky in range(k):
                        for kx in range(k):
                            s += (int(xi[oy + ky, ox + kx, ic]) * int(wi[oc, ic, ky, kx])) >> 8
                out[oy, ox, oc] = s
    return out


def conv2d_q8_batch(x, w, b):
    import numpy as np

    n, h, ww, cin = x.shape
    cout, _, k, _ = w.shape
    ho, wo = h - k + 1, ww - k + 1
    out = np.zeros((n, ho, wo, cout), dtype=np.int64)
    xi = x.astype(np.int64)
    wi = w.astype(np.int64)
    for ky in range(k):
        for kx in range(k):
            patch = xi[:, ky : ky + ho, kx : kx + wo, :]
            for oc in range(cout):
                prod = patch * wi[oc, :, ky, kx].reshape(1, 1, 1, cin)
                out[:, :, :, oc] += (prod >> 8).sum(axis=-1)
    out += b.astype(np.int64).reshape(1, 1, 1, -1)
    return out.astype(np.int32)


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


def avgpool2x2_batch(x):
    import numpy as np

    n, h, w, c = x.shape
    x = x.reshape(n, h // 2, 2, w // 2, 2, c)
    return (x.sum(axis=(2, 4)) // 4).astype(np.int32)


def fc_q8_vec(x, w, b):
    import numpy as np

    in_dim, out_dim = w.shape
    out = np.zeros(out_dim, dtype=np.int32)
    xi = np.asarray(x, dtype=np.int32).reshape(-1)
    for o in range(out_dim):
        s = int(b[o])
        for i in range(in_dim):
            s += (int(xi[i]) * int(w[i, o])) >> 8
        # saturate to int32 range for pathological weights
        if s > 2147483647:
            s = 2147483647
        elif s < -2147483648:
            s = -2147483648
        out[o] = s
    return out


def train_fc_on_features(feats, labels, sizes, epochs, lr, seed=42):
    """feats (N,D) float already scaled; sizes = [h1, h2, ..., nclass] or [h1, nclass]."""
    import numpy as np

    np.random.seed(seed)
    n, d = feats.shape
    layers_w = []
    layers_b = []
    prev = d
    for out_dim in sizes:
        layers_w.append(np.random.randn(prev, out_dim).astype(np.float32) * 0.05)
        layers_b.append(np.zeros(out_dim, dtype=np.float32))
        prev = out_dim

    bs = 64
    best_acc = -1.0
    best_w = None
    best_b = None
    for epoch in range(epochs):
        order = np.random.permutation(n)
        correct = 0
        diverged = False
        for start in range(0, n, bs):
            idx = order[start : start + bs]
            x = feats[idx]
            yb = labels[idx]
            bsz = len(idx)
            acts = [x]
            h = x
            for i, (w, b) in enumerate(zip(layers_w, layers_b)):
                z = h @ w + b
                if i < len(layers_w) - 1:
                    h = relu(z)
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
            # softmax cross-entropy grad
            m = np.max(logits, axis=1, keepdims=True)
            e = np.exp(np.clip(logits - m, -40.0, 40.0))
            p = e / (np.sum(e, axis=1, keepdims=True) + 1e-8)
            g = (p - t) / bsz

            for i in range(len(layers_w) - 1, -1, -1):
                w = layers_w[i]
                prev_act = acts[i]
                dw = prev_act.T @ g
                db = np.sum(g, axis=0)
                # stabilize updates
                dw = np.clip(dw, -1.0, 1.0)
                db = np.clip(db, -1.0, 1.0)
                layers_w[i] = np.clip(w - lr * dw, -8.0, 8.0).astype(np.float32)
                layers_b[i] = np.clip(layers_b[i] - lr * db, -8.0, 8.0).astype(np.float32)
                if i > 0:
                    g = (g @ w.T) * (acts[i] > 0)
                    g = np.clip(g, -1.0, 1.0)

        if diverged:
            print("  fc epoch %d DIVERGED — keeping best checkpoint" % (epoch + 1))
            break

        acc = correct / float(n)
        print("  fc epoch %d train_acc=%.4f" % (epoch + 1, acc))
        if acc >= best_acc:
            best_acc = acc
            best_w = [w.copy() for w in layers_w]
            best_b = [b.copy() for b in layers_b]

    if best_w is None:
        best_w, best_b = layers_w, layers_b
    else:
        layers_w, layers_b = best_w, best_b
        print("  using best train_acc=%.4f" % best_acc)

    out = []
    for w, b in zip(layers_w, layers_b):
        out.append(q8_clip(w * 256.0))
        out.append(q8_clip(b * 256.0))
    return out


def write_weights_header(path, guard, macros_and_arrays):
    """macros_and_arrays: list of (macro_name, array_name, values) or (None, array_name, values)."""
    lines = ["#ifndef %s" % guard, "#define %s" % guard, ""]
    for item in macros_and_arrays:
        if len(item) == 3 and item[0]:
            lines.append("#define %s %d" % (item[0], len(item[2])))
    lines.append("")
    for item in macros_and_arrays:
        name = item[1]
        vals = item[2]
        lines.extend(emit_array(name, vals))
        lines.append("")
    lines.append("#endif")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote %s" % path)


def write_input_header(path, guard, prefix, pixels, label, pred, sample_idx):
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        "/* MNIST sample %d (label %d, pred %d) */" % (sample_idx, label, pred),
        "",
        "#define %s_BATCH_SIZE 1" % prefix,
        "#define %s_ENABLE_IC 1" % prefix,
        "",
    ]
    lines.extend(emit_array("%s_BATCH_SAMPLES" % prefix, pixels))
    lines.append("")
    lines.append("static const int %s_BATCH_LABELS[1] = { %d };" % (prefix, label))
    lines.append("static const int %s_BATCH_INDICES[1] = { %d };" % (prefix, sample_idx))
    lines.append("")
    lines.append("#define %s_INPUT %s_BATCH_SAMPLES" % (prefix, prefix))
    lines.append("#define %s_EXPECTED_CLASS %d" % (prefix, pred))
    lines.append("#define %s_SAMPLE_LABEL %d" % (prefix, label))
    lines.append("")
    lines.append("#endif")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote %s" % path)
