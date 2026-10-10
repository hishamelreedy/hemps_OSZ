#!/usr/bin/env python3
"""
Train/export MLP784 (784-30-30-10) Q8 assets for HEMPS.
Optional ic_experiment slack merge via --bounds-json.
"""

import argparse
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

DEFAULT_IC_EXPERIMENT_BOUNDS = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "Hesham",
    "Experiments",
    "NoC_Model",
    "plasma_os",
    "software",
    "ic_experiment",
    "data",
    "golden_bounds_float_mlp.json",
)

MNIST_TRAIN_IMG = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
MNIST_TRAIN_LBL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
MNIST_TEST_IMG = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
MNIST_TEST_LBL = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"

SAMPLE_IDX = 0
H1, H2 = 30, 30
EPOCHS = 15
LR = 1.0
BATCH_SIZE = 64
SEED = 42
TRAIN_LIMIT = 10000

IC_LAYER_MAP = {
    "l1": "FC1",
    "l2": "FC2",
    "l3": "FC3",
    "fc1": "FC1",
    "fc2": "FC2",
    "fc3": "FC3",
}


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
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows * cols)
    labels = np.frombuffer(lraw, dtype=np.uint8, offset=8)
    return images.astype(np.float32), labels


def relu(x):
    import numpy as np

    return np.maximum(0, x)


def q8_clip(v):
    import numpy as np

    return np.clip(np.round(v), -128, 127).astype(np.int32)


def fc_q8_vec(x, w, b):
    """Match VHDL/hybrid: accumulate (xi*wi)>>8 per product."""
    import numpy as np

    in_dim = w.shape[0]
    out_dim = w.shape[1]
    out = np.zeros(out_dim, dtype=np.int32)
    xi = np.asarray(x, dtype=np.int32).reshape(-1)
    for o in range(out_dim):
        s = int(b[o])
        for i in range(in_dim):
            s += (int(xi[i]) * int(w[i, o])) >> 8
        out[o] = s
    return out


def infer_q8(x, w1, b1, w2, b2, w3, b3):
    h1 = relu(fc_q8_vec(x, w1, b1))
    h2 = relu(fc_q8_vec(h1, w2, b2))
    logits = fc_q8_vec(h2, w3, b3)
    return h1.astype(int).tolist(), h2.astype(int).tolist(), logits.astype(int).tolist(), int(logits.argmax())


def train_mlp(images, labels, epochs, lr, train_limit):
    import numpy as np

    np.random.seed(SEED)
    n = min(train_limit, len(images))
    x = images[:n] / 256.0
    y = labels[:n]

    w1 = np.random.randn(784, H1).astype(np.float32) * 0.05
    b1 = np.zeros(H1, dtype=np.float32)
    w2 = np.random.randn(H1, H2).astype(np.float32) * 0.05
    b2 = np.zeros(H2, dtype=np.float32)
    w3 = np.random.randn(H2, 10).astype(np.float32) * 0.05
    b3 = np.zeros(10, dtype=np.float32)

    for epoch in range(epochs):
        order = np.random.permutation(n)
        correct = 0
        for start in range(0, n, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            xb = x[idx]
            yb = y[idx]
            bs = len(idx)

            h1 = relu(xb @ w1 + b1)
            h2 = relu(h1 @ w2 + b2)
            logits = h2 @ w3 + b3
            correct += int(np.sum(np.argmax(logits, axis=1) == yb))

            t = np.zeros((bs, 10), dtype=np.float32)
            t[np.arange(bs), yb] = 1.0
            logits_c = np.clip(logits, -40.0, 40.0)
            p = 1.0 / (1.0 + np.exp(-logits_c))
            grad3 = (p - t) / bs
            gh2 = (grad3 @ w3.T) * (h2 > 0)
            gh1 = (gh2 @ w2.T) * (h1 > 0)

            w3 -= lr * (h2.T @ grad3)
            b3 -= lr * np.sum(grad3, axis=0)
            w2 -= lr * (h1.T @ gh2)
            b2 -= lr * np.sum(gh2, axis=0)
            w1 -= lr * (xb.T @ gh1)
            b1 -= lr * np.sum(gh1, axis=0)

        print("epoch %d train_acc=%.4f" % (epoch + 1, correct / n))

    return (
        q8_clip(w1 * 256.0),
        q8_clip(b1 * 256.0),
        q8_clip(w2 * 256.0),
        q8_clip(b2 * 256.0),
        q8_clip(w3 * 256.0),
        q8_clip(b3 * 256.0),
    )


def eval_q8_accuracy(images, labels, w1, b1, w2, b2, w3, b3, limit=None):
    n = len(images) if limit is None else min(limit, len(images))
    correct = 0
    for i in range(n):
        _, _, _, pred = infer_q8(images[i], w1, b1, w2, b2, w3, b3)
        if pred == int(labels[i]):
            correct += 1
    return correct / n


def stats(vec):
    return min(vec), max(vec), sum(vec) // len(vec)


def emit_array(name, values, per_line=8):
    lines = ["static const int %s[%d] = {" % (name, len(values))]
    for i in range(0, len(values), per_line):
        chunk = ", ".join(str(int(v)) for v in values[i : i + per_line])
        lines.append("    " + chunk + ",")
    lines.append("};")
    return lines


def load_ic_experiment_bounds(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    per_layer = data.get("per_layer", data.get("layers", {}))
    mapped = {}
    for key, layer in per_layer.items():
        if not isinstance(layer, dict):
            continue
        name = IC_LAYER_MAP.get(key.lower(), key.upper())
        if name not in ("FC1", "FC2", "FC3"):
            continue
        mapped[name] = {
            "avg_slack": int(layer.get("avg_slack", 10)),
            "slot_slack": int(layer.get("slot_slack", 2)),
            "safe_slot_mismatches": int(layer.get("safe_slot_mismatches", 0)),
        }
    return mapped


def build_golden_json(fc1, fc2, fc3, bounds_override=None):
    layers = {}
    for name, vec in (("FC1", fc1), ("FC2", fc2), ("FC3", fc3)):
        gmin, gmax, gavg = stats(vec)
        entry = {
            "golden_vector": vec,
            "gmin": gmin,
            "gmax": gmax,
            "gavg": gavg,
            "avg_slack": 3 if name != "FC3" else 4,
            "slot_slack": 2,
            "safe_slot_mismatches": 0,
        }
        if bounds_override and name in bounds_override:
            for key in ("avg_slack", "slot_slack", "safe_slot_mismatches"):
                if key in bounds_override[name]:
                    entry[key] = bounds_override[name][key]
        layers[name] = entry
    return {"layers": layers}


def run_gen_ic_headers(json_path):
    script = os.path.join(TOOLS_DIR, "gen_ic_headers.py")
    cmd = [sys.executable, script, json_path, "--network", "mlp784"]
    print("Running: %s" % " ".join(cmd))
    subprocess.check_call(cmd)


def select_batch_indices(images, labels, batch_size, split, start_idx):
    """Pick batch_size MNIST indices (sequential from split, or --batch-start)."""
    n = len(images)
    indices = []
    i = start_idx
    while len(indices) < batch_size and i < n:
        indices.append(i)
        i += 1
    if len(indices) < batch_size:
        raise RuntimeError(
            "need %d samples from %s starting at %d, only %d available"
            % (batch_size, split, start_idx, len(indices))
        )
    return indices


def main():
    parser = argparse.ArgumentParser(description="Export MLP784 Q8 assets for HEMPS")
    parser.add_argument("--sample-idx", type=int, default=None, help="MNIST train index (default: first correct Q8 pred)")
    parser.add_argument("--batch-size", type=int, default=1, help="Export N MNIST images for batch sim (default: 1)")
    parser.add_argument("--batch-split", choices=("train", "test"), default="test", help="MNIST split for batch export")
    parser.add_argument("--batch-start", type=int, default=0, help="First index in split for batch export")
    parser.add_argument(
        "--batch-use-accel", action="store_true",
        help="Set MLP784_BATCH_USE_ACCEL=1 in mlp784_input.h for HW FC in batch sim",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-limit", type=int, default=TRAIN_LIMIT)
    parser.add_argument("--bounds-json", default=None, help="ic_experiment golden_bounds JSON")
    parser.add_argument("--skip-ic-headers", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("numpy required: pip install numpy", file=sys.stderr)
        return 1

    bounds_path = args.bounds_json
    if bounds_path is None and os.path.isfile(DEFAULT_IC_EXPERIMENT_BOUNDS):
        bounds_path = DEFAULT_IC_EXPERIMENT_BOUNDS
        print("Found ic_experiment bounds: %s" % bounds_path)

    train_images, train_labels = load_mnist("train")
    test_images, test_labels = load_mnist("test")
    w1, b1, w2, b2, w3, b3 = train_mlp(
        train_images, train_labels, args.epochs, LR, args.train_limit
    )

    train_acc = eval_q8_accuracy(train_images, train_labels, w1, b1, w2, b2, w3, b3, limit=5000)
    test_acc = eval_q8_accuracy(test_images, test_labels, w1, b1, w2, b2, w3, b3)
    print("Q8 train_acc(5k)=%.4f test_acc=%.4f" % (train_acc, test_acc))

    sample_idx = args.sample_idx
    if sample_idx is None:
        # Prefer a non-zero, dense digit so prediction is not a sparse "always 0" artifact.
        best = None
        for i in range(min(5000, len(train_images))):
            lbl = int(train_labels[i])
            if lbl == 0:
                continue
            _, _, _, p = infer_q8(train_images[i], w1, b1, w2, b2, w3, b3)
            if p != lbl:
                continue
            nnz = int((train_images[i] != 0).sum())
            if nnz < 120:
                continue
            cand = (nnz, i, lbl, int(p))
            if best is None or cand[0] > best[0]:
                best = cand
            # Good enough: dense non-zero with correct Q8 pred
            if nnz >= 180:
                break
        if best is None:
            sample_idx = SAMPLE_IDX
            for i in range(min(1000, len(train_images))):
                _, _, _, p = infer_q8(train_images[i], w1, b1, w2, b2, w3, b3)
                if p == int(train_labels[i]):
                    sample_idx = i
                    break
            print("auto-selected fallback sample idx=%d" % sample_idx)
        else:
            sample_idx = best[1]
            print(
                "auto-selected dense non-zero sample idx=%d label=%d pred=%d nnz=%d"
                % (best[1], best[2], best[3], best[0])
            )

    batch_split_images = train_images if args.batch_split == "train" else test_images
    batch_split_labels = train_labels if args.batch_split == "train" else test_labels
    # Keep IC goldens aligned: batch sample 0 must be the golden when batch_size==1 on train.
    if max(1, args.batch_size) == 1 and args.batch_split == "train":
        batch_indices = [sample_idx]
    else:
        batch_indices = select_batch_indices(
            batch_split_images,
            batch_split_labels,
            max(1, args.batch_size),
            args.batch_split,
            args.batch_start,
        )

    batch_samples = []
    batch_labels = []
    batch_preds = []
    for idx in batch_indices:
        img = batch_split_images[idx]
        fc1_b, fc2_b, fc3_b, pred_b = infer_q8(img, w1, b1, w2, b2, w3, b3)
        lbl = int(batch_split_labels[idx])
        batch_samples.extend(img.astype(int).tolist())
        batch_labels.append(lbl)
        batch_preds.append(int(pred_b))
        print("batch[%d] mnist_%s idx=%d label=%d pred=%d %s" % (
            len(batch_labels) - 1,
            args.batch_split,
            idx,
            lbl,
            pred_b,
            "OK" if pred_b == lbl else "MISS",
        ))

    batch_correct = sum(1 for a, b in zip(batch_labels, batch_preds) if a == b)
    print("batch Q8 accuracy: %d/%d = %.4f" % (
        batch_correct, len(batch_labels), batch_correct / float(len(batch_labels)),
    ))

    sample = train_images[sample_idx]
    fc1, fc2, fc3, pred = infer_q8(sample, w1, b1, w2, b2, w3, b3)
    label = int(train_labels[sample_idx])
    print("golden sample idx=%d label=%d pred=%d" % (sample_idx, label, pred))

    bounds_override = None
    if bounds_path and os.path.isfile(bounds_path):
        bounds_override = load_ic_experiment_bounds(bounds_path)
        print("Merged ic_experiment slack bounds for: %s" % ", ".join(sorted(bounds_override.keys())))

    os.makedirs(DATA_DIR, exist_ok=True)
    golden = build_golden_json(fc1, fc2, fc3, bounds_override)
    json_path = os.path.join(DATA_DIR, "mlp784_golden.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)

    w1_blob = w1.reshape(-1).astype(int).tolist() + b1.astype(int).tolist()
    w2_blob = w2.reshape(-1).astype(int).tolist() + b2.astype(int).tolist()
    w3_blob = w3.reshape(-1).astype(int).tolist() + b3.astype(int).tolist()
    input_q8 = sample.astype(int).tolist()

    weights_h = [
        "#ifndef MLP784_WEIGHTS_H_",
        "#define MLP784_WEIGHTS_H_",
        "",
        "/* Q8 MLP784 weights exported by tools/export_mlp784_assets.py */",
        "",
    ]
    weights_h.extend(emit_array("MLP784_FC1_WEIGHTS", w1_blob))
    weights_h.append("")
    weights_h.extend(emit_array("MLP784_FC2_WEIGHTS", w2_blob))
    weights_h.append("")
    weights_h.extend(emit_array("MLP784_FC3_WEIGHTS", w3_blob))
    weights_h.append("")
    weights_h.append("#define MLP784_FC1_WEIGHTS_WORDS %d" % len(w1_blob))
    weights_h.append("#define MLP784_FC2_WEIGHTS_WORDS %d" % len(w2_blob))
    weights_h.append("#define MLP784_FC3_WEIGHTS_WORDS %d" % len(w3_blob))
    weights_h.append("")
    weights_h.append("#endif")
    weights_h.append("")

    batch_size = len(batch_indices)
    input_h = [
        "#ifndef MLP784_INPUT_H_",
        "#define MLP784_INPUT_H_",
        "",
        "/* Golden: MNIST train sample %d (label %d, pred %d) */" % (sample_idx, label, pred),
        "/* Batch: %d x MNIST %s samples starting at index %d */" % (
            batch_size, args.batch_split, args.batch_start,
        ),
        "",
        "#define MLP784_BATCH_SIZE %d" % batch_size,
        "#define MLP784_BATCH_SPLIT_%s 1" % args.batch_split.upper(),
    ]
    if batch_size > 1:
        input_h.append("/* IC goldens match sample 0 only; disabled when batch>1 */")
        input_h.append("#define MLP784_ENABLE_IC 0")
        if args.batch_use_accel:
            input_h.append("/* HW tensor_accel for batch (export --batch-use-accel) */")
            input_h.append("#define MLP784_BATCH_USE_ACCEL 1")
        else:
            input_h.append("#define MLP784_BATCH_USE_ACCEL 0")
    else:
        # Guarded: mlp784_niles_config.h may force IC off for the baseline arm.
        input_h.append("#ifndef MLP784_ENABLE_IC")
        input_h.append("#define MLP784_ENABLE_IC 1")
        input_h.append("#endif")
        input_h.append("#define MLP784_BATCH_USE_ACCEL 0")
        input_h.append("#define MLP784_ACCEL_UNIT 0")
        input_h.append("#define CNN_TENSOR_DEBUG 1")
    input_h.append("")
    input_h.extend(emit_array("MLP784_BATCH_SAMPLES", batch_samples))
    input_h.append("")
    input_h.extend(emit_array("MLP784_BATCH_LABELS", batch_labels))
    input_h.append("")
    input_h.extend(emit_array("MLP784_BATCH_INDICES", batch_indices))
    input_h.append("")
    input_h.extend(emit_array("MLP784_INPUT", input_q8))
    input_h.append("")
    input_h.append("#define MLP784_EXPECTED_CLASS %d" % pred)
    input_h.append("#define MLP784_SAMPLE_LABEL %d" % label)
    input_h.append("")
    input_h.append("#endif")
    input_h.append("")

    os.makedirs(INCLUDE_DIR, exist_ok=True)
    with open(os.path.join(INCLUDE_DIR, "mlp784_weights.h"), "w", encoding="utf-8") as f:
        f.write("\n".join(weights_h))
    with open(os.path.join(INCLUDE_DIR, "mlp784_input.h"), "w", encoding="utf-8") as f:
        f.write("\n".join(input_h))

    batch_json = {
        "batch_size": batch_size,
        "split": args.batch_split,
        "start_idx": args.batch_start,
        "indices": batch_indices,
        "labels": batch_labels,
        "offline_preds": batch_preds,
        "offline_correct": batch_correct,
        "batch_use_accel": bool(args.batch_use_accel) if batch_size > 1 else True,
    }
    batch_json_path = os.path.join(DATA_DIR, "mlp784_batch.json")
    with open(batch_json_path, "w", encoding="utf-8") as f:
        json.dump(batch_json, f, indent=2)

    print("Wrote mlp784_weights.h, mlp784_input.h, %s, %s" % (json_path, batch_json_path))

    if not args.skip_ic_headers:
        run_gen_ic_headers(json_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
