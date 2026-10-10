#!/usr/bin/env python3
"""Offline bit-exact check: SW cnn_* vs HW tensor_accel behavioral models.

Covers FC (MLP784), CONV2D, AVGPOOL2X2, RELU matching software/include/cnn_*.h
and hardware/sc/pe/tensor_accel/tensor_accel.cpp.
"""

from __future__ import print_function

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCLUDE = os.path.join(REPO, "software", "include")


def relu(x):
    return x if x > 0 else 0


def fc_blob(input_vec, weights, in_dim, out_dim, apply_relu):
    bias = weights[in_dim * out_dim :]
    out = []
    for o in range(out_dim):
        s = bias[o]
        for i in range(in_dim):
            s += (input_vec[i] * weights[i * out_dim + o]) >> 8
        out.append(relu(s) if apply_relu else s)
    return out


def conv2d_q8(inp, weights, in_h, in_w, in_c, out_c, k, apply_relu=False):
    """Matches cnn_conv2d_q8 (+ optional ReLU like accel CONV with dims1 bit24)."""
    out_h = in_h - k + 1
    out_w = in_w - k + 1
    w_per_oc = in_c * k * k
    bias = weights[out_c * w_per_oc :]
    out = [0] * (out_h * out_w * out_c)
    for oc in range(out_c):
        w_oc = weights[oc * w_per_oc : (oc + 1) * w_per_oc]
        for oy in range(out_h):
            for ox in range(out_w):
                s = bias[oc]
                for ic in range(in_c):
                    for ky in range(k):
                        for kx in range(k):
                            xi = inp[((oy + ky) * in_w + (ox + kx)) * in_c + ic]
                            wi = w_oc[ic * (k * k) + ky * k + kx]
                            s += (xi * wi) >> 8
                v = relu(s) if apply_relu else s
                out[(oy * out_w + ox) * out_c + oc] = v
    return out


def trunc_div(a, b):
    """Match C signed integer division (truncate toward zero)."""
    return int(a / float(b))


def avgpool2x2_q8(inp, in_h, in_w, channels):
    out_h = in_h // 2
    out_w = in_w // 2
    out = [0] * (out_h * out_w * channels)
    for oy in range(out_h):
        for ox in range(out_w):
            for c in range(channels):
                v00 = inp[((oy * 2) * in_w + (ox * 2)) * channels + c]
                v01 = inp[((oy * 2) * in_w + (ox * 2 + 1)) * channels + c]
                v10 = inp[((oy * 2 + 1) * in_w + (ox * 2)) * channels + c]
                v11 = inp[((oy * 2 + 1) * in_w + (ox * 2 + 1)) * channels + c]
                out[(oy * out_w + ox) * channels + c] = trunc_div(v00 + v01 + v10 + v11, 4)
    return out


def relu_n(inp):
    return [relu(x) for x in inp]


def fc_hw(input_vec, weights, in_dim, out_dim, apply_relu):
    """Byte-address form matching tensor_accel.cpp run_fc_behavioral."""
    bias_off = in_dim * out_dim
    out = []
    for o in range(out_dim):
        s = weights[bias_off + o]
        for i in range(in_dim):
            x = input_vec[i]
            w = weights[i * out_dim + o]
            s += (x * w) >> 8
        out.append(relu(s) if apply_relu else s)
    return out


def conv_hw(inp, weights, in_h, in_w, in_c, out_c, k, apply_relu=False):
    """Byte-address form matching tensor_accel.cpp run_conv2d_behavioral."""
    out_h = in_h - k + 1
    out_w = in_w - k + 1
    w_per_oc = in_c * k * k
    bias_off = out_c * w_per_oc
    out = [0] * (out_h * out_w * out_c)
    for oc in range(out_c):
        bias = weights[bias_off + oc]
        w_oc = oc * w_per_oc
        for oy in range(out_h):
            for ox in range(out_w):
                s = bias
                for ic in range(in_c):
                    for ky in range(k):
                        for kx in range(k):
                            in_idx = ((oy + ky) * in_w + (ox + kx)) * in_c + ic
                            w_idx = w_oc + ic * (k * k) + ky * k + kx
                            s += (inp[in_idx] * weights[w_idx]) >> 8
                v = relu(s) if apply_relu else s
                out[(oy * out_w + ox) * out_c + oc] = v
    return out


def pool_hw(inp, in_h, in_w, channels):
    return avgpool2x2_q8(inp, in_h, in_w, channels)


def relu_hw(inp):
    return relu_n(inp)


def parse_int_array_from_header(path, symbol):
    text = open(path, "r", encoding="utf-8").read()
    marker = "static const int %s[" % symbol
    start = text.find(marker)
    if start < 0:
        raise RuntimeError("symbol %s not found in %s" % (symbol, path))
    brace = text.find("{", start)
    end = text.find("};", brace)
    body = text[brace + 1 : end]
    vals = []
    for part in body.replace("\n", " ").split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals


def parse_define(path, name, default=None):
    for line in open(path, encoding="utf-8"):
        if line.startswith("#define %s" % name):
            return int(line.split()[-1])
    return default


def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b)) if a else 0


def check_vec(name, sw, hw):
    diff = max_abs_diff(sw, hw)
    ok = diff == 0 and len(sw) == len(hw)
    print("  %s: n=%d max_diff=%d %s" % (name, len(sw), diff, "OK" if ok else "FAIL"))
    if not ok:
        for i, (a, b) in enumerate(zip(sw, hw)):
            if a != b:
                print("    first mismatch i=%d sw=%d hw=%d" % (i, a, b))
                break
    return ok


def argmax(vec):
    return max(range(len(vec)), key=lambda i: vec[i])


def checksum(vec):
    return sum(vec)


def test_tiny_ops():
    ok = True
    print("tiny fixture: FC / CONV / POOL / RELU")

    tiny_in = [1, 2, 3, 4]
    tiny_w = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ok &= check_vec("FC", fc_blob(tiny_in, tiny_w, 4, 2, True),
                    fc_hw(tiny_in, tiny_w, 4, 2, True))

    # 4x4x1 -> 2x2x2, k=3
    in_h, in_w, in_c, out_c, k = 4, 4, 1, 2, 3
    n_in = in_h * in_w * in_c
    conv_in = [((i * 17) % 251) - 100 for i in range(n_in)]
    w_per = in_c * k * k
    conv_w = [((i * 13) % 127) - 60 for i in range(out_c * w_per)]
    conv_w += [i * 10 for i in range(out_c)]
    for apply_relu in (False, True):
        sw = conv2d_q8(conv_in, conv_w, in_h, in_w, in_c, out_c, k, apply_relu)
        hw = conv_hw(conv_in, conv_w, in_h, in_w, in_c, out_c, k, apply_relu)
        ok &= check_vec("CONV2D relu=%d" % int(apply_relu), sw, hw)

    pool_in = [((i * 19) % 200) - 80 for i in range(4 * 4 * 3)]
    ok &= check_vec("AVGPOOL2X2", avgpool2x2_q8(pool_in, 4, 4, 3),
                    pool_hw(pool_in, 4, 4, 3))

    relu_in = [-5, 0, 3, -1, 100]
    ok &= check_vec("RELU", relu_n(relu_in), relu_hw(relu_in))
    return ok


def test_mlp784(sample):
    input_h = os.path.join(INCLUDE, "mlp784_input.h")
    weights_h = os.path.join(INCLUDE, "mlp784_weights.h")

    batch_size = parse_define(input_h, "MLP784_BATCH_SIZE", 1)
    w1 = parse_int_array_from_header(weights_h, "MLP784_FC1_WEIGHTS")
    w2 = parse_int_array_from_header(weights_h, "MLP784_FC2_WEIGHTS")
    w3 = parse_int_array_from_header(weights_h, "MLP784_FC3_WEIGHTS")

    if batch_size > 1:
        all_samples = parse_int_array_from_header(input_h, "MLP784_BATCH_SAMPLES")
        labels = parse_int_array_from_header(input_h, "MLP784_BATCH_LABELS")
    else:
        all_samples = parse_int_array_from_header(input_h, "MLP784_INPUT")
        labels = [parse_define(input_h, "MLP784_SAMPLE_LABEL", 0)]

    samples = range(batch_size) if sample is None else [sample]
    ok_all = True

    for s in samples:
        if s < 0 or s >= batch_size:
            print("skip invalid sample %d" % s)
            continue
        base = s * 784
        inp = all_samples[base : base + 784]
        print("")
        print("sample=%d in_sum=%d label=%d" % (s, checksum(inp), labels[s]))

        h1_sw = fc_blob(inp, w1, 784, 30, True)
        h1_hw = fc_hw(inp, w1, 784, 30, True)
        ok_all &= check_vec("FC1", h1_sw, h1_hw)

        h2_sw = fc_blob(h1_sw, w2, 30, 30, True)
        h2_hw = fc_hw(h1_hw, w2, 30, 30, True)
        ok_all &= check_vec("FC2", h2_sw, h2_hw)

        logits_sw = fc_blob(h2_sw, w3, 30, 10, False)
        logits_hw = fc_hw(h2_hw, w3, 30, 10, False)
        ok_all &= check_vec("FC3", logits_sw, logits_hw)

        pred = argmax(logits_sw)
        print("  pipeline pred=%d label=%d %s" % (
            pred, labels[s], "OK" if pred == labels[s] else "MISS",
        ))

    return ok_all


def main():
    parser = argparse.ArgumentParser(
        description="Offline tensor accel bit-exact validation (FC/CONV/POOL/RELU)"
    )
    parser.add_argument("--sample", type=int, default=None, help="MLP784 sample index")
    parser.add_argument("--skip-mlp", action="store_true", help="Only run tiny fixtures")
    args = parser.parse_args()

    ok = test_tiny_ops()
    if not args.skip_mlp:
        ok &= test_mlp784(args.sample)

    print("")
    print("overall: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
