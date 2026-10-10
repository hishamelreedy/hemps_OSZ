#!/usr/bin/env python3
"""Offline SW-only LeNet pred check on TB batch + golden (no numpy)."""

from __future__ import print_function

import re
import sys
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCLUDE = os.path.join(REPO, "software", "include")


def parse_arr(text, name):
    m = re.search(r"static const int %s\[(\d+)\] = \{([^}]+)\}" % name, text, re.S)
    if not m:
        raise RuntimeError("missing %s" % name)
    return [int(x) for x in re.findall(r"-?\d+", m.group(2))]


def parse_define(text, name, default=None):
    m = re.search(r"#define %s\s+(-?\d+)" % name, text)
    return int(m.group(1)) if m else default


def relu(x):
    return x if x > 0 else 0


def conv2d(inp, wblob, in_h, in_w, in_c, out_c, k):
    out_h, out_w = in_h - k + 1, in_w - k + 1
    w_per = in_c * k * k
    bias = wblob[out_c * w_per :]
    out = [0] * (out_h * out_w * out_c)
    for oc in range(out_c):
        base = oc * w_per
        for oy in range(out_h):
            for ox in range(out_w):
                s = bias[oc]
                for ic in range(in_c):
                    for ky in range(k):
                        for kx in range(k):
                            xi = inp[((oy + ky) * in_w + (ox + kx)) * in_c + ic]
                            wi = wblob[base + ic * (k * k) + ky * k + kx]
                            s += (xi * wi) >> 8
                out[(oy * out_w + ox) * out_c + oc] = relu(s)
    return out


def pool(inp, in_h, in_w, ch):
    oh, ow = in_h // 2, in_w // 2
    out = [0] * (oh * ow * ch)
    for oy in range(oh):
        for ox in range(ow):
            for c in range(ch):
                v00 = inp[((oy * 2) * in_w + (ox * 2)) * ch + c]
                v01 = inp[((oy * 2) * in_w + (ox * 2 + 1)) * ch + c]
                v10 = inp[((oy * 2 + 1) * in_w + (ox * 2)) * ch + c]
                v11 = inp[((oy * 2 + 1) * in_w + (ox * 2 + 1)) * ch + c]
                out[(oy * ow + ox) * ch + c] = (v00 + v01 + v10 + v11) // 4
    return out


def fc(inp, wblob, in_dim, out_dim, do_relu):
    bias = wblob[in_dim * out_dim :]
    out = [0] * out_dim
    for o in range(out_dim):
        s = bias[o]
        for i in range(in_dim):
            s += (inp[i] * wblob[i * out_dim + o]) >> 8
        out[o] = relu(s) if do_relu else s
    return out


def infer(img, c1, c2, f1, f2, f3):
    t = conv2d(list(img), c1, 28, 28, 1, 6, 5)
    t = pool(t, 24, 24, 6)
    t = conv2d(t, c2, 12, 12, 6, 16, 5)
    p2 = pool(t, 8, 8, 16)
    t = fc(p2, f1, 256, 120, True)
    t = fc(t, f2, 120, 84, True)
    logits = fc(t, f3, 84, 10, False)
    pred = max(range(10), key=lambda i: logits[i])
    return pred, logits, max(p2), sum(p2)


def main():
    wtxt = open(os.path.join(INCLUDE, "lenet5_weights.h")).read()
    itxt = open(os.path.join(INCLUDE, "lenet5_input.h")).read()
    btxt = open(os.path.join(INCLUDE, "lenet5_tb_batch.h")).read()
    c1 = parse_arr(wtxt, "LENET5_C1_WEIGHTS")
    c2 = parse_arr(wtxt, "LENET5_C2_WEIGHTS")
    f1 = parse_arr(wtxt, "LENET5_FC1_WEIGHTS")
    f2 = parse_arr(wtxt, "LENET5_FC2_WEIGHTS")
    f3 = parse_arr(wtxt, "LENET5_FC3_WEIGHTS")
    gin = parse_arr(itxt, "LENET5_BATCH_SAMPLES")
    glabel = parse_define(itxt, "LENET5_SAMPLE_LABEL")
    batch = parse_arr(btxt, "LENET5_TB_BATCH_SAMPLES")
    labs = parse_arr(btxt, "LENET5_TB_BATCH_LABELS")
    n = parse_define(btxt, "LENET5_TB_BATCH_SIZE")
    if len(batch) != n * 784:
        print("FAIL batch size mismatch %d vs %d*784" % (len(batch), n))
        return 1

    gp, gl, gmax, gsum = infer(gin, c1, c2, f1, f2, f3)
    print("GOLDEN SW pred=%d label=%d p2_max=%d p2_sum=%d logits=%s" % (
        gp, glabel, gmax, gsum, gl))
    if gp != glabel or gmax <= 0:
        print("FAIL golden SW")
        return 1
    print("OK golden SW")

    correct = 0
    pred_hist = [0] * 10
    p2_dead = 0
    for s in range(n):
        img = batch[s * 784 : (s + 1) * 784]
        pred, logits, p2m, p2s = infer(img, c1, c2, f1, f2, f3)
        pred_hist[pred] += 1
        if p2m <= 0:
            p2_dead += 1
        if pred == labs[s]:
            correct += 1
        if s < 5 or s == n - 1:
            print("  sample %d label=%d SW_pred=%d p2_max=%d" % (
                s, labs[s], pred, p2m))

    acc = 100.0 * correct / float(n)
    print("BATCH SW: correct=%d/%d (%.1f%%) p2_dead=%d pred_hist=%s" % (
        correct, n, acc, p2_dead, pred_hist))
    if p2_dead == n:
        print("FAIL all P2 dead")
        return 1
    if pred_hist[0] == n:
        print("FAIL collapsed to class 0")
        return 1
    if acc < 40.0:
        print("FAIL SW accuracy too low")
        return 1
    print("OK SW batch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
