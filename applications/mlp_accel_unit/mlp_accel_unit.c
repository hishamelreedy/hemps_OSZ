/*
 * On-PE unit test: SW cnn_fc_blob vs HW accel_fc_hw on the same static buffers.
 * Maps to fc1 PE; logs max abs diff per layer for each batch sample.
 */
#include "mlp784_shared.h"

static int fc1_weights[MLP_FC1_WEIGHTS];
static int fc2_weights[MLP_FC2_WEIGHTS];
static int fc3_weights[MLP_FC3_WEIGHTS];

static int input[MLP_INPUT_DIM];
static int hidden1[MLP_HIDDEN1];
static int hidden2[MLP_HIDDEN2];
static int logits_sw[MLP_OUTPUT_DIM];
static int logits_hw[MLP_OUTPUT_DIM];
static int hw_tmp[MLP_HIDDEN1];

static int abs_diff(int a, int b) {
    int d = a - b;
    return d < 0 ? -d : d;
}

static int vec_max_diff(const int *a, const int *b, int n) {
    int i, m = 0, d;

    for (i = 0; i < n; i++) {
        d = abs_diff(a[i], b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

static void report_layer(const char *name, int max_diff, int n, const int *sw, const int *hw) {
    int sw_ix, hw_ix, i;

    Echo("ACCEL unit layer:");
    Echo(name);
    Echo("ACCEL unit max diff:");
    Echo(itoa(max_diff));
    sw_ix = 0;
    hw_ix = 0;
    for (i = 1; i < n; i++) {
        if (sw[i] > sw[sw_ix])
            sw_ix = i;
        if (hw[i] > hw[hw_ix])
            hw_ix = i;
    }
    Echo("ACCEL unit sw argmax:");
    Echo(itoa(sw_ix));
    Echo("ACCEL unit hw argmax:");
    Echo(itoa(hw_ix));
    Echo("ACCEL unit sw maxval:");
    Echo(itoa(sw[sw_ix]));
    Echo("ACCEL unit hw maxval:");
    Echo(itoa(hw[hw_ix]));
}

int main() {
    int sample;

    Echo("ACCEL unit test start\n");
    mlp_init_fc1_weights(fc1_weights);
    mlp_init_fc2_weights(fc2_weights);
    mlp_init_fc3_weights(fc3_weights);

    for (sample = 0; sample < MLP784_BATCH_SIZE; sample++) {
        int d1, d2, d3;

        Echo("ACCEL unit sample:");
        Echo(itoa(sample));
        mlp_load_input(input, sample);
        Echo("ACCEL unit in sum:");
        Echo(itoa(mlp_input_checksum(input)));

        cnn_fc_blob(input, hidden1, fc1_weights, MLP_INPUT_DIM, MLP_HIDDEN1, 1);
        accel_fc_hw(input, hw_tmp, fc1_weights, MLP_INPUT_DIM, MLP_HIDDEN1, 1);
        d1 = vec_max_diff(hidden1, hw_tmp, MLP_HIDDEN1);
        report_layer("fc1", d1, MLP_HIDDEN1, hidden1, hw_tmp);

        cnn_fc_blob(hidden1, hidden2, fc2_weights, MLP_HIDDEN1, MLP_HIDDEN2, 1);
        accel_fc_hw(hidden1, hw_tmp, fc2_weights, MLP_HIDDEN1, MLP_HIDDEN2, 1);
        d2 = vec_max_diff(hidden2, hw_tmp, MLP_HIDDEN2);
        report_layer("fc2", d2, MLP_HIDDEN2, hidden2, hw_tmp);

        cnn_fc_blob(hidden2, logits_sw, fc3_weights, MLP_HIDDEN2, MLP_OUTPUT_DIM, 0);
        accel_fc_hw(hidden2, logits_hw, fc3_weights, MLP_HIDDEN2, MLP_OUTPUT_DIM, 0);
        d3 = vec_max_diff(logits_sw, logits_hw, MLP_OUTPUT_DIM);
        report_layer("fc3", d3, MLP_OUTPUT_DIM, logits_sw, logits_hw);

        Echo("ACCEL unit pred sw:");
        Echo(itoa(cnn_mlp_argmax(logits_sw, MLP_OUTPUT_DIM)));
        Echo("ACCEL unit pred hw:");
        Echo(itoa(cnn_mlp_argmax(logits_hw, MLP_OUTPUT_DIM)));
        Echo("ACCEL unit expected:");
        Echo(itoa(MLP784_BATCH_LABELS[sample]));
    }

    Echo("ACCEL unit done\n");
    exit();
}
