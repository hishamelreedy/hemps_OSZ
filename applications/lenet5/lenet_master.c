#include "lenet5.h"

Message msg;

static void lenet_master_recv_and_score(int sample, int *output, int *correct) {
    int recv_len;
    int predicted;

    recv_len = cnn_recv_tensor(lenet_fc3, output, LENET_OUTPUT_DIM);
    if (recv_len != LENET_OUTPUT_DIM) {
        Echo("LENET recv error\n");
        exit();
    }

    if (LENET5_BATCH_SIZE == 1 && lenet_ic_check_fc3(output) != IC_CHECK_PASS)
        Echo("[IC-ALERT] master fc3 check failed\n");

    predicted = cnn_mlp_argmax(output, LENET_OUTPUT_DIM);

    Echo("LENET sample:");
    Echo(itoa(sample));
    Echo("LENET mnist idx:");
    Echo(itoa(LENET5_BATCH_INDICES[sample]));
    Echo("LENET expected class:");
    Echo(itoa(LENET5_BATCH_LABELS[sample]));
    Echo("LENET predicted class:");
    Echo(itoa(predicted));
    Echo("LENET sample end tick:");
    Echo(itoa(GetTick()));

    if (predicted == LENET5_BATCH_LABELS[sample])
        (*correct)++;
}

int main() {
#if LENET5_ACCEL_UNIT
    Echo("LENET master accel-unit stub\n");
    exit();
#endif
    static int input[LENET_INPUT_DIM];
    static int output[LENET_OUTPUT_DIM];
    int sample;
    int correct = 0;

    Echo("LENET master start\n");
    Echo("LENET batch size:");
    Echo(itoa(LENET5_BATCH_SIZE));

#if LENET5_SPATIAL_PIPELINES > 0
    {
        int pipe = lenet_pipeline_row();

        Echo("LENET spatial pipelines:");
        Echo(itoa(LENET5_SPATIAL_PIPELINES));
        Echo("LENET spatial pipe row:");
        Echo(itoa(pipe));

        if (pipe < 0 || pipe >= LENET5_SPATIAL_PIPELINES) {
            Echo("LENET spatial pipe id error\n");
            exit();
        }

        lenet_load_input(input, pipe);
        cnn_send_tensor(input, LENET_INPUT_DIM, lenet_c1p1);
        lenet_master_recv_and_score(pipe, output, &correct);

        Echo("LENET spatial pipe correct:");
        Echo(itoa(correct));
        Echo("LENET spatial done\n");
        exit();
    }
#elif LENET5_BATCH_SIZE == 1
    lenet_load_input(input, 0);
    cnn_send_tensor(input, LENET_INPUT_DIM, lenet_c1p1);
    lenet_master_recv_and_score(0, output, &correct);
    Echo("LENET master done\n");
    exit();
#else
    /*
     * Pipelined batch: inject all inputs back-to-back, then drain all outputs.
     * Safe for small BATCH_SIZE (e.g. 2); larger batches may need a recv window
     * inside the send loop to avoid fc3->master pipe deadlock.
     */
    Echo("LENET pipeline send loop\n");

    for (sample = 0; sample < LENET5_BATCH_SIZE; sample++) {
        lenet_load_input(input, sample);
        Echo("LENET send sample:");
        Echo(itoa(sample));
        Echo(" tick:");
        Echo(itoa(GetTick()));
        cnn_send_tensor(input, LENET_INPUT_DIM, lenet_c1p1);
    }

    Echo("LENET pipeline recv loop\n");

    for (sample = 0; sample < LENET5_BATCH_SIZE; sample++) {
        Echo("LENET recv sample:");
        Echo(itoa(sample));
        Echo(" tick:");
        Echo(itoa(GetTick()));
        lenet_master_recv_and_score(sample, output, &correct);
    }

    Echo("LENET batch correct:");
    Echo(itoa(correct));
    Echo("LENET batch done\n");
    exit();
#endif
}
