#include "mlp784.h"

#if MLP784_ACCEL_UNIT
#include "mlp_accel_unit_test.h"
#endif

Message msg;

static int fc1_weights[MLP_FC1_WEIGHTS];
static int weights_ready = 0;

int main() {
    static int input[MLP_INPUT_DIM];
    static int hidden1[MLP_HIDDEN1];
    int i;
    int sample;
    /* Always match master RR schedule: N_INF * (W_HB + W_N). */
    int nloops = MLP784_INFERENCE_LOOPS;

    if (!weights_ready) {
        mlp_init_fc1_weights(fc1_weights);
        weights_ready = 1;
    }

#if MLP784_ACCEL_UNIT
    Echo("ACCEL unit fc1 start\n");
#else
    Echo("MLP fc1 start\n");
#endif
    Echo("MLP fc1 total inferences:");
    Echo(itoa(nloops));
    Echo(" (N_INF=");
    Echo(itoa(NILES_N_INF));
    Echo(" W_HB=");
    Echo(itoa(NILES_HB_W_HB));
    Echo(" W_N=");
    Echo(itoa(NILES_HB_W_N));
    Echo(")\n");

    for (i = 0; i < nloops; i++) {
#if MLP784_ACCEL_UNIT
        Echo("ACCEL unit sample:");
        Echo(itoa(i));
#endif
        sample = niles_loop_sample_index(i);
        mlp_recv_input_recover(input, sample);

#if MLP784_ACCEL_UNIT
        if (i == 0) {
            Echo("ACCEL unit in sum:");
            Echo(itoa(mlp_input_checksum(input)));
        }
        mlp_accel_unit_fc1(input, hidden1, fc1_weights);
#else
        mlp_run_fc1(input, hidden1, fc1_weights);
#endif
        cnn_send_tensor(hidden1, MLP_HIDDEN1, mlp_fc2);
    }

#if MLP784_ACCEL_UNIT
    Echo("ACCEL unit fc1 done\n");
#else
    Echo("MLP fc1 done\n");
#endif
    exit();
}
