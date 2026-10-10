#include "mlp784.h"
#if MLP784_ACCEL_UNIT
#include "mlp_accel_unit_test.h"
#endif

Message msg;

static int fc2_weights[MLP_FC2_WEIGHTS];
static int weights_ready = 0;

int main() {
    static int hidden1[MLP_HIDDEN1];
    static int hidden2[MLP_HIDDEN2];
    int i;
    /* Always match master RR schedule: N_INF * (W_HB + W_N). */
    int nloops = MLP784_INFERENCE_LOOPS;

    if (!weights_ready) {
        mlp_init_fc2_weights(fc2_weights);
        weights_ready = 1;
    }

#if MLP784_ACCEL_UNIT
    Echo("ACCEL unit fc2 start\n");
#else
    Echo("MLP fc2 start\n");
#endif
    Echo("MLP fc2 total inferences:");
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
        mlp_recv_hidden1_recover(hidden1);

#if MLP784_ACCEL_UNIT
        mlp_accel_unit_fc2(hidden1, hidden2, fc2_weights);
#else
        if (niles_is_semantic_probe(i) && mlp_ic_check_fc1(hidden1) != IC_CHECK_PASS) {
            mlp_ic_alert(1, mlp_fc1);
            mlp_ic_apply_psb(1, hidden1, GOLDEN_FC1, MLP_HIDDEN1);
        }
        mlp_run_fc2(hidden1, hidden2, fc2_weights);
#endif
        cnn_send_tensor(hidden2, MLP_HIDDEN2, mlp_fc3);
    }

#if MLP784_ACCEL_UNIT
    Echo("ACCEL unit fc2 done\n");
#else
    Echo("MLP fc2 done\n");
#endif
    exit();
}
