#include "mlp784.h"
#if MLP784_ACCEL_UNIT
#include "mlp_accel_unit_test.h"
#endif

Message msg;

static int fc3_weights[MLP_FC3_WEIGHTS];
static int weights_ready = 0;

int main() {
    static int hidden2[MLP_HIDDEN2];
    static int output[MLP_OUTPUT_DIM];
    int i;
    /* Always match master RR schedule: N_INF * (W_HB + W_N). */
    int nloops = MLP784_INFERENCE_LOOPS;

    if (!weights_ready) {
        mlp_init_fc3_weights(fc3_weights);
        weights_ready = 1;
    }

#if MLP784_ACCEL_UNIT
    Echo("ACCEL unit fc3 start\n");
#else
    Echo("MLP fc3 start\n");
#endif
    Echo("MLP fc3 total inferences:");
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
        mlp_recv_hidden2_recover(hidden2);

#if MLP784_ACCEL_UNIT
        mlp_accel_unit_fc3(hidden2, output, fc3_weights);
#else
        if (niles_is_semantic_probe(i) && mlp_ic_check_fc2(hidden2) != IC_CHECK_PASS) {
            mlp_ic_alert(2, mlp_fc2);
            mlp_ic_apply_psb(2, hidden2, GOLDEN_FC2, MLP_HIDDEN2);
        }
        mlp_run_fc3(hidden2, output, fc3_weights);
#endif
        cnn_send_tensor(output, MLP_OUTPUT_DIM, mlp_master);
    }

#if MLP784_ACCEL_UNIT
    Echo("ACCEL unit fc3 done\n");
#else
    Echo("MLP fc3 done\n");
#endif
    exit();
}
