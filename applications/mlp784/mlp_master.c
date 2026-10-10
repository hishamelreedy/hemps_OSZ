#include "mlp784.h"

Message msg;

/*
 * Run one end-to-end inference.
 * msgcode == NILES_MSG_HB_INFER: golden semantic HB (Stage II stats check).
 * msgcode == NILES_MSG_NORMAL: workload sample (Stage I on slave recv only).
 */
static int mlp_run_one_sample(int sample, int seq_idx, int msgcode, int *correct_out) {
    static int input[MLP_INPUT_DIM];
    static int output[MLP_OUTPUT_DIM];
    int recv_len;
    int s;
    int predicted;
    int is_hb = (msgcode == NILES_MSG_HB_INFER);
    int ic_ok = 1;

    for (s = 0; s < MLP_OUTPUT_DIM; s++)
        output[s] = 0;

    mlp_load_input(input, sample);

    if (is_hb) {
        Echo("[NILES-HB-SEM] probe=");
        Echo(itoa(seq_idx));
        Echo(" sample=");
        Echo(itoa(sample));
        Echo(" msgcode=");
        Echo(itoa(msgcode));
        Echo(" start tick:");
        Echo(itoa(GetTick()));
        Echo("\n");
    } else {
        Echo("MLP sample start tick:");
        Echo(itoa(GetTick()));
        Echo("\n");
    }

    cnn_send_tensor(input, MLP_INPUT_DIM, mlp_fc1);
    Echo("MLP send done tick:");
    Echo(itoa(GetTick()));
    Echo("\n");

    recv_len = mlp_recv_output_recover(output);
    Echo("MLP recv done len=");
    Echo(itoa(recv_len));
    Echo(" tick:");
    Echo(itoa(GetTick()));
    Echo("\n");

    /* Stage II envelope: security HB path only. */
    if (is_hb && mlp_ic_check_fc3(output) != IC_CHECK_PASS) {
        ic_ok = 0;
        mlp_ic_alert(3, mlp_fc3);
        mlp_ic_apply_psb(3, output, GOLDEN_FC3, MLP_OUTPUT_DIM);
    }

    predicted = cnn_mlp_argmax(output, MLP_OUTPUT_DIM);

    if (is_hb) {
        Echo("[NILES-HB-SEM] probe=");
        Echo(itoa(seq_idx));
        Echo(" expected=");
        Echo(itoa(MLP784_BATCH_LABELS[sample]));
        Echo(" predicted=");
        Echo(itoa(predicted));
        Echo(" ic_ok=");
        Echo(itoa(ic_ok));
        Echo(" end tick:");
        Echo(itoa(GetTick()));
        Echo("\n");
    } else {
        Echo("MLP sample:");
        Echo(itoa(sample));
        Echo("MLP mnist idx:");
        Echo(itoa(MLP784_BATCH_INDICES[sample]));
        Echo("MLP expected class:");
        Echo(itoa(MLP784_BATCH_LABELS[sample]));
        Echo("MLP predicted class:");
        Echo(itoa(predicted));
        Echo("MLP sample end tick:");
        Echo(itoa(GetTick()));
        Echo("\n");
    }

    if (predicted == MLP784_BATCH_LABELS[sample])
        (*correct_out)++;

    return ic_ok;
}

int main() {
    int b;
    int h;
    int w;
    int seq = 0;
    int sample;
    int correct = 0;
    int semantic_correct = 0;
    int semantic_probes = 0;
    int golden = NILES_GOLDEN_SAMPLE;

    if (MLP784_BATCH_SIZE > 0)
        golden = NILES_GOLDEN_SAMPLE % MLP784_BATCH_SIZE;

    Echo("Hesham is Testing \n");
    Echo("MLP master start\n");
    Echo("MLP batch size:");
    Echo(itoa(MLP784_BATCH_SIZE));
    Echo("\n");
    Echo("NILES W_HB:");
    Echo(itoa(NILES_HB_W_HB));
    Echo(" W_N:");
    Echo(itoa(NILES_HB_W_N));
    Echo(" N_INF:");
    Echo(itoa(NILES_N_INF));
    Echo("\n");
    Echo("MLP inference loops:");
    Echo(itoa(MLP784_INFERENCE_LOOPS));
    Echo("\n");
    Echo("NILES HB mode (legacy):");
    Echo(itoa(NILES_HB_MODE));
    Echo("\n");

    if (niles_hb_enabled())
        Echo("NILES weighted RR: HB then normal per inference batch\n");

    /*
     * Three-loop weighted round-robin:
     *   for each of N_INF work batches:
     *     K=W_HB heartbeat (golden) inferences  -> msgcode HB
     *     W_N normal inferences                 -> msgcode NORMAL
     * rho_hb = W_HB/(W_HB+W_N); slaves use MLP784_INFERENCE_LOOPS.
     */
    for (b = 0; b < NILES_N_INF; b++) {
        for (h = 0; h < NILES_HB_W_HB; h++) {
            semantic_probes++;
            mlp_run_one_sample(golden, seq, NILES_MSG_HB_INFER, &semantic_correct);
            seq++;
#if NILES_HB_INTERVAL_US > 0
            /* Optional spacing between HB flits (sim tick proxy unused if 0). */
#endif
        }

        for (w = 0; w < NILES_HB_W_N; w++) {
            if (MLP784_BATCH_SIZE <= 0)
                sample = 0;
            else
                sample = (b + w) % MLP784_BATCH_SIZE;
            mlp_run_one_sample(sample, seq, NILES_MSG_NORMAL, &correct);
            seq++;
        }
    }

#if NILES_ENABLE_HEARTBEAT && MLP784_ENABLE_IC
    if (semantic_probes > 0) {
        Echo("NILES semantic HB correct:");
        Echo(itoa(semantic_correct));
        Echo(" NILES semantic HB probes:");
        Echo(itoa(semantic_probes));
        Echo("\n");
    }
#endif

    if (NILES_HB_W_N > 0) {
        Echo("MLP batch correct:");
        Echo(itoa(correct));
        Echo("\n");
        Echo("MLP batch done\n");
    } else {
        Echo("MLP hb-only done\n");
    }

    exit();
}
