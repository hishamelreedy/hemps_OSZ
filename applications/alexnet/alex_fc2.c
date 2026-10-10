#include "alexnet.h"
Message msg;
static int fc2_weights[ALEX_FC2_WEIGHTS];
static int ready;
int main() {
    static int h1[ALEX_FC1_OUT];
    static int out[ALEX_OUTPUT_DIM];
    if (!ready) { alex_init_fc2(fc2_weights); ready = 1; }
    Echo("ALEX fc2 start\n");
    cnn_recv_tensor(alex_fc1, h1, ALEX_FC1_OUT);
    if (alex_ic_fc1(h1) != IC_CHECK_PASS) alex_ic_alert(4, alex_fc1);
    alex_run_fc2(h1, out, fc2_weights);
    cnn_send_tensor(out, ALEX_OUTPUT_DIM, alex_master);
    Echo("ALEX fc2 done\n");
    exit();
}
