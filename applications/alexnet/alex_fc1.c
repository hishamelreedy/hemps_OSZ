#include "alexnet.h"
Message msg;
static int fc1_weights[ALEX_FC1_WEIGHTS];
static int ready;
int main() {
    static int c3[ALEX_C3_DIM];
    static int h1[ALEX_FC1_OUT];
    if (!ready) { alex_init_fc1(fc1_weights); ready = 1; }
    Echo("ALEX fc1 start\n");
    cnn_recv_tensor(alex_c3, c3, ALEX_C3_DIM);
    if (alex_ic_c3(c3) != IC_CHECK_PASS) alex_ic_alert(3, alex_c3);
    alex_run_fc1(c3, h1, fc1_weights);
    cnn_send_tensor(h1, ALEX_FC1_OUT, alex_fc2);
    Echo("ALEX fc1 done\n");
    exit();
}
