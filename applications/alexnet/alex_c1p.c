#include "alexnet.h"
Message msg;
static int c1_weights[ALEX_C1_WEIGHTS];
static int ready;
int main() {
    static int input[ALEX_INPUT_DIM];
    static int p1[ALEX_P1_DIM];
    static int tmp[ALEX_C1_H * ALEX_C1_W * ALEX_C1_OUT];
    if (!ready) { alex_init_c1(c1_weights); ready = 1; }
    Echo("ALEX c1p start\n");
    cnn_recv_tensor(alex_master, input, ALEX_INPUT_DIM);
    alex_run_c1p(input, p1, tmp, c1_weights);
    cnn_send_tensor(p1, ALEX_P1_DIM, alex_c2p);
    Echo("ALEX c1p done\n");
    exit();
}
