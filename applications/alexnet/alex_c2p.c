#include "alexnet.h"
Message msg;
static int c2_weights[ALEX_C2_WEIGHTS];
static int ready;
int main() {
    static int p1[ALEX_P1_DIM];
    static int p2[ALEX_P2_DIM];
    static int tmp[ALEX_C2_H * ALEX_C2_W * ALEX_C2_OUT];
    if (!ready) { alex_init_c2(c2_weights); ready = 1; }
    Echo("ALEX c2p start\n");
    cnn_recv_tensor(alex_c1p, p1, ALEX_P1_DIM);
    if (alex_ic_c1p(p1) != IC_CHECK_PASS) alex_ic_alert(1, alex_c1p);
    alex_run_c2p(p1, p2, tmp, c2_weights);
    cnn_send_tensor(p2, ALEX_P2_DIM, alex_c3);
    Echo("ALEX c2p done\n");
    exit();
}
