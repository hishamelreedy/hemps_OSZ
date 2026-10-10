#include "alexnet.h"
Message msg;
static int c3_weights[ALEX_C3_WEIGHTS];
static int ready;
int main() {
    static int p2[ALEX_P2_DIM];
    static int c3[ALEX_C3_DIM];
    if (!ready) { alex_init_c3(c3_weights); ready = 1; }
    Echo("ALEX c3 start\n");
    cnn_recv_tensor(alex_c2p, p2, ALEX_P2_DIM);
    if (alex_ic_c2p(p2) != IC_CHECK_PASS) alex_ic_alert(2, alex_c2p);
    alex_run_c3(p2, c3, c3_weights);
    cnn_send_tensor(c3, ALEX_C3_DIM, alex_fc1);
    Echo("ALEX c3 done\n");
    exit();
}
