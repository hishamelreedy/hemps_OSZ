#include "vgg16.h"
Message msg;
static int b2_weights[VGG_B2_WEIGHTS];
static int ready;
int main() {
    static int b1[VGG_B1_DIM];
    static int b2[VGG_B2_DIM];
    static int tmp[10 * 10 * 16];
    if (!ready) { vgg_init_b2(b2_weights); ready = 1; }
    Echo("VGG b2 start\n");
    cnn_recv_tensor(vgg_b1, b1, VGG_B1_DIM);
    if (vgg_ic_b1(b1) != IC_CHECK_PASS) vgg_ic_alert(1, vgg_b1);
    vgg_run_b2(b1, b2, tmp, b2_weights);
    cnn_send_tensor(b2, VGG_B2_DIM, vgg_b3);
    Echo("VGG b2 done\n");
    exit();
}
