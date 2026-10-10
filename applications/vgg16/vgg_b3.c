#include "vgg16.h"
Message msg;
static int b3_weights[VGG_B3_WEIGHTS];
static int ready;
int main() {
    static int b2[VGG_B2_DIM];
    static int b3[VGG_B3_DIM];
    if (!ready) { vgg_init_b3(b3_weights); ready = 1; }
    Echo("VGG b3 start\n");
    cnn_recv_tensor(vgg_b2, b2, VGG_B2_DIM);
    if (vgg_ic_b2(b2) != IC_CHECK_PASS) vgg_ic_alert(2, vgg_b2);
    vgg_run_b3(b2, b3, b3_weights);
    cnn_send_tensor(b3, VGG_B3_DIM, vgg_fc1);
    Echo("VGG b3 done\n");
    exit();
}
