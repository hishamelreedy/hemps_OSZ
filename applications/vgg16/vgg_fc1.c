#include "vgg16.h"
Message msg;
static int fc1_weights[VGG_FC1_WEIGHTS];
static int ready;
int main() {
    static int b3[VGG_B3_DIM];
    static int h1[VGG_FC1_OUT];
    if (!ready) { vgg_init_fc1(fc1_weights); ready = 1; }
    Echo("VGG fc1 start\n");
    cnn_recv_tensor(vgg_b3, b3, VGG_B3_DIM);
    if (vgg_ic_b3(b3) != IC_CHECK_PASS) vgg_ic_alert(3, vgg_b3);
    cnn_fc_blob(b3, h1, fc1_weights, VGG_B3_DIM, VGG_FC1_OUT, 1);
    cnn_send_tensor(h1, VGG_FC1_OUT, vgg_fc2);
    Echo("VGG fc1 done\n");
    exit();
}
