#include "vgg16.h"
Message msg;
static int fc2_weights[VGG_FC2_WEIGHTS];
static int ready;
int main() {
    static int h1[VGG_FC1_OUT];
    static int out[VGG_OUTPUT_DIM];
    if (!ready) { vgg_init_fc2(fc2_weights); ready = 1; }
    Echo("VGG fc2 start\n");
    cnn_recv_tensor(vgg_fc1, h1, VGG_FC1_OUT);
    if (vgg_ic_fc1(h1) != IC_CHECK_PASS) vgg_ic_alert(4, vgg_fc1);
    cnn_fc_blob(h1, out, fc2_weights, VGG_FC1_OUT, VGG_OUTPUT_DIM, 0);
    cnn_send_tensor(out, VGG_OUTPUT_DIM, vgg_master);
    Echo("VGG fc2 done\n");
    exit();
}
