#include "vgg16.h"
Message msg;
static int b1_weights[VGG_B1_WEIGHTS];
static int ready;
int main() {
    static int input[VGG_INPUT_DIM];
    static int b1[VGG_B1_DIM];
    static int tmp[28 * 28 * 8];
    if (!ready) { vgg_init_b1(b1_weights); ready = 1; }
    Echo("VGG b1 start\n");
    cnn_recv_tensor(vgg_master, input, VGG_INPUT_DIM);
    vgg_run_b1(input, b1, tmp, b1_weights);
    cnn_send_tensor(b1, VGG_B1_DIM, vgg_b2);
    Echo("VGG b1 done\n");
    exit();
}
