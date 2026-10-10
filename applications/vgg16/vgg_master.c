#include "vgg16.h"
Message msg;
int main() {
    static int input[VGG_INPUT_DIM];
    static int output[VGG_OUTPUT_DIM];
    int predicted;
    vgg_load_input(input, 0);
    Echo("VGG master start\n");
    cnn_send_tensor(input, VGG_INPUT_DIM, vgg_b1);
    if (cnn_recv_tensor(vgg_fc2, output, VGG_OUTPUT_DIM) != VGG_OUTPUT_DIM) {
        Echo("VGG recv error\n");
        exit();
    }
    if (vgg_ic_fc2(output) != IC_CHECK_PASS)
        Echo("[IC-ALERT] master fc2 check failed\n");
    predicted = cnn_mlp_argmax(output, VGG_OUTPUT_DIM);
    Echo("VGG expected class:");
    Echo(itoa(VGG16_BATCH_LABELS[0]));
    Echo("VGG predicted class:");
    Echo(itoa(predicted));
    Echo("VGG master done\n");
    exit();
}
