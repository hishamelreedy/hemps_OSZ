#include "alexnet.h"
Message msg;
int main() {
    static int input[ALEX_INPUT_DIM];
    static int output[ALEX_OUTPUT_DIM];
    int predicted;
    alex_load_input(input, 0);
    Echo("ALEX master start\n");
    cnn_send_tensor(input, ALEX_INPUT_DIM, alex_c1p);
    if (cnn_recv_tensor(alex_fc2, output, ALEX_OUTPUT_DIM) != ALEX_OUTPUT_DIM) {
        Echo("ALEX recv error\n");
        exit();
    }
    if (alex_ic_fc2(output) != IC_CHECK_PASS)
        Echo("[IC-ALERT] master fc2 check failed\n");
    predicted = cnn_mlp_argmax(output, ALEX_OUTPUT_DIM);
    Echo("ALEX expected class:");
    Echo(itoa(ALEXNET_BATCH_LABELS[0]));
    Echo("ALEX predicted class:");
    Echo(itoa(predicted));
    Echo("ALEX master done\n");
    exit();
}
