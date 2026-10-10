#include "lenet5.h"



Message msg;



int main() {

    static int h2[LENET_FC2_OUT];

    static int output[LENET_OUTPUT_DIM];

    int sample;



    Echo("LENET fc3 start\n");



    for (sample = 0; sample < LENET5_BATCH_SIZE; sample++) {

        cnn_recv_tensor(lenet_fc2, h2, LENET_FC2_OUT);



        if (LENET5_BATCH_SIZE == 1 && lenet_ic_check_fc2(h2) != IC_CHECK_PASS)

            lenet_ic_alert(4, lenet_fc2);



        lenet_run_fc3(h2, output, LENET5_FC3_WEIGHTS);

        cnn_send_tensor(output, LENET_OUTPUT_DIM, lenet_master);

    }



    Echo("LENET fc3 done\n");

    exit();

}

