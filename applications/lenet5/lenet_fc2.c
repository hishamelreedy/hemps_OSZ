#include "lenet5.h"



Message msg;



int main() {


    static int h1[LENET_FC1_OUT];

    static int h2[LENET_FC2_OUT];

    int sample;



    Echo("LENET fc2 start\n");



    for (sample = 0; sample < LENET5_BATCH_SIZE; sample++) {

        cnn_recv_tensor(lenet_fc1, h1, LENET_FC1_OUT);



        if (LENET5_BATCH_SIZE == 1 && lenet_ic_check_fc1(h1) != IC_CHECK_PASS)

            lenet_ic_alert(3, lenet_fc1);



        lenet_run_fc2(h1, h2, LENET5_FC2_WEIGHTS);

        cnn_send_tensor(h2, LENET_FC2_OUT, lenet_fc3);

    }



    Echo("LENET fc2 done\n");

    exit();

}

