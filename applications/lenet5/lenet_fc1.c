#include "lenet5.h"



Message msg;



int main() {

    static int p2[LENET_P2_DIM];

    static int h1[LENET_FC1_OUT];

    int sample;



    Echo("LENET fc1 start\n");



    for (sample = 0; sample < LENET5_BATCH_SIZE; sample++) {

        cnn_recv_tensor(lenet_c2p2, p2, LENET_P2_DIM);



        if (LENET5_BATCH_SIZE == 1 && lenet_ic_check_c2p2(p2) != IC_CHECK_PASS)

            lenet_ic_alert(2, lenet_c2p2);



        lenet_run_fc1(p2, h1, LENET5_FC1_WEIGHTS);

        cnn_send_tensor(h1, LENET_FC1_OUT, lenet_fc2);

    }



    Echo("LENET fc1 done\n");

    exit();

}

