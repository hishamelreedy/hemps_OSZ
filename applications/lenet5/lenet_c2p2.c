#include "lenet5.h"



Message msg;



int main() {

    static int p1[LENET_P1_DIM];

    static int p2[LENET_P2_DIM];

    static int c2_tmp[LENET_C2_H * LENET_C2_W * LENET_C2_OUT_C];

    int sample;
    int recv_len;



    Echo("LENET c2p2 start\n");



    for (sample = 0; sample < LENET5_BATCH_SIZE; sample++) {

        cnn_recv_tensor(lenet_c1p1, p1, LENET_P1_DIM);



        if (LENET5_BATCH_SIZE == 1 && lenet_ic_check_c1p1(p1) != IC_CHECK_PASS)

            lenet_ic_alert(1, lenet_c1p1);



        lenet_run_c2p2(p1, p2, c2_tmp, LENET5_C2_WEIGHTS);

        cnn_send_tensor(p2, LENET_P2_DIM, lenet_fc1);

    }



    Echo("LENET c2p2 done\n");

    exit();

}

