#include "lenet5.h"

#if LENET5_ACCEL_UNIT

#include "lenet_accel_unit_test.h"

#endif



Message msg;



int main() {

#if LENET5_ACCEL_UNIT

    lenet_accel_unit_run();

#endif

    static int input[LENET_INPUT_DIM];

    static int p1[LENET_P1_DIM];

    static int c1_tmp[LENET_C1_H * LENET_C1_W * LENET_C1_OUT_C];

    int sample;



    Echo("LENET c1p1 start\n");



    for (sample = 0; sample < LENET5_BATCH_SIZE; sample++) {

        cnn_recv_tensor(lenet_master, input, LENET_INPUT_DIM);

        lenet_run_c1p1(input, p1, c1_tmp, LENET5_C1_WEIGHTS);

        cnn_send_tensor(p1, LENET_P1_DIM, lenet_c2p2);

    }



    Echo("LENET c1p1 done\n");

    exit();

}

