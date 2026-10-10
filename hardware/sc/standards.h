/*
 * HeMPS_defaults.h
 *
 *  Created on: May 18, 2016
 *      Author: mruaro
 */

#ifndef STANDARDS_H_
#define STANDARDS_H_

#include <systemc.h>
#include <math.h>
#include "../../include/hemps_pkg.h"

#ifdef DUPLICATED_CHANNEL
//duplicated channel
  #define NPORT 10

  #define EAST0  0
  #define EAST1  1
  #define WEST0  2
  #define WEST1  3
  #define NORTH0 4
  #define NORTH1 5
  #define SOUTH0 6
  #define SOUTH1 7
  #define LOCAL0 8
  #define LOCAL1 9 
#else
//single channel
  #define NPORT 5
  #define EAST 	0
  #define WEST 	1
  #define NORTH 	2
  #define SOUTH 	3
  #define LOCAL   4
#endif

#define GROUND   5
#define EAST_IO  0
#define WEST_IO  1
#define NORTH_IO   2
#define SOUTH_IO   3

#define IO_SR_PATHS  5

#define TAM_FLIT 	16
#define METADEFLIT (TAM_FLIT/2)
#define QUARTOFLIT (TAM_FLIT/4)


	// Memory map constants.
#define DEBUG 					0x20000000
#define IRQ_MASK 				0x20000010
#define IRQ_STATUS_ADDR 		0x20000020
#define TIME_SLICE_ADDR 		0x20000060
#define TIMEOUT_REG_ADDR 		0x20000040
#define CLOCK_HOLD        0x20000090
#define CLOCK_HOLD_WAIT_KERNEL 				0x20000410

#define END_SIM 				0x20000080
#define NET_ADDRESS 			0x20000140

#define DMA_SIZE 				0x20000200
#define DMA_ADDR 				0x20000210
#define DMA_SIZE_2 				0x20000205
#define DMA_ADDR_2 				0x20000215
#define DMA_OP 					0x20000220
#define START_DMA 				0x20000230
#define DMA_ACK 				0x20000240
#define DMA_SEND_ACTIVE 		0x20000250
#define DMA_RECEIVE_ACTIVE		0x20000260
#define DMA_SEND_KERNEL   0x20000264
#define DMA_WAIT_KERNEL     0x20000268
#define DMA_FAIL_KERNEL      0x20000272


#define SCHEDULING_REPORT		0x20000270
#define ADD_PIPE_DEBUG			0x20000280
#define REM_PIPE_DEBUG			0x20000285
#define ADD_REQUEST_DEBUG		0x20000290
#define REM_REQUEST_DEBUG		0x20000295

#define TICK_COUNTER_ADDR 		0x20000300
#define REQ_APP_REG 			0x20000350
#define ACK_APP_REG 			0x20000360

#define SLACK_TIME_MONITOR		0x20000364

 //seek registers mapping
#define SEEK_SERVICE_REGISTER   0x20000370
#define SEEK_TARGET_REGISTER    0x20000380
#define SEEK_SOURCE_REGISTER    0x20000390
#define SEEK_PAYLOAD_REGISTER   0x20000394
#define SEEK_BACKTRACK          0x20000500
#define DMNI_TIMEOUT_SIGNAL     0x20000530
#define WRAPPER_REGISTER        0x20000540
#define WRAPPER_MASK_GO_REGISTER   0x20000544
#define WRAPPER_MASK_BACK_REGISTER   0x20000548
#define SEEK_OPMODE_REGISTER    0x20000550
#define SEEK_BACKTRACK_REG_SEL  0x20000554

// ARTUR LOG
// #define SEEK_CAM_READ           0x20000600
// enable seek log
#define SEEK_LOG

//Kernel pending service FIFO
#define PENDING_SERVICE_INTR	0x20000400

// Tensor accelerator MMIO (fixed-latency stub)
#define ACCEL_CTRL              0x20000420
#define ACCEL_CYCLES            0x20000424
#define ACCEL_STATUS            0x20000428
#define ACCEL_IN_ADDR           0x20000430
#define ACCEL_OUT_ADDR          0x20000434
#define ACCEL_WEIGHT_ADDR       0x20000438
#define ACCEL_DIMS              0x2000043C
#define ACCEL_DBG_OP            0x20000440
#define ACCEL_DBG_RUNS          0x20000444
#define ACCEL_DBG_IN0           0x20000448
#define ACCEL_DBG_W0            0x2000044C
#define ACCEL_DBG_OUT0          0x20000450
#define ACCEL_DBG_IN_SUM        0x20000454
#define ACCEL_DBG_OUT_MAX       0x20000458
#define ACCEL_DBG_STATUS        0x2000045C
#define ACCEL_OPCODE            0x20000460
#define ACCEL_DIMS1             0x20000464

/* Map CPU bus address (paged or nominal) to nominal ACCEL MMIO address. */
static inline sc_uint<32> accel_bus_to_nominal(sc_uint<32> bus_addr, sc_uint<8> page) {
	sc_uint<32> page_off = (sc_uint<32>)page * (sc_uint<32>)PAGE_SIZE_BYTES;
	sc_uint<32> stripped = bus_addr - page_off;

	if (stripped >= (sc_uint<32>)ACCEL_CTRL && stripped <= (sc_uint<32>)ACCEL_DIMS1)
		return stripped;
	if (bus_addr >= (sc_uint<32>)ACCEL_CTRL && bus_addr <= (sc_uint<32>)ACCEL_DIMS1)
		return bus_addr;
	return stripped;
}

#define ACCEL_CTRL_FC_START     0x00000001
#define ACCEL_CTRL_START        ACCEL_CTRL_FC_START
#define ACCEL_CTRL_CLEAR_DONE   0x00000002
#define ACCEL_DIMS_RELU_SHIFT   31
#define ACCEL_DIMS_RELU_MASK    0x80000000u
#define ACCEL_STATUS_BUSY       0x00000001
#define ACCEL_STATUS_DONE       0x00000002

#define ACCEL_OP_FC             0u
#define ACCEL_OP_CONV2D         1u
#define ACCEL_OP_AVGPOOL2X2     2u
#define ACCEL_OP_RELU           3u

#define ACCEL_DBG_OP_IDLE       0u
#define ACCEL_DBG_OP_WR_IN      1u
#define ACCEL_DBG_OP_WR_OUT     2u
#define ACCEL_DBG_OP_WR_WEIGHT  3u
#define ACCEL_DBG_OP_WR_DIMS    4u
#define ACCEL_DBG_OP_WR_CTRL    5u
#define ACCEL_DBG_OP_FC_START   6u
#define ACCEL_DBG_OP_FC_CLEAR   7u
#define ACCEL_DBG_OP_WR_OPCODE  8u
#define ACCEL_DBG_OP_WR_DIMS1   9u
#define ACCEL_DBG_OP_CONV_START 10u
#define ACCEL_DBG_OP_POOL_START 11u
#define ACCEL_DBG_OP_RELU_START 12u

#define ACCEL_DBG_ERR_NONE      0u
#define ACCEL_DBG_ERR_ZERO_DIM  1u
#define ACCEL_DBG_ERR_BAD_IN    2u
#define ACCEL_DBG_ERR_BAD_W     4u
#define ACCEL_DBG_ERR_BAD_OUT   8u
#define ACCEL_DBG_ERR_RAM_OOB   16u
#define ACCEL_DBG_ERR_BAD_OP    32u

#define SLACK_MONITOR_WINDOW 	50000
#define KERNEL_DEBUG_STATE    0x20000558

// Access Point registers
#define K1_REG                0x20000560
#define K2_REG                0x20000564
#define AP_MASK               0x20000568
#define APP_ID_REG            0x20000570
#define AP_THRESHOLD          0x20000574
#define AP_STATUS             0x20000578


//DMNI config code
#define CODE_CS_NET 		1
#define CODE_MEM_ADDR		2
#define CODE_MEM_SIZE  		3
#define CODE_MEM_ADDR2		4
#define CODE_MEM_SIZE2		5
#define CODE_OP				6

#define MEMORY_WORD_SIZE	4

#define BUFFER_TAM          8 // must be power of two
#define BUFFER_TAM_FIFO_PDN 4
#define BUFFER_TAM_SENDER   6

#define INPUT_DIRECTION      0
#define OUTPUT_DIRECTION     1

#define BUFFER_IN_WRAPPER       32
#define BUFFER_IN_PERIPHERAL    32
#define BUFFER_OUT_PERIPHERAL   64

typedef sc_uint<TAM_FLIT > regflit;
typedef sc_uint<16> regaddress;

typedef sc_uint<3> 				reg3;
typedef sc_uint<4> 				reg4;
typedef sc_uint<8> 				reg8;
typedef sc_uint<10> 			reg10;
typedef sc_uint<11> 			reg11;
typedef sc_uint<16> 			reg16;
typedef sc_uint<32> 			reg32;
typedef sc_uint<40> 			reg40;
typedef sc_uint<NPORT> 			regNport;
typedef sc_uint<TAM_FLIT> 		regflit;
typedef sc_uint<(TAM_FLIT/2)> 	regmetadeflit;
typedef sc_uint<(TAM_FLIT/4)> 	regquartoflit;
typedef sc_uint<(3*NPORT)> 		reg_mux;

//seek constants ksdjfhsdkfh k skdfjh
#define NPORT_SEEK 5
#define SOURCE_SIZE 32
#define TARGET_SIZE 16
#define SEEK_PAYLOAD_SIZE 8
#define TAM_SERVICE_SEEK 6
//these definitions are needed in the seek module
#define EAST 0
#define WEST 1
#define NORTH 2
#define SOUTH 3
#define LOCAL 4

#define SOURCE_ROUTING_TYPE 7
typedef sc_uint<TAM_SERVICE_SEEK>     reg_seek_service;
typedef sc_uint<SOURCE_SIZE>          reg_seek_source;
typedef sc_uint<TARGET_SIZE>          reg_seek_target;
typedef sc_uint<SEEK_PAYLOAD_SIZE>    reg_seek_payload;
typedef sc_uint<1>                    reg_seek_opmode;
typedef sc_uint<32>                   reg_seek_reg_backtrack;
#define GLOBAL_MODE   0
#define RESTRICT_MODE 1
//seek service types 
#define START_APP_SERVICE               1
#define TARGET_UNREACHABLE_SERVICE      2
#define CLEAR_SERVICE                   3
#define BACKTRACK_SERVICE               4
#define SEARCHPATH_SERVICE              5
#define END_TASK_SERVICE                6
#define SET_SECURE_ZONE_SERVICE         7
// #define PACKET_RESEND_SERVICE           8
#define SET_AP_SERVICE                  8
#define WARD_SERVICE                    9
#define OPEN_SECURE_ZONE_SERVICE        10
#define SECURE_ZONE_CLOSED_SERVICE      11
#define SECURE_ZONE_OPENED_SERVICE      12
#define FREEZE_TASK_SERVICE             13
#define UNFREEZE_TASK_SERVICE           14
#define MASTER_CANDIDATE_SERVICE        15
#define TASK_ALLOCATED_SERVICE          16
#define INITIALIZE_SLAVE_SERVICE        17
#define INITIALIZE_CLUSTER_SERVICE      18
#define LOAN_PROCESSOR_REQUEST_SERVICE  19
#define LOAN_PROCESSOR_RELEASE_SERVICE  20
#define END_TASK_OTHER_CLUSTER_SERVICE  21
#define WAIT_KERNEL_SERVICE             22
#define SEND_KERNEL_SERVICE             23
#define WAIT_KERNEL_SERVICE_ACK         24
#define FAIL_KERNEL_SERVICE             25
#define NEW_APP_SERVICE                 26
#define NEW_APP_ACK_SERVICE             27
#define GMV_READY_SERVICE               28
#define SET_SZ_RECEIVED_SERVICE         29
#define SET_EXCESS_SZ_SERVICE           30
#define RCV_FREEZE_TASK_SERVICE         31
#endif
