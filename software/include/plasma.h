/*!\file plasma.h 
 * HEMPS VERSION - 8.0 - support for RT applications 
 * 
 * Distribution:  June 2016 
 * 
 * Edited by: Marcelo Ruaro - contact: marcelo.ruaro@acad.pucrs.br 
 * 
 * Research group: GAPH-PUCRS   -  contact:  fernando.moraes@pucrs.br 
 * 
 * \brief  Plasma Hardware Definitions 
 */ 
 
#ifndef __PLASMA_H__ 
#define __PLASMA_H__ 
 
/*********** Hardware addresses ***********/ 
#define UART_WRITE        		0x20000000 
#define UART_READ         		0x20000000 
#define IRQ_MASK          		0x20000010 
#define IRQ_STATUS        		0x20000020 
#define TIME_SLICE       		0x20000060
#define TIMEOUT_REG		        0x20000040
#define SYS_CALL		   		0x20000070 
#define END_SIM 		   		0x20000080 
#define CLOCK_HOLD 		   		0x20000090 
#define CLOCK_HOLD_WAIT_KERNEL	0x20000410 

/* Network Interface*/ 
#define	NI_STATUS_RECV		0x20000100 
#define	NI_STATUS_SEND		0x20000110 
#define	NI_RECV				0x20000120 
#define	NI_SEND				0x20000130 
#define	NI_CONFIG			0x20000140 
#define	NI_ACK				0x20000150 
#define	NI_NACK				0x20000160 
#define	NI_END				0x20000170 
#define	CURRENT_PAGE		0x20000180 
#define NEXT_PAGE			0x20000190 
 
/* DMNI*/ 
#define DMNI_SIZE_2				0x20000205 
#define DMNI_ADDRESS_2 			0x20000215 
#define DMNI_SIZE		  		0x20000200 
#define DMNI_ADDRESS		  	0x20000210 
#define DMNI_OP			  		0x20000220 
#define DMNI_START		  		0x20000230 
#define DMNI_ACK			  	0x20000240 
#define DMNI_SEND_ACTIVE	  	0x20000250 
#define DMNI_RECEIVE_ACTIVE		0x20000260 
#define DMA_SEND_KERNEL			0x20000264 
#define DMA_WAIT_KERNEL			0x20000268 
#define DMA_FAIL_KERNEL			0x20000272 
 
 
//Scheduling report 
#define SCHEDULING_REPORT	0x20000270 
#define INTERRUPTION		0x10000 
//#define SYSCALL			0x20000 
#define SCHEDULER			0x40000 
#define IDLE				0x80000 
 
//Communication graphical debbug 
#define ADD_PIPE_DEBUG			0x20000280 
#define REM_PIPE_DEBUG			0x20000285 
#define ADD_REQUEST_DEBUG		0x20000290 
#define REM_REQUEST_DEBUG		0x20000295 
 
/* DMNI operations */ 
#define READ	0 
#define WRITE	1 
 
#define TICK_COUNTER	  	0x20000300 
#define CURRENT_TASK	  	0x20000400 
 
#define REQ_APP		  		0x20000350 
#define ACK_APP		  		0x20000360 
 
#define SLACK_TIME_MONITOR		0x20000364 
 
//seek registers mapping 
#define SEEK_SERVICE_REGISTER	0x20000370 
#define SEEK_TARGET_REGISTER	0x20000380 
#define SEEK_SOURCE_REGISTER	0x20000390 
#define SEEK_PAYLOAD_REGISTER	0x20000394 
#define SEEK_BACKTRACK			0x20000500 
 
#define DMNI_TIMEOUT_SIGNAL		0x20000530 
#define WRAPPER_REGISTER        0x20000540 
#define WRAPPER_MASK_GO_REGISTER  0x20000544 
#define WRAPPER_MASK_BACK_REGISTER 0x20000548 
#define SEEK_OPMODE_REGISTER	  0x20000550 
 
#define SEEK_BACKTRACK_REG_SEL	  0x20000554 
 
#define KERNEL_DEBUG_STATE	    0x20000558 
 
// Access Point registers 
#define K1_REG                0x20000560 
#define K2_REG                0x20000564 
#define AP_MASK               0x20000568 
#define APP_ID_REG            0x20000570 
#define AP_THRESHOLD          0x20000574 
#define AP_STATUS             0x20000578
 
 
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
#define ACCEL_CTRL_FC_START     0x00000001
#define ACCEL_CTRL_START        ACCEL_CTRL_FC_START
#define ACCEL_CTRL_CLEAR_DONE   0x00000002
#define ACCEL_DIMS_RELU_SHIFT   31
#define ACCEL_DIMS_RELU_MASK    0x80000000u
#define ACCEL_STATUS_BUSY       0x00000001
#define ACCEL_STATUS_DONE       0x00000002

/* Instruction opcodes (ACCEL_OPCODE) */
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
 
#define SLACK_TIME_WINDOW		50000 // half milisecond 
 
/*********** Interrupt bits **************/ 
#define IRQ_PENDING_SERVICE			0x01 //bit 0 
#define IRQ_SLACK_TIME				0x02 //bit 1 
#define IRQ_SCHEDULER				0x08 //bit 3
#define IRQ_TIMEOUT				    0x10 //bit 4 
#define IRQ_NOC					 	0x20 //bit 5 
#define IRQ_AP				 	    0x40 //bit 6 
#define IRQ_SEEK				 	0x80 //bit 7 
 
 
          
/*Memory Access*/ 
#define MemoryRead(A) (*(volatile unsigned int*)(A)) 
#define MemoryWrite(A,V) *(volatile unsigned int*)(A)=(V) 
 
#endif /*__PLASMA_H__*/ 
 
