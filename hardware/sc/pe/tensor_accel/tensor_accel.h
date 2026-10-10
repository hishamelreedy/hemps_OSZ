//------------------------------------------------------------------------------------------------
//
//  Behavioral Q8 tensor accelerator (SystemC, hybrid PE).
//  Opcodes: FC, CONV2D, AVGPOOL2X2, RELU. Pointer-based PE-RAM access (no DMA).
//
//------------------------------------------------------------------------------------------------

#ifndef _tensor_accel_h
#define _tensor_accel_h

#include <systemc.h>
#include "../../standards.h"

#ifndef ACCEL_DEBUG
#define ACCEL_DEBUG 1
#endif

class ram;

SC_MODULE(tensor_accel) {

	sc_in<bool>              clock;
	sc_in<bool>              reset;

	sc_in<sc_uint<32> >      mem_address;
	sc_in<sc_uint<32> >      mem_read_address;
	sc_out<sc_uint<32> >     mem_rdata;
	sc_out<bool>             mem_hit;

	sc_in<sc_uint<32> >      mem_write_address;
	sc_in<sc_uint<32> >      mem_write_data;
	sc_in<bool>              mem_write;

	sc_out<bool>             pause_cpu;

	sc_in<sc_uint<32> >      tick_counter;
	sc_in<sc_uint<8> >       current_page;

	ram                     *mem;
	regaddress               router_address;
	unsigned char            shift_mem_page;

	void mmio_read();
	void update_pause();
	void reset_regs();
	void service_mmio_write(sc_uint<32> addr, sc_uint<32> wdata);

	SC_HAS_PROCESS(tensor_accel);

	tensor_accel(
		sc_module_name name_,
		ram *mem_,
		regaddress router_address_,
		unsigned char shift_mem_page_
	);

private:
	void run_opcode();
	void run_fc_behavioral();
	void run_conv2d_behavioral();
	void run_avgpool_behavioral();
	void run_relu_behavioral();
	void record_mmio_op(sc_uint<32> op);
	void clear_debug_snapshot();

	sc_signal<sc_uint<32> > reg_in_addr;
	sc_signal<sc_uint<32> > reg_out_addr;
	sc_signal<sc_uint<32> > reg_w_addr;
	sc_signal<sc_uint<16> > reg_in_dim;
	sc_signal<sc_uint<15> > reg_out_dim;
	sc_signal<bool>         reg_fc_relu;
	sc_signal<sc_uint<32> > reg_opcode;
	sc_signal<sc_uint<32> > reg_dims1;
	sc_signal<bool>         busy;
	sc_signal<bool>         done;

	sc_signal<sc_uint<32> > reg_dbg_op;
	sc_signal<sc_uint<32> > reg_dbg_runs;
	sc_signal<sc_uint<32> > reg_dbg_in0;
	sc_signal<sc_uint<32> > reg_dbg_w0;
	sc_signal<sc_uint<32> > reg_dbg_out0;
	sc_signal<sc_uint<32> > reg_dbg_in_sum;
	sc_signal<sc_uint<32> > reg_dbg_out_max;
	sc_signal<sc_uint<32> > reg_dbg_status;

	unsigned int pe_word_idx(unsigned int byte_addr) const;
	int          load_s32(unsigned int byte_addr, unsigned int &err_flags) const;
	void         store_s32(unsigned int byte_addr, int value, unsigned int &err_flags) const;

#if ACCEL_DEBUG
	void accel_log_flush();
	void accel_log_queue(const char *fmt, ...);
	char accel_log_line[384];
	bool accel_log_pending;
#endif
};

#endif
