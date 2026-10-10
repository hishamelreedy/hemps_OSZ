//------------------------------------------------------------------------------------------------
//
//  Behavioral Q8 tensor accelerator (SystemC, hybrid PE).
//  Opcodes: FC, CONV2D (NHWC valid stride-1), AVGPOOL2X2, RELU.
//  Large tensors live in PE RAM; MMIO only carries pointers + shapes + opcode.
//
//------------------------------------------------------------------------------------------------

#include "tensor_accel.h"
#include "../../agent_debug_log.h"
#ifdef TENSOR_ACCEL_UNIT_TB
#include "tb_ram.h"
#else
#include "../memory/ram.h"
#endif

#include <cstdio>
#include <cstdarg>
#include <cstring>

#ifndef ACCEL_DEBUG
#define ACCEL_DEBUG 1
#endif

tensor_accel::tensor_accel(
	sc_module_name name_,
	ram *mem_,
	regaddress router_address_,
	unsigned char shift_mem_page_
) :
	sc_module(name_),
	mem(mem_),
	router_address(router_address_),
	shift_mem_page(shift_mem_page_)
#if ACCEL_DEBUG
	, accel_log_pending(false)
#endif
{
#if ACCEL_DEBUG
	accel_log_line[0] = '\0';
	SC_METHOD(accel_log_flush);
	sensitive << clock.pos();
#endif
	SC_METHOD(mmio_read);
	sensitive << mem_address;
	sensitive << current_page;
	sensitive << reg_in_addr << reg_out_addr << reg_w_addr;
	sensitive << reg_in_dim << reg_out_dim << reg_fc_relu;
	sensitive << reg_opcode << reg_dims1;
	sensitive << busy << done;
	sensitive << reg_dbg_op << reg_dbg_runs << reg_dbg_in0 << reg_dbg_w0;
	sensitive << reg_dbg_out0 << reg_dbg_in_sum << reg_dbg_out_max << reg_dbg_status;

	SC_METHOD(update_pause);
	sensitive << busy;

	reset_regs();
}

void tensor_accel::reset_regs() {
	reg_in_addr.write(0);
	reg_out_addr.write(0);
	reg_w_addr.write(0);
	reg_in_dim.write(0);
	reg_out_dim.write(0);
	reg_fc_relu.write(true);
	reg_opcode.write(ACCEL_OP_FC);
	reg_dims1.write(0);
	busy.write(false);
	done.write(false);
	clear_debug_snapshot();
}

void tensor_accel::clear_debug_snapshot() {
	reg_dbg_op.write(ACCEL_DBG_OP_IDLE);
	reg_dbg_runs.write(0);
	reg_dbg_in0.write(0);
	reg_dbg_w0.write(0);
	reg_dbg_out0.write(0);
	reg_dbg_in_sum.write(0);
	reg_dbg_out_max.write(0);
	reg_dbg_status.write(ACCEL_DBG_ERR_NONE);
}

void tensor_accel::record_mmio_op(sc_uint<32> op) {
	reg_dbg_op.write(op);
}

void tensor_accel::update_pause() {
	pause_cpu.write(busy.read());
}

#if ACCEL_DEBUG
void tensor_accel::accel_log_queue(const char *fmt, ...) {
	va_list ap;

	snprintf(accel_log_line, sizeof(accel_log_line), "[ACCEL] pe=%x ",
		(unsigned int)router_address);
	va_start(ap, fmt);
	vsnprintf(accel_log_line + strlen(accel_log_line),
		sizeof(accel_log_line) - strlen(accel_log_line), fmt, ap);
	va_end(ap);
	accel_log_pending = true;
}

void tensor_accel::accel_log_flush() {
	unsigned int row, col;
	char path[64];
	FILE *fp;

	if (reset.read() == true) {
		accel_log_pending = false;
		accel_log_line[0] = '\0';
		return;
	}
	if (accel_log_pending == false)
		return;

	row = (unsigned int)router_address.range(15, 8);
	col = (unsigned int)router_address.range(7, 0);
	sprintf(path, "log/accel_log%dx%d.txt", row, col);
	fp = fopen(path, "a");
	if (fp != NULL) {
		fprintf(fp, "%s\n", accel_log_line);
		fclose(fp);
	}
	accel_log_pending = false;
	accel_log_line[0] = '\0';
}
#endif

unsigned int tensor_accel::pe_word_idx(unsigned int byte_addr) const {
	unsigned int addr = byte_addr;
	unsigned int page = (unsigned int)current_page.read();

	/* Match mlite ptr |= page: task C pointers are in-page offsets. */
	if (page != 0u) {
		addr &= (0xF0000000u | (0xFFFFFFFFu >> (32u - shift_mem_page)));
		addr |= page * PAGE_SIZE_BYTES;
	}

	/* Same local-RAM decode as CPU (cpu_enable_ram: bits [30:28] == 0). */
	if ((addr >> 28) != 0u)
		return RAM_SIZE;

	return addr >> 2;
}

int tensor_accel::load_s32(unsigned int byte_addr, unsigned int &err_flags) const {
	unsigned int idx = pe_word_idx(byte_addr);

	if (idx >= RAM_SIZE) {
		err_flags |= ACCEL_DBG_ERR_RAM_OOB;
		return 0;
	}

	return (int)(unsigned int)mem->ram_data[idx];
}

void tensor_accel::store_s32(unsigned int byte_addr, int value, unsigned int &err_flags) const {
	unsigned int idx = pe_word_idx(byte_addr);

	if (idx >= RAM_SIZE) {
		err_flags |= ACCEL_DBG_ERR_RAM_OOB;
		return;
	}

	mem->ram_data[idx] = (unsigned long)(unsigned int)value;
}

void tensor_accel::run_opcode() {
	unsigned int op = (unsigned int)reg_opcode.read();

	reg_dbg_runs.write(reg_dbg_runs.read() + 1);

	switch (op) {
	case ACCEL_OP_FC:
		run_fc_behavioral();
		break;
	case ACCEL_OP_CONV2D:
		run_conv2d_behavioral();
		break;
	case ACCEL_OP_AVGPOOL2X2:
		run_avgpool_behavioral();
		break;
	case ACCEL_OP_RELU:
		run_relu_behavioral();
		break;
	default:
		reg_dbg_status.write(ACCEL_DBG_ERR_BAD_OP);
#if ACCEL_DEBUG
		accel_log_queue("BAD_OP op=%u", op);
#endif
		break;
	}
}

void tensor_accel::run_fc_behavioral() {
	unsigned int in_base = (unsigned int)reg_in_addr.read();
	unsigned int out_base = (unsigned int)reg_out_addr.read();
	unsigned int w_base = (unsigned int)reg_w_addr.read();
	unsigned int in_dim = (unsigned int)reg_in_dim.read();
	unsigned int out_dim = (unsigned int)reg_out_dim.read();
	unsigned int apply_relu = reg_fc_relu.read() ? 1u : 0u;
	unsigned int bias_base = w_base + in_dim * out_dim * 4u;
	unsigned int err_flags = ACCEL_DBG_ERR_NONE;
	unsigned int o, i;
	long long sum;
	long long in_sum = 0;
	int x, w, out_val;
	int out_max = 0;
	int dbg_in0 = 0, dbg_w0 = 0, dbg_out0 = 0;

	record_mmio_op(ACCEL_DBG_OP_FC_START);

	if (in_dim == 0u || out_dim == 0u) {
		err_flags |= ACCEL_DBG_ERR_ZERO_DIM;
		reg_dbg_status.write(err_flags);
#if ACCEL_DEBUG
		accel_log_queue("FC abort ZERO_DIM in=%u out=%u", in_dim, out_dim);
#endif
		return;
	}
	if (in_base == 0u) err_flags |= ACCEL_DBG_ERR_BAD_IN;
	if (w_base == 0u) err_flags |= ACCEL_DBG_ERR_BAD_W;
	if (out_base == 0u) err_flags |= ACCEL_DBG_ERR_BAD_OUT;

	dbg_in0 = load_s32(in_base, err_flags);
	dbg_w0 = load_s32(w_base, err_flags);
	// #region agent log
	agent_debug_log("H-E", "tensor_accel.cpp:run_fc", "ptr_map",
		in_base, pe_word_idx(in_base),
		(unsigned int)current_page.read(),
		(unsigned int)dbg_in0, (unsigned int)dbg_w0);
	// #endregion

	for (o = 0; o < out_dim; o++) {
		sum = load_s32(bias_base + o * 4u, err_flags);
		for (i = 0; i < in_dim; i++) {
			x = load_s32(in_base + i * 4u, err_flags);
			if (o == 0) in_sum += x;
			w = load_s32(w_base + (i * out_dim + o) * 4u, err_flags);
			sum += ((long long)x * (long long)w) >> 8;
		}
		out_val = apply_relu ? ((sum > 0) ? (int)sum : 0) : (int)sum;
		store_s32(out_base + o * 4u, out_val, err_flags);
		if (o == 0 || out_val > out_max) out_max = out_val;
	}

	dbg_out0 = load_s32(out_base, err_flags);
	reg_dbg_in0.write((sc_uint<32>)(unsigned int)dbg_in0);
	reg_dbg_w0.write((sc_uint<32>)(unsigned int)dbg_w0);
	reg_dbg_out0.write((sc_uint<32>)(unsigned int)dbg_out0);
	reg_dbg_in_sum.write((sc_uint<32>)(unsigned int)in_sum);
	reg_dbg_out_max.write((sc_uint<32>)(unsigned int)out_max);
	reg_dbg_status.write(err_flags);

#if ACCEL_DEBUG
	accel_log_queue(
		"op=FC in=%u out=%u relu=%u in_sum=%lld out_max=%d err=0x%x",
		in_dim, out_dim, apply_relu, in_sum, out_max, err_flags);
#endif
}

void tensor_accel::run_conv2d_behavioral() {
	/*
	 * ACCEL_DIMS packed as: in_h | in_w<<8 | in_c<<16 | out_c<<24
	 * Stored via write to ACCEL_DIMS into in_dim/out_dim/relu split — instead
	 * keep full word in reg_dims1 high half? Simpler: SW packs into:
	 *   ACCEL_DIMS  = in_h | (in_w<<8) | (in_c<<16) | (out_c<<24)
	 *   We store that full word by writing in_dim = bits15:0, out_dim = bits30:16,
	 *   relu = bit31 — THAT BREAKS byte packing.
	 *
	 * Use ACCEL_DIMS1 for conv shape A and reuse ACCEL_DIMS low fields:
	 *   ACCEL_DIMS  = in_h | (in_w << 16)     (16+16)
	 *   ACCEL_DIMS1 = in_c | (out_c << 8) | (k << 16) | (relu << 24)
	 */
	unsigned int in_h = (unsigned int)reg_in_dim.read();
	unsigned int in_w = (unsigned int)reg_out_dim.read();
	unsigned int d1 = (unsigned int)reg_dims1.read();
	unsigned int in_c = d1 & 0xffu;
	unsigned int out_c = (d1 >> 8) & 0xffu;
	unsigned int k = (d1 >> 16) & 0xffu;
	unsigned int apply_relu = (d1 >> 24) & 0x1u;
	unsigned int in_base = (unsigned int)reg_in_addr.read();
	unsigned int out_base = (unsigned int)reg_out_addr.read();
	unsigned int w_base = (unsigned int)reg_w_addr.read();
	unsigned int err_flags = ACCEL_DBG_ERR_NONE;
	unsigned int out_h, out_w, w_per_oc, bias_base;
	unsigned int oc, oy, ox, ic, ky, kx;
	long long sum;
	int xi, wi, out_val;
	int out_max = 0;

	record_mmio_op(ACCEL_DBG_OP_CONV_START);

	if (in_h == 0u || in_w == 0u || in_c == 0u || out_c == 0u || k == 0u) {
		err_flags |= ACCEL_DBG_ERR_ZERO_DIM;
		reg_dbg_status.write(err_flags);
		return;
	}
	if (in_h < k || in_w < k) {
		err_flags |= ACCEL_DBG_ERR_ZERO_DIM;
		reg_dbg_status.write(err_flags);
		return;
	}
	if (in_base == 0u) err_flags |= ACCEL_DBG_ERR_BAD_IN;
	if (w_base == 0u) err_flags |= ACCEL_DBG_ERR_BAD_W;
	if (out_base == 0u) err_flags |= ACCEL_DBG_ERR_BAD_OUT;

	out_h = in_h - k + 1u;
	out_w = in_w - k + 1u;
	w_per_oc = in_c * k * k;
	bias_base = w_base + out_c * w_per_oc * 4u;

	reg_dbg_in0.write((sc_uint<32>)(unsigned int)load_s32(in_base, err_flags));
	reg_dbg_w0.write((sc_uint<32>)(unsigned int)load_s32(w_base, err_flags));

	for (oc = 0; oc < out_c; oc++) {
		int bias = load_s32(bias_base + oc * 4u, err_flags);
		unsigned int w_oc = w_base + oc * w_per_oc * 4u;

		for (oy = 0; oy < out_h; oy++) {
			for (ox = 0; ox < out_w; ox++) {
				sum = bias;
				for (ic = 0; ic < in_c; ic++) {
					for (ky = 0; ky < k; ky++) {
						for (kx = 0; kx < k; kx++) {
							unsigned int in_idx =
								(((oy + ky) * in_w + (ox + kx)) * in_c + ic) * 4u;
							unsigned int w_idx =
								(ic * (k * k) + ky * k + kx) * 4u;
							xi = load_s32(in_base + in_idx, err_flags);
							wi = load_s32(w_oc + w_idx, err_flags);
							sum += ((long long)xi * (long long)wi) >> 8;
						}
					}
				}
				out_val = apply_relu ? ((sum > 0) ? (int)sum : 0) : (int)sum;
				store_s32(out_base + ((oy * out_w + ox) * out_c + oc) * 4u,
					out_val, err_flags);
				if ((oc | oy | ox) == 0u || out_val > out_max)
					out_max = out_val;
			}
		}
	}

	reg_dbg_out0.write((sc_uint<32>)(unsigned int)load_s32(out_base, err_flags));
	reg_dbg_out_max.write((sc_uint<32>)(unsigned int)out_max);
	reg_dbg_status.write(err_flags);

#if ACCEL_DEBUG
	accel_log_queue(
		"op=CONV2D %ux%ux%u -> %ux%ux%u k=%u relu=%u err=0x%x",
		in_h, in_w, in_c, out_h, out_w, out_c, k, apply_relu, err_flags);
#endif
}

void tensor_accel::run_avgpool_behavioral() {
	/* ACCEL_DIMS: in_h in low 16, in_w in out_dim; ACCEL_DIMS1: channels */
	unsigned int in_h = (unsigned int)reg_in_dim.read();
	unsigned int in_w = (unsigned int)reg_out_dim.read();
	unsigned int channels = (unsigned int)reg_dims1.read() & 0xffffu;
	unsigned int in_base = (unsigned int)reg_in_addr.read();
	unsigned int out_base = (unsigned int)reg_out_addr.read();
	unsigned int err_flags = ACCEL_DBG_ERR_NONE;
	unsigned int out_h, out_w, oy, ox, c;
	int v00, v01, v10, v11, out_val;

	record_mmio_op(ACCEL_DBG_OP_POOL_START);

	if (in_h == 0u || in_w == 0u || channels == 0u || (in_h & 1u) || (in_w & 1u)) {
		err_flags |= ACCEL_DBG_ERR_ZERO_DIM;
		reg_dbg_status.write(err_flags);
		return;
	}
	if (in_base == 0u) err_flags |= ACCEL_DBG_ERR_BAD_IN;
	if (out_base == 0u) err_flags |= ACCEL_DBG_ERR_BAD_OUT;

	out_h = in_h / 2u;
	out_w = in_w / 2u;

	for (oy = 0; oy < out_h; oy++) {
		for (ox = 0; ox < out_w; ox++) {
			for (c = 0; c < channels; c++) {
				v00 = load_s32(in_base + (((oy * 2u) * in_w + (ox * 2u)) * channels + c) * 4u, err_flags);
				v01 = load_s32(in_base + (((oy * 2u) * in_w + (ox * 2u + 1u)) * channels + c) * 4u, err_flags);
				v10 = load_s32(in_base + (((oy * 2u + 1u) * in_w + (ox * 2u)) * channels + c) * 4u, err_flags);
				v11 = load_s32(in_base + (((oy * 2u + 1u) * in_w + (ox * 2u + 1u)) * channels + c) * 4u, err_flags);
				out_val = (v00 + v01 + v10 + v11) / 4;
				store_s32(out_base + ((oy * out_w + ox) * channels + c) * 4u, out_val, err_flags);
			}
		}
	}

	reg_dbg_status.write(err_flags);

#if ACCEL_DEBUG
	accel_log_queue("op=AVGPOOL2X2 %ux%ux%u err=0x%x",
		in_h, in_w, channels, err_flags);
#endif
}

void tensor_accel::run_relu_behavioral() {
	unsigned int n = (unsigned int)reg_in_dim.read()
		| (((unsigned int)reg_out_dim.read()) << 16);
	unsigned int in_base = (unsigned int)reg_in_addr.read();
	unsigned int out_base = (unsigned int)reg_out_addr.read();
	unsigned int err_flags = ACCEL_DBG_ERR_NONE;
	unsigned int i;
	int v;

	record_mmio_op(ACCEL_DBG_OP_RELU_START);

	if (n == 0u) {
		err_flags |= ACCEL_DBG_ERR_ZERO_DIM;
		reg_dbg_status.write(err_flags);
		return;
	}
	if (in_base == 0u) err_flags |= ACCEL_DBG_ERR_BAD_IN;
	if (out_base == 0u) err_flags |= ACCEL_DBG_ERR_BAD_OUT;

	for (i = 0; i < n; i++) {
		v = load_s32(in_base + i * 4u, err_flags);
		if (v < 0) v = 0;
		store_s32(out_base + i * 4u, v, err_flags);
	}

	reg_dbg_status.write(err_flags);

#if ACCEL_DEBUG
	accel_log_queue("op=RELU n=%u err=0x%x", n, err_flags);
#endif
}

void tensor_accel::mmio_read() {
	/* Same as other PE MMIO: decode latched address_reg (+ page strip for tasks). */
	sc_uint<32> addr =
		accel_bus_to_nominal(mem_address.read(), current_page.read());
	sc_uint<32> dims = (sc_uint<32>)reg_in_dim.read()
		| ((sc_uint<32>)reg_out_dim.read() << 16)
		| (reg_fc_relu.read() ? (sc_uint<32>)0x80000000u : (sc_uint<32>)0u);

	if (addr == ACCEL_CTRL || addr == ACCEL_STATUS ||
	    addr == ACCEL_IN_ADDR || addr == ACCEL_OUT_ADDR ||
	    addr == ACCEL_WEIGHT_ADDR || addr == ACCEL_DIMS ||
	    addr == ACCEL_CYCLES || addr == ACCEL_OPCODE || addr == ACCEL_DIMS1 ||
	    addr == ACCEL_DBG_OP || addr == ACCEL_DBG_RUNS ||
	    addr == ACCEL_DBG_IN0 || addr == ACCEL_DBG_W0 ||
	    addr == ACCEL_DBG_OUT0 || addr == ACCEL_DBG_IN_SUM ||
	    addr == ACCEL_DBG_OUT_MAX || addr == ACCEL_DBG_STATUS) {
		mem_hit.write(true);
	} else {
		mem_hit.write(false);
		mem_rdata.write(0);
		return;
	}

	if (addr == ACCEL_STATUS) {
		if (busy.read())
			mem_rdata.write(ACCEL_STATUS_BUSY);
		else if (done.read())
			mem_rdata.write(ACCEL_STATUS_DONE);
		else
			mem_rdata.write(0);
	} else if (addr == ACCEL_IN_ADDR) {
		mem_rdata.write(reg_in_addr.read());
	} else if (addr == ACCEL_OUT_ADDR) {
		mem_rdata.write(reg_out_addr.read());
	} else if (addr == ACCEL_WEIGHT_ADDR) {
		mem_rdata.write(reg_w_addr.read());
	} else if (addr == ACCEL_DIMS || addr == ACCEL_CYCLES) {
		mem_rdata.write(dims);
	} else if (addr == ACCEL_OPCODE) {
		mem_rdata.write(reg_opcode.read());
		// #region agent log
		agent_debug_log("H-B", "tensor_accel.cpp:mmio_read", "opcode_read",
			(unsigned int)mem_address.read(), (unsigned int)addr,
			(unsigned int)reg_opcode.read(), 1u,
			(unsigned int)current_page.read());
		// #endregion
	} else if (addr == ACCEL_DIMS1) {
		mem_rdata.write(reg_dims1.read());
	} else if (addr == ACCEL_DBG_OP) {
		mem_rdata.write(reg_dbg_op.read());
	} else if (addr == ACCEL_DBG_RUNS) {
		mem_rdata.write(reg_dbg_runs.read());
	} else if (addr == ACCEL_DBG_IN0) {
		mem_rdata.write(reg_dbg_in0.read());
	} else if (addr == ACCEL_DBG_W0) {
		mem_rdata.write(reg_dbg_w0.read());
	} else if (addr == ACCEL_DBG_OUT0) {
		mem_rdata.write(reg_dbg_out0.read());
	} else if (addr == ACCEL_DBG_IN_SUM) {
		mem_rdata.write(reg_dbg_in_sum.read());
	} else if (addr == ACCEL_DBG_OUT_MAX) {
		mem_rdata.write(reg_dbg_out_max.read());
	} else if (addr == ACCEL_DBG_STATUS) {
		mem_rdata.write(reg_dbg_status.read());
	} else {
		mem_rdata.write(0);
	}
}

void tensor_accel::service_mmio_write(sc_uint<32> addr, sc_uint<32> wdata) {
	if (busy.read() == true) {
#if ACCEL_DEBUG
		accel_log_queue("MMIO drop busy addr=0x%x wdata=0x%x",
			(unsigned int)addr, (unsigned int)wdata);
#endif
		return;
	}
	#if ACCEL_DEBUG
	accel_log_queue("[ACCEL]MMIO Service Write addr=0x%x wdata=0x%x",
		(unsigned int)addr, (unsigned int)wdata);
  #endif
	if (addr == ACCEL_IN_ADDR) {
		reg_in_addr.write(wdata);
		record_mmio_op(ACCEL_DBG_OP_WR_IN);
	} else if (addr == ACCEL_OUT_ADDR) {
		reg_out_addr.write(wdata);
		record_mmio_op(ACCEL_DBG_OP_WR_OUT);
	} else if (addr == ACCEL_WEIGHT_ADDR) {
		reg_w_addr.write(wdata);
		record_mmio_op(ACCEL_DBG_OP_WR_WEIGHT);
	} else if (addr == ACCEL_DIMS) {
		reg_in_dim.write(wdata(15, 0));
		reg_out_dim.write(wdata(30, 16));
		reg_fc_relu.write(wdata[31] == 1);
		record_mmio_op(ACCEL_DBG_OP_WR_DIMS);
	} else if (addr == ACCEL_OPCODE) {
		reg_opcode.write(wdata);
		record_mmio_op(ACCEL_DBG_OP_WR_OPCODE);
	} else if (addr == ACCEL_DIMS1) {
		reg_dims1.write(wdata);
		record_mmio_op(ACCEL_DBG_OP_WR_DIMS1);
	} else if (addr == ACCEL_CTRL) {
		record_mmio_op(ACCEL_DBG_OP_WR_CTRL);
		if (wdata == ACCEL_CTRL_CLEAR_DONE) {
			done.write(false);
			record_mmio_op(ACCEL_DBG_OP_FC_CLEAR);
		} else if (wdata == ACCEL_CTRL_START) {
#if ACCEL_DEBUG
			accel_log_queue(
				"START op=%u in=%u out=%u in_a=0x%x out_a=0x%x w_a=0x%x",
				(unsigned int)reg_opcode.read(),
				(unsigned int)reg_in_dim.read(),
				(unsigned int)reg_out_dim.read(),
				(unsigned int)reg_in_addr.read(),
				(unsigned int)reg_out_addr.read(),
				(unsigned int)reg_w_addr.read());
#endif
			busy.write(true);
			done.write(false);
			run_opcode();
			busy.write(false);
			done.write(true);
		}
	}
#if ACCEL_DEBUG
	if (accel_log_pending)
		accel_log_flush();
#endif
}
