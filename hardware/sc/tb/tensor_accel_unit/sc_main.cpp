//------------------------------------------------------------------------------------------------
// Standalone SystemC unit TB for tensor_accel (no NoC / Plasma / kernel).
// Builds against real pe/tensor_accel sources + stub tb_ram.h.
//------------------------------------------------------------------------------------------------

#include <systemc.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#include "tb_ram.h"
#include "../../pe/tensor_accel/tensor_accel.h"
#include "lenet5_weights.h"
#include "lenet5_input.h"
#include "lenet5_tb_batch.h"

using std::vector;
using std::string;

/* LeNet dims (match software/include/lenet5_shared.h) */
static const int LN_IN_H = 28, LN_IN_W = 28, LN_IN_C = 1;
static const int LN_C1_OUT_C = 6, LN_C1_K = 5, LN_C1_H = 24, LN_C1_W = 24;
static const int LN_P1_H = 12, LN_P1_W = 12, LN_P1_DIM = 12 * 12 * 6;
static const int LN_C2_OUT_C = 16, LN_C2_K = 5, LN_C2_H = 8, LN_C2_W = 8;
static const int LN_P2_H = 4, LN_P2_W = 4, LN_P2_DIM = 4 * 4 * 16;
static const int LN_FC1_OUT = 120, LN_FC2_OUT = 84, LN_OUT_DIM = 10;

static int g_fails = 0;
static int g_checks = 0;

static void check_eq(const char *name, int got, int exp) {
	g_checks++;
	if (got != exp) {
		std::printf("FAIL %s: got=%d exp=%d\n", name, got, exp);
		g_fails++;
	} else {
		std::printf("OK   %s\n", name);
	}
}

static void check_vec(const char *name, const vector<int> &got, const vector<int> &exp) {
	g_checks++;
	if (got.size() != exp.size()) {
		std::printf("FAIL %s: size got=%zu exp=%zu\n", name, got.size(), exp.size());
		g_fails++;
		return;
	}
	for (size_t i = 0; i < got.size(); i++) {
		if (got[i] != exp[i]) {
			std::printf("FAIL %s: first mismatch i=%zu got=%d exp=%d\n",
				name, i, got[i], exp[i]);
			g_fails++;
			return;
		}
	}
	std::printf("OK   %s (n=%zu)\n", name, got.size());
}

static int relu(int x) { return x > 0 ? x : 0; }

static int argmax(const vector<int> &v) {
	int best = 0;
	for (size_t i = 1; i < v.size(); i++)
		if (v[i] > v[best]) best = (int)i;
	return best;
}

static vector<int> to_vec(const int *p, int n) {
	return vector<int>(p, p + n);
}

static vector<int> ref_fc(const vector<int> &in, const vector<int> &w,
		int in_dim, int out_dim, int apply_relu) {
	vector<int> out(out_dim);
	const int *bias = &w[in_dim * out_dim];
	for (int o = 0; o < out_dim; o++) {
		long long s = bias[o];
		for (int i = 0; i < in_dim; i++)
			s += ((long long)in[i] * w[i * out_dim + o]) >> 8;
		out[o] = apply_relu ? relu((int)s) : (int)s;
	}
	return out;
}

static vector<int> ref_conv(const vector<int> &in, const vector<int> &w,
		int in_h, int in_w, int in_c, int out_c, int k, int apply_relu) {
	int out_h = in_h - k + 1;
	int out_w = in_w - k + 1;
	int w_per = in_c * k * k;
	vector<int> out(out_h * out_w * out_c);
	const int *bias = &w[out_c * w_per];
	for (int oc = 0; oc < out_c; oc++) {
		for (int oy = 0; oy < out_h; oy++) {
			for (int ox = 0; ox < out_w; ox++) {
				long long s = bias[oc];
				for (int ic = 0; ic < in_c; ic++) {
					for (int ky = 0; ky < k; ky++) {
						for (int kx = 0; kx < k; kx++) {
							int xi = in[((oy + ky) * in_w + (ox + kx)) * in_c + ic];
							int wi = w[oc * w_per + ic * (k * k) + ky * k + kx];
							s += ((long long)xi * wi) >> 8;
						}
					}
				}
				out[(oy * out_w + ox) * out_c + oc] =
					apply_relu ? relu((int)s) : (int)s;
			}
		}
	}
	return out;
}

static vector<int> ref_pool(const vector<int> &in, int in_h, int in_w, int ch) {
	int out_h = in_h / 2, out_w = in_w / 2;
	vector<int> out(out_h * out_w * ch);
	for (int oy = 0; oy < out_h; oy++) {
		for (int ox = 0; ox < out_w; ox++) {
			for (int c = 0; c < ch; c++) {
				int v00 = in[((oy * 2) * in_w + (ox * 2)) * ch + c];
				int v01 = in[((oy * 2) * in_w + (ox * 2 + 1)) * ch + c];
				int v10 = in[((oy * 2 + 1) * in_w + (ox * 2)) * ch + c];
				int v11 = in[((oy * 2 + 1) * in_w + (ox * 2 + 1)) * ch + c];
				out[(oy * out_w + ox) * ch + c] = (v00 + v01 + v10 + v11) / 4;
			}
		}
	}
	return out;
}

SC_MODULE(accel_tb) {
	sc_clock clock;
	sc_signal<bool> reset;
	sc_signal<sc_uint<32> > mem_address;
	sc_signal<sc_uint<32> > mem_read_address;
	sc_signal<sc_uint<32> > mem_rdata;
	sc_signal<bool> mem_hit;
	sc_signal<sc_uint<32> > mem_write_address;
	sc_signal<sc_uint<32> > mem_write_data;
	sc_signal<bool> mem_write;
	sc_signal<bool> pause_cpu;
	sc_signal<sc_uint<32> > tick_counter;
	sc_signal<sc_uint<8> > current_page;

	ram *mem;
	tensor_accel *dut;

	/* Byte bases in stub PE RAM (weights reused per slave stage). */
	static const unsigned BUF_A = 0x00001000u; /* activation A */
	static const unsigned BUF_B = 0x00008000u; /* activation B / tmp */
	static const unsigned W_BASE = 0x00020000u; /* weight blob (largest: FC1 ~30k words) */

	/* Legacy aliases for small opcode smoke tests */
	static const unsigned IN_BASE = BUF_A;
	static const unsigned OUT_BASE = BUF_B;

	SC_CTOR(accel_tb) :
		clock("clock", 10, SC_NS),
		mem(0),
		dut(0)
	{
		mem = new ram();
		unsigned char shift = 18; /* log2(PAGE_SIZE_BYTES=262144) */
		dut = new tensor_accel("dut", mem, 0, shift);

		dut->clock(clock);
		dut->reset(reset);
		dut->mem_address(mem_address);
		dut->mem_read_address(mem_read_address);
		dut->mem_rdata(mem_rdata);
		dut->mem_hit(mem_hit);
		dut->mem_write_address(mem_write_address);
		dut->mem_write_data(mem_write_data);
		dut->mem_write(mem_write);
		dut->pause_cpu(pause_cpu);
		dut->tick_counter(tick_counter);
		dut->current_page(current_page);

		SC_THREAD(run);
	}

	~accel_tb() {
		delete dut;
		delete mem;
	}

	void poke_word(unsigned byte_addr, int value) {
		unsigned idx = byte_addr >> 2;
		if (idx >= RAM_SIZE) {
			std::printf("poke OOB addr=0x%x idx=%u RAM_SIZE=%u\n",
				byte_addr, idx, (unsigned)RAM_SIZE);
			sc_stop();
			return;
		}
		mem->ram_data[idx] = (unsigned long)(unsigned int)value;
	}

	int peek_word(unsigned byte_addr) {
		unsigned idx = byte_addr >> 2;
		if (idx >= RAM_SIZE)
			return 0;
		return (int)(unsigned int)mem->ram_data[idx];
	}

	void store_vec(unsigned base, const vector<int> &v) {
		for (size_t i = 0; i < v.size(); i++)
			poke_word(base + (unsigned)i * 4u, v[i]);
	}

	vector<int> load_vec(unsigned base, size_t n) {
		vector<int> v(n);
		for (size_t i = 0; i < n; i++)
			v[i] = peek_word(base + (unsigned)i * 4u);
		return v;
	}

	void mmio_wr(unsigned addr, unsigned data) {
		dut->service_mmio_write(addr, data);
		wait(SC_ZERO_TIME);
	}

	unsigned mmio_rd(unsigned addr) {
		mem_address.write(addr);
		mem_read_address.write(addr);
		wait(SC_ZERO_TIME);
		return (unsigned)mem_rdata.read();
	}

	void start_op() {
		mmio_wr(ACCEL_CTRL, ACCEL_CTRL_START);
		mmio_wr(ACCEL_CTRL, ACCEL_CTRL_CLEAR_DONE);
	}

	void hw_conv(unsigned in_b, unsigned out_b, unsigned w_b,
			int in_h, int in_w, int in_c, int out_c, int k, int apply_relu) {
		mmio_wr(ACCEL_OPCODE, ACCEL_OP_CONV2D);
		mmio_wr(ACCEL_IN_ADDR, in_b);
		mmio_wr(ACCEL_OUT_ADDR, out_b);
		mmio_wr(ACCEL_WEIGHT_ADDR, w_b);
		mmio_wr(ACCEL_DIMS, (unsigned)in_h | ((unsigned)in_w << 16));
		mmio_wr(ACCEL_DIMS1, (unsigned)in_c | ((unsigned)out_c << 8) |
			((unsigned)k << 16) | (apply_relu ? (1u << 24) : 0u));
		start_op();
	}

	void hw_pool(unsigned in_b, unsigned out_b, int in_h, int in_w, int ch) {
		mmio_wr(ACCEL_OPCODE, ACCEL_OP_AVGPOOL2X2);
		mmio_wr(ACCEL_IN_ADDR, in_b);
		mmio_wr(ACCEL_OUT_ADDR, out_b);
		mmio_wr(ACCEL_DIMS, (unsigned)in_h | ((unsigned)in_w << 16));
		mmio_wr(ACCEL_DIMS1, (unsigned)ch);
		start_op();
	}

	void hw_fc(unsigned in_b, unsigned out_b, unsigned w_b,
			int in_dim, int out_dim, int apply_relu) {
		unsigned dims = (unsigned)in_dim | ((unsigned)out_dim << 16) |
			(apply_relu ? ACCEL_DIMS_RELU_MASK : 0u);
		mmio_wr(ACCEL_OPCODE, ACCEL_OP_FC);
		mmio_wr(ACCEL_IN_ADDR, in_b);
		mmio_wr(ACCEL_OUT_ADDR, out_b);
		mmio_wr(ACCEL_WEIGHT_ADDR, w_b);
		mmio_wr(ACCEL_DIMS, dims);
		start_op();
	}

	void test_fc() {
		const int in_dim = 4, out_dim = 2;
		vector<int> in = {1, 2, 3, 4};
		vector<int> w = {1, 0, 0, 1, 0, 0, 0, 0, 5, -3}; /* 4x2 + bias */
		store_vec(IN_BASE, in);
		store_vec(W_BASE, w);
		for (int i = 0; i < out_dim; i++)
			poke_word(OUT_BASE + (unsigned)i * 4u, 0xDEAD);

		hw_fc(IN_BASE, OUT_BASE, W_BASE, in_dim, out_dim, 1);
		check_eq("FC dbg_status", (int)mmio_rd(ACCEL_DBG_STATUS), 0);
		check_vec("FC", load_vec(OUT_BASE, out_dim), ref_fc(in, w, in_dim, out_dim, 1));
	}

	void test_conv_pool_relu() {
		const int in_h = 4, in_w = 4, in_c = 1, out_c = 2, k = 3;
		vector<int> in(in_h * in_w * in_c);
		for (int i = 0; i < (int)in.size(); i++)
			in[i] = ((i * 17) % 251) - 100;
		int w_per = in_c * k * k;
		vector<int> w(out_c * w_per + out_c);
		for (int i = 0; i < out_c * w_per; i++)
			w[i] = ((i * 13) % 127) - 60;
		for (int i = 0; i < out_c; i++)
			w[out_c * w_per + i] = i * 10;

		store_vec(IN_BASE, in);
		store_vec(W_BASE, w);
		hw_conv(IN_BASE, OUT_BASE, W_BASE, in_h, in_w, in_c, out_c, k, 1);

		int out_h = in_h - k + 1, out_w = in_w - k + 1;
		vector<int> conv_exp = ref_conv(in, w, in_h, in_w, in_c, out_c, k, 1);
		check_eq("CONV dbg_status", (int)mmio_rd(ACCEL_DBG_STATUS), 0);
		check_vec("CONV2D+ReLU", load_vec(OUT_BASE, conv_exp.size()), conv_exp);

		store_vec(IN_BASE, conv_exp);
		hw_pool(IN_BASE, OUT_BASE, out_h, out_w, out_c);
		vector<int> pool_exp = ref_pool(conv_exp, out_h, out_w, out_c);
		check_vec("AVGPOOL2X2", load_vec(OUT_BASE, pool_exp.size()), pool_exp);

		vector<int> relu_in = {-5, 0, 3, -1, 100};
		store_vec(IN_BASE, relu_in);
		mmio_wr(ACCEL_OPCODE, ACCEL_OP_RELU);
		mmio_wr(ACCEL_IN_ADDR, IN_BASE);
		mmio_wr(ACCEL_OUT_ADDR, OUT_BASE);
		mmio_wr(ACCEL_DIMS, (unsigned)relu_in.size());
		start_op();
		vector<int> relu_exp = {0, 0, 3, 0, 100};
		check_vec("RELU", load_vec(OUT_BASE, relu_exp.size()), relu_exp);
	}

	void test_lenet_c1_smoke() {
		const int in_h = 8, in_w = 8, in_c = 1, out_c = 2, k = 5;
		vector<int> in(in_h * in_w * in_c);
		for (int i = 0; i < (int)in.size(); i++)
			in[i] = (i % 17) - 8;
		int w_per = in_c * k * k;
		vector<int> w(out_c * w_per + out_c, 0);
		for (int i = 0; i < out_c * w_per; i++)
			w[i] = ((i * 7) % 50) - 25;
		w[out_c * w_per + 0] = 2;
		w[out_c * w_per + 1] = -1;

		store_vec(IN_BASE, in);
		store_vec(W_BASE, w);
		hw_conv(IN_BASE, OUT_BASE, W_BASE, in_h, in_w, in_c, out_c, k, 1);
		vector<int> exp = ref_conv(in, w, in_h, in_w, in_c, out_c, k, 1);
		check_vec("LeNet-shaped C1 (8x8 k=5)", load_vec(OUT_BASE, exp.size()), exp);
	}

	/*
	 * Full LeNet-5 on one PE RAM (no NoC). Stages mirror applications/lenet5 slaves:
	 *   master   -> provides MNIST input
	 *   slave 1  -> lenet_c1p1  (conv1 + relu + pool1)
	 *   slave 2  -> lenet_c2p2  (conv2 + relu + pool2)
	 *   slave 3  -> lenet_fc1
	 *   slave 4  -> lenet_fc2
	 *   slave 5  -> lenet_fc3
	 * Weight blobs are loaded one stage at a time into W_BASE (same PE-RAM model).
	 * Loops LENET5_TB_BATCH_SIZE samples (default 100 via lenet5_tb_batch.h).
	 */
	void test_full_lenet5_one(int sample, const vector<int> &w_c1, const vector<int> &w_c2,
			const vector<int> &w_fc1, const vector<int> &w_fc2, const vector<int> &w_fc3,
			int &sw_hw_ok, int &label_ok) {
		const int dim = LN_IN_H * LN_IN_W * LN_IN_C;
		vector<int> input = to_vec(LENET5_TB_BATCH_SAMPLES + sample * dim, dim);
		int label = LENET5_TB_BATCH_LABELS[sample];

		/* // master code: load MNIST sample into BUF_A */

		/* -------- slave 1 / lenet_c1p1 -------- */
		/* // slave 1 code: C1 conv+relu then avgpool -> P1 */
		store_vec(BUF_A, input);
		store_vec(W_BASE, w_c1);
		hw_conv(BUF_A, BUF_B, W_BASE, LN_IN_H, LN_IN_W, LN_IN_C, LN_C1_OUT_C, LN_C1_K, 1);
		vector<int> c1_sw = ref_conv(input, w_c1, LN_IN_H, LN_IN_W, LN_IN_C, LN_C1_OUT_C, LN_C1_K, 1);
		vector<int> c1_hw = load_vec(BUF_B, c1_sw.size());

		store_vec(BUF_A, c1_hw);
		hw_pool(BUF_A, BUF_B, LN_C1_H, LN_C1_W, LN_C1_OUT_C);
		vector<int> p1_sw = ref_pool(c1_sw, LN_C1_H, LN_C1_W, LN_C1_OUT_C);
		vector<int> p1_hw = load_vec(BUF_B, (size_t)LN_P1_DIM);

		/* -------- slave 2 / lenet_c2p2 -------- */
		/* // slave 2 code: C2 conv+relu then avgpool -> P2 */
		store_vec(BUF_A, p1_hw);
		store_vec(W_BASE, w_c2);
		hw_conv(BUF_A, BUF_B, W_BASE, LN_P1_H, LN_P1_W, LN_C1_OUT_C, LN_C2_OUT_C, LN_C2_K, 1);
		vector<int> c2_sw = ref_conv(p1_sw, w_c2, LN_P1_H, LN_P1_W, LN_C1_OUT_C, LN_C2_OUT_C, LN_C2_K, 1);
		vector<int> c2_hw = load_vec(BUF_B, c2_sw.size());

		store_vec(BUF_A, c2_hw);
		hw_pool(BUF_A, BUF_B, LN_C2_H, LN_C2_W, LN_C2_OUT_C);
		vector<int> p2_sw = ref_pool(c2_sw, LN_C2_H, LN_C2_W, LN_C2_OUT_C);
		vector<int> p2_hw = load_vec(BUF_B, (size_t)LN_P2_DIM);

		/* -------- slave 3 / lenet_fc1 -------- */
		/* // slave 3 code: FC 256 -> 120 + ReLU */
		store_vec(BUF_A, p2_hw);
		store_vec(W_BASE, w_fc1);
		hw_fc(BUF_A, BUF_B, W_BASE, LN_P2_DIM, LN_FC1_OUT, 1);
		vector<int> fc1_sw = ref_fc(p2_sw, w_fc1, LN_P2_DIM, LN_FC1_OUT, 1);
		vector<int> fc1_hw = load_vec(BUF_B, (size_t)LN_FC1_OUT);

		/* -------- slave 4 / lenet_fc2 -------- */
		/* // slave 4 code: FC 120 -> 84 + ReLU */
		store_vec(BUF_A, fc1_hw);
		store_vec(W_BASE, w_fc2);
		hw_fc(BUF_A, BUF_B, W_BASE, LN_FC1_OUT, LN_FC2_OUT, 1);
		vector<int> fc2_sw = ref_fc(fc1_sw, w_fc2, LN_FC1_OUT, LN_FC2_OUT, 1);
		vector<int> fc2_hw = load_vec(BUF_B, (size_t)LN_FC2_OUT);

		/* -------- slave 5 / lenet_fc3 -------- */
		/* // slave 5 code: FC 84 -> 10 (logits, no ReLU) */
		store_vec(BUF_A, fc2_hw);
		store_vec(W_BASE, w_fc3);
		hw_fc(BUF_A, BUF_B, W_BASE, LN_FC2_OUT, LN_OUT_DIM, 0);
		vector<int> fc3_sw = ref_fc(fc2_sw, w_fc3, LN_FC2_OUT, LN_OUT_DIM, 0);
		vector<int> fc3_hw = load_vec(BUF_B, (size_t)LN_OUT_DIM);

		bool match_layers =
			(c1_hw == c1_sw) && (p1_hw == p1_sw) &&
			(c2_hw == c2_sw) && (p2_hw == p2_sw) &&
			(fc1_hw == fc1_sw) && (fc2_hw == fc2_sw) && (fc3_hw == fc3_sw);
		int pred_sw = argmax(fc3_sw);
		int pred_hw = argmax(fc3_hw);
		if (match_layers && pred_sw == pred_hw)
			sw_hw_ok++;
		if (pred_hw == label)
			label_ok++;

		if (sample < 3 || sample == LENET5_TB_BATCH_SIZE - 1 || !match_layers || pred_hw != label) {
			std::printf("  sample %d idx=%d pred_sw=%d pred_hw=%d label=%d sw_hw=%s\n",
				sample, LENET5_TB_BATCH_INDICES[sample], pred_sw, pred_hw, label,
				match_layers && pred_sw == pred_hw ? "OK" : "FAIL");
		}
		if (!match_layers || pred_sw != pred_hw) {
			g_checks++;
			g_fails++;
			std::printf("FAIL LeNet sample %d SW vs HW mismatch\n", sample);
		} else {
			g_checks++;
		}
	}

	void test_full_lenet5() {
		std::printf("--- test_full_lenet5 batch=%d (SW ref vs HW accel) ---\n",
			LENET5_TB_BATCH_SIZE);

		vector<int> w_c1 = to_vec(LENET5_C1_WEIGHTS, LENET5_C1_WEIGHTS_WORDS);
		vector<int> w_c2 = to_vec(LENET5_C2_WEIGHTS, LENET5_C2_WEIGHTS_WORDS);
		vector<int> w_fc1 = to_vec(LENET5_FC1_WEIGHTS, LENET5_FC1_WEIGHTS_WORDS);
		vector<int> w_fc2 = to_vec(LENET5_FC2_WEIGHTS, LENET5_FC2_WEIGHTS_WORDS);
		vector<int> w_fc3 = to_vec(LENET5_FC3_WEIGHTS, LENET5_FC3_WEIGHTS_WORDS);

		int sw_hw_ok = 0, label_ok = 0;
		for (int s = 0; s < LENET5_TB_BATCH_SIZE; s++)
			test_full_lenet5_one(s, w_c1, w_c2, w_fc1, w_fc2, w_fc3, sw_hw_ok, label_ok);

		std::printf("LeNet batch: sw_hw_match=%d/%d  pred_correct=%d/%d (acc=%.2f%%)\n",
			sw_hw_ok, LENET5_TB_BATCH_SIZE,
			label_ok, LENET5_TB_BATCH_SIZE,
			100.0 * label_ok / (double)LENET5_TB_BATCH_SIZE);
		check_eq("LeNet batch all SW==HW", sw_hw_ok, LENET5_TB_BATCH_SIZE);
	}

	void run() {
		reset.write(true);
		current_page.write(0);
		tick_counter.write(0);
		mem_address.write(0);
		mem_read_address.write(0);
		mem_write.write(false);
		wait(20, SC_NS);
		reset.write(false);
		dut->reset_regs();
		wait(SC_ZERO_TIME);

		std::printf("=== tensor_accel SystemC unit TB ===\n");
		test_fc();
		test_conv_pool_relu();
		test_lenet_c1_smoke();
		test_full_lenet5();

		std::printf("=== summary: %d checks, %d fails ===\n", g_checks, g_fails);
		if (g_fails)
			std::printf("RESULT: FAIL\n");
		else
			std::printf("RESULT: PASS\n");

		sc_stop();
	}
};

int sc_main(int argc, char *argv[]) {
	(void)argc;
	(void)argv;
	accel_tb tb("tb");
	sc_start();
	return g_fails ? 1 : 0;
}
