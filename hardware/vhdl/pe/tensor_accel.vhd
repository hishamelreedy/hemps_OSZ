------------------------------------------------------------------------------------------------
-- Behavioral Q8 tensor accelerator (simulation model).
-- Opcodes: FC, CONV2D (NHWC valid stride-1), AVGPOOL2X2, RELU.
-- On START: nested loops compute the full op in one clock event via tensor_behav_pkg;
-- MIPS is stalled for one cycle (busy).
------------------------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.standards.all;
use work.tensor_behav_pkg.all;

entity tensor_accel is
    port(
        clock       : in  std_logic;
        reset       : in  std_logic;
        mem_address : in  std_logic_vector(31 downto 0);
        mem_wdata   : in  std_logic_vector(31 downto 0);
        mem_write   : in  std_logic;
        mem_rdata   : out std_logic_vector(31 downto 0);
        mem_hit     : out std_logic;
        pause_cpu   : out std_logic;
        ram_en      : out std_logic;
        ram_we      : out std_logic;
        ram_addr    : out std_logic_vector(31 downto 2);
        ram_wdata   : out std_logic_vector(31 downto 0);
        ram_rdata   : in  std_logic_vector(31 downto 0)
    );
end entity tensor_accel;

architecture rtl of tensor_accel is

    constant ACCEL_CTRL_START      : std_logic_vector(31 downto 0) := x"00000001";
    constant ACCEL_CTRL_CLEAR_DONE : std_logic_vector(31 downto 0) := x"00000002";
    constant ACCEL_STATUS_BUSY     : std_logic_vector(31 downto 0) := x"00000001";
    constant ACCEL_STATUS_DONE     : std_logic_vector(31 downto 0) := x"00000002";

    signal busy          : std_logic;
    signal done          : std_logic;

    signal reg_in_addr   : unsigned(31 downto 0);
    signal reg_out_addr  : unsigned(31 downto 0);
    signal reg_w_addr    : unsigned(31 downto 0);
    signal reg_in_dim    : unsigned(15 downto 0);
    signal reg_out_dim   : unsigned(14 downto 0);
    signal reg_fc_relu   : std_logic;
    signal reg_opcode    : unsigned(31 downto 0);
    signal reg_dims1     : unsigned(31 downto 0);

    function relu64(v : signed(63 downto 0)) return signed is
    begin
        if v < 0 then
            return (others => '0');
        else
            return resize(v, 32);
        end if;
    end function;

    function bias_base_fc(w_base, in_dim, out_dim : unsigned) return unsigned is
    begin
        return w_base + in_dim * out_dim * 4;
    end function;

    procedure run_fc_behavioral(
        in_base  : unsigned;
        out_base : unsigned;
        w_base   : unsigned;
        in_dim   : unsigned;
        out_dim  : unsigned;
        apply_relu : boolean
    ) is
        variable sum    : signed(63 downto 0);
        variable x      : signed(31 downto 0);
        variable w      : signed(31 downto 0);
        variable b_base : unsigned(31 downto 0);
        variable o      : integer;
        variable i      : integer;
        variable out_v  : signed(31 downto 0);
        variable in_i   : integer;
        variable out_i  : integer;
    begin
        in_i := to_integer(in_dim);
        out_i := to_integer(out_dim);
        if in_i <= 0 or out_i <= 0 then
            return;
        end if;
        b_base := bias_base_fc(w_base, in_dim, out_dim);

        for o in 0 to out_i - 1 loop
            sum := behav_load_s32(b_base + to_unsigned(o * 4, 32));

            for i in 0 to in_i - 1 loop
                x := behav_load_s32(in_base + to_unsigned(i * 4, 32));
                w := behav_load_s32(w_base + to_unsigned((i * out_i + o) * 4, 32));
                sum := sum + shift_right(x * w, 8);
            end loop;

            if apply_relu then
                out_v := relu64(sum);
            else
                out_v := resize(sum, 32);
            end if;
            behav_store_s32(out_base + to_unsigned(o * 4, 32), out_v);
        end loop;
    end procedure;

    procedure run_conv2d_behavioral(
        in_base  : unsigned;
        out_base : unsigned;
        w_base   : unsigned;
        in_h     : integer;
        in_w     : integer;
        in_c     : integer;
        out_c    : integer;
        k        : integer;
        apply_relu : boolean
    ) is
        variable out_h    : integer;
        variable out_w    : integer;
        variable w_per_oc : integer;
        variable b_base   : unsigned(31 downto 0);
        variable sum      : signed(63 downto 0);
        variable xi       : signed(31 downto 0);
        variable wi       : signed(31 downto 0);
        variable out_v    : signed(31 downto 0);
        variable bias     : signed(31 downto 0);
        variable oc, oy, ox, ic, ky, kx : integer;
        variable in_idx, w_idx, out_idx : integer;
    begin
        if in_h <= 0 or in_w <= 0 or in_c <= 0 or out_c <= 0 or k <= 0 then
            return;
        end if;
        if in_h < k or in_w < k then
            return;
        end if;

        out_h := in_h - k + 1;
        out_w := in_w - k + 1;
        w_per_oc := in_c * k * k;
        b_base := w_base + to_unsigned(out_c * w_per_oc * 4, 32);

        for oc in 0 to out_c - 1 loop
            bias := behav_load_s32(b_base + to_unsigned(oc * 4, 32));
            for oy in 0 to out_h - 1 loop
                for ox in 0 to out_w - 1 loop
                    sum := resize(bias, 64);
                    for ic in 0 to in_c - 1 loop
                        for ky in 0 to k - 1 loop
                            for kx in 0 to k - 1 loop
                                in_idx := (((oy + ky) * in_w + (ox + kx)) * in_c + ic) * 4;
                                w_idx := (oc * w_per_oc + ic * (k * k) + ky * k + kx) * 4;
                                xi := behav_load_s32(in_base + to_unsigned(in_idx, 32));
                                wi := behav_load_s32(w_base + to_unsigned(w_idx, 32));
                                sum := sum + shift_right(xi * wi, 8);
                            end loop;
                        end loop;
                    end loop;
                    if apply_relu then
                        out_v := relu64(sum);
                    else
                        out_v := resize(sum, 32);
                    end if;
                    out_idx := ((oy * out_w + ox) * out_c + oc) * 4;
                    behav_store_s32(out_base + to_unsigned(out_idx, 32), out_v);
                end loop;
            end loop;
        end loop;
    end procedure;

    procedure run_avgpool_behavioral(
        in_base  : unsigned;
        out_base : unsigned;
        in_h     : integer;
        in_w     : integer;
        channels : integer
    ) is
        variable out_h, out_w : integer;
        variable oy, ox, c    : integer;
        variable v00, v01, v10, v11 : signed(31 downto 0);
        variable out_v : signed(31 downto 0);
        variable idx   : integer;
    begin
        if in_h <= 0 or in_w <= 0 or channels <= 0 then
            return;
        end if;
        if (in_h rem 2) /= 0 or (in_w rem 2) /= 0 then
            return;
        end if;

        out_h := in_h / 2;
        out_w := in_w / 2;

        for oy in 0 to out_h - 1 loop
            for ox in 0 to out_w - 1 loop
                for c in 0 to channels - 1 loop
                    v00 := behav_load_s32(in_base + to_unsigned((((oy * 2) * in_w + (ox * 2)) * channels + c) * 4, 32));
                    v01 := behav_load_s32(in_base + to_unsigned((((oy * 2) * in_w + (ox * 2 + 1)) * channels + c) * 4, 32));
                    v10 := behav_load_s32(in_base + to_unsigned((((oy * 2 + 1) * in_w + (ox * 2)) * channels + c) * 4, 32));
                    v11 := behav_load_s32(in_base + to_unsigned((((oy * 2 + 1) * in_w + (ox * 2 + 1)) * channels + c) * 4, 32));
                    out_v := resize((resize(v00, 64) + resize(v01, 64) + resize(v10, 64) + resize(v11, 64)) / 4, 32);
                    idx := ((oy * out_w + ox) * channels + c) * 4;
                    behav_store_s32(out_base + to_unsigned(idx, 32), out_v);
                end loop;
            end loop;
        end loop;
    end procedure;

    procedure run_relu_behavioral(
        in_base  : unsigned;
        out_base : unsigned;
        n        : integer
    ) is
        variable i : integer;
        variable v : signed(31 downto 0);
    begin
        if n <= 0 then
            return;
        end if;
        for i in 0 to n - 1 loop
            v := behav_load_s32(in_base + to_unsigned(i * 4, 32));
            if v < 0 then
                v := (others => '0');
            end if;
            behav_store_s32(out_base + to_unsigned(i * 4, 32), v);
        end loop;
    end procedure;

    procedure run_opcode(
        opcode   : integer;
        in_base  : unsigned;
        out_base : unsigned;
        w_base   : unsigned;
        in_dim   : unsigned;
        out_dim  : unsigned;
        fc_relu  : std_logic;
        dims1    : unsigned
    ) is
        variable in_h, in_w, in_c, out_c, k, ch, n : integer;
        variable apply_relu : boolean;
    begin
        case opcode is
            when ACCEL_OP_FC =>
                run_fc_behavioral(in_base, out_base, w_base, in_dim,
                    resize(out_dim, 16), fc_relu = '1');
            when ACCEL_OP_CONV2D =>
                in_h := to_integer(in_dim);
                in_w := to_integer(out_dim);
                in_c := to_integer(dims1(7 downto 0));
                out_c := to_integer(dims1(15 downto 8));
                k := to_integer(dims1(23 downto 16));
                apply_relu := dims1(24) = '1';
                run_conv2d_behavioral(in_base, out_base, w_base,
                    in_h, in_w, in_c, out_c, k, apply_relu);
            when ACCEL_OP_AVGPOOL2X2 =>
                in_h := to_integer(in_dim);
                in_w := to_integer(out_dim);
                ch := to_integer(dims1(15 downto 0));
                run_avgpool_behavioral(in_base, out_base, in_h, in_w, ch);
            when ACCEL_OP_RELU =>
                n := to_integer(in_dim) + to_integer(out_dim) * 65536;
                run_relu_behavioral(in_base, out_base, n);
            when others =>
                null;
        end case;
    end procedure;

begin

    pause_cpu <= busy;

    mem_hit <= '1' when mem_address = ACCEL_CTRL or mem_address = ACCEL_STATUS or
                       mem_address = ACCEL_IN_ADDR or mem_address = ACCEL_OUT_ADDR or
                       mem_address = ACCEL_WEIGHT_ADDR or mem_address = ACCEL_DIMS or
                       mem_address = ACCEL_OPCODE or mem_address = ACCEL_DIMS1 else '0';

    mem_rdata <= ACCEL_STATUS_BUSY when (mem_address = ACCEL_STATUS and busy = '1') else
                 ACCEL_STATUS_DONE when (mem_address = ACCEL_STATUS and done = '1') else
                 (others => '0') when mem_address = ACCEL_STATUS else
                 std_logic_vector(reg_in_addr) when mem_address = ACCEL_IN_ADDR else
                 std_logic_vector(reg_out_addr) when mem_address = ACCEL_OUT_ADDR else
                 std_logic_vector(reg_w_addr) when mem_address = ACCEL_WEIGHT_ADDR else
                 reg_fc_relu & std_logic_vector(reg_out_dim) & std_logic_vector(reg_in_dim)
                     when mem_address = ACCEL_DIMS else
                 std_logic_vector(reg_opcode) when mem_address = ACCEL_OPCODE else
                 std_logic_vector(reg_dims1) when mem_address = ACCEL_DIMS1 else
                 (others => '0');

    ram_en    <= '0';
    ram_we    <= '0';
    ram_addr  <= (others => '0');
    ram_wdata <= (others => '0');

    process(clock, reset)
    begin
        if reset = '1' then
            busy <= '0';
            done <= '0';
            reg_in_addr <= (others => '0');
            reg_out_addr <= (others => '0');
            reg_w_addr <= (others => '0');
            reg_in_dim <= (others => '0');
            reg_out_dim <= (others => '0');
            reg_fc_relu <= '1';
            reg_opcode <= to_unsigned(ACCEL_OP_FC, 32);
            reg_dims1 <= (others => '0');
        elsif rising_edge(clock) then
            if busy = '1' then
                busy <= '0';
            elsif mem_write = '1' and busy = '0' then
                if mem_address = ACCEL_IN_ADDR then
                    reg_in_addr <= unsigned(mem_wdata);
                elsif mem_address = ACCEL_OUT_ADDR then
                    reg_out_addr <= unsigned(mem_wdata);
                elsif mem_address = ACCEL_WEIGHT_ADDR then
                    reg_w_addr <= unsigned(mem_wdata);
                elsif mem_address = ACCEL_DIMS then
                    reg_in_dim <= unsigned(mem_wdata(15 downto 0));
                    reg_out_dim <= unsigned(mem_wdata(30 downto 16));
                    reg_fc_relu <= mem_wdata(31);
                elsif mem_address = ACCEL_OPCODE then
                    reg_opcode <= unsigned(mem_wdata);
                elsif mem_address = ACCEL_DIMS1 then
                    reg_dims1 <= unsigned(mem_wdata);
                elsif mem_address = ACCEL_CTRL then
                    if mem_wdata = ACCEL_CTRL_CLEAR_DONE then
                        done <= '0';
                    elsif mem_wdata = ACCEL_CTRL_START then
                        run_opcode(
                            to_integer(reg_opcode),
                            reg_in_addr,
                            reg_out_addr,
                            reg_w_addr,
                            reg_in_dim,
                            reg_out_dim,
                            reg_fc_relu,
                            reg_dims1
                        );
                        done <= '1';
                        busy <= '1';
                    end if;
                end if;
            end if;
        end if;
    end process;

end architecture rtl;
