------------------------------------------------------------------------------------------------
-- Behavioral PE RAM shadow for single-cycle tensor accelerator (simulation model).
-- Keeps a word array in sync with port-A/B writes; accel reads/writes via for-loops.
------------------------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

package tensor_behav_pkg is

    constant BEHAV_MEM_WORDS : positive := 32768;

    type behav_mem_t is array (0 to BEHAV_MEM_WORDS - 1) of std_logic_vector(31 downto 0);

    shared variable g_behav_mem : behav_mem_t;

    impure function behav_load_word(word_idx : natural) return std_logic_vector;
    impure function behav_load_s32(byte_addr : unsigned) return signed;
    impure procedure behav_store_s32(byte_addr : unsigned; value : signed);
    impure procedure behav_store_word(word_idx : natural; word : std_logic_vector(31 downto 0));
    impure procedure behav_load_from_file(file_path : string);
    impure procedure behav_init_pe_shadow(row, col : natural);

end package tensor_behav_pkg;

package body tensor_behav_pkg is

    function to_word_idx(byte_addr : unsigned) return natural is
    begin
        return to_integer(byte_addr(31 downto 2));
    end function;

    impure function behav_load_word(word_idx : natural) return std_logic_vector is
    begin
        if word_idx >= BEHAV_MEM_WORDS then
            return (others => '0');
        end if;
        return g_behav_mem(word_idx);
    end function;

    impure function behav_load_s32(byte_addr : unsigned) return signed is
    begin
        return signed(behav_load_word(to_word_idx(byte_addr)));
    end function;

    impure procedure behav_store_s32(byte_addr : unsigned; value : signed) is
        variable idx : natural;
        variable v   : signed(63 downto 0);
    begin
        idx := to_word_idx(byte_addr);
        if idx >= BEHAV_MEM_WORDS then
            return;
        end if;
        v := value;
        if v < 0 then
            g_behav_mem(idx) := (others => '0');
        else
            g_behav_mem(idx) := std_logic_vector(resize(v, 32));
        end if;
    end procedure;

    impure procedure behav_store_word(word_idx : natural; word : std_logic_vector(31 downto 0)) is
    begin
        if word_idx < BEHAV_MEM_WORDS then
            g_behav_mem(word_idx) := word;
        end if;
    end procedure;

    impure procedure behav_load_from_file(file_path : string) is
        file ram_file : text;
        variable l     : line;
        variable idx   : natural := 0;
        variable word  : std_logic_vector(31 downto 0);
        variable good  : boolean;
    begin
        file_open(ram_file, file_path, read_mode);
        while not endfile(ram_file) and idx < BEHAV_MEM_WORDS loop
            readline(ram_file, l);
            read(l, word, good);
            if good then
                g_behav_mem(idx) := word;
                idx := idx + 1;
            end if;
        end loop;
        file_close(ram_file);
    end procedure;

    impure procedure behav_init_pe_shadow(row, col : natural) is
        variable fpath : line;
        variable path  : string(1 to 40);
    begin
        write(fpath, string'("ram_pe/ram"));
        write(fpath, row);
        write(fpath, string'("x"));
        write(fpath, col);
        write(fpath, string'(".txt"));
        read(fpath, path);
        behav_load_from_file(path);
    end procedure;

end package body tensor_behav_pkg;
