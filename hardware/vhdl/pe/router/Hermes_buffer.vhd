-----------------------------------------------------------------------------------------------
--
--  DISTRIBUTED HEMPS  - version 5.0
--
--  Research group: GAPH-PUCRS    -    contact   fernando.moraes@pucrs.br
--
--  Distribution:  June 2016
--
--  Source name:  Hermes_buffer.vhd
--
--  Brief description: Description of queues of flits
--
---------------------------------------------------------------------------------------
--                                      BUFFER
--                         --------------
--                   RX ->|             |-> H
--              DATA_IN ->|             |<- ACK_H
--             CLOCK_RX ->|             |
--             CREDIT_O <-|             |-> DATA_AV
--                        |             |-> DATA
--                        |             |<- DATA_ACK
--                        |             |
--                        |             |   SENDER
--                        |             |=> de todas
--                        |             |   as portas
--                         --------------
--
--  Quando o algoritmo de chaveamento resulta no bloqueio dos flits de um pacote,
--  ocorre uma perda de desempenho em toda rede de interconex�o, porque os flits s�o
--  bloqueados n�o somente na chave atual, mas em todas as intermedi�rias.
--  Para diminuir a perda de desempenho foi adicionada uma fila em cada porta de
--  entrada da chave, reduzindo as chaves afetadas com o bloqueio dos flits de um
--  pacote. � importante observar que quanto maior for o tamanho da fila menor ser� o
--  n�mero de chaves intermedi�rias afetadas.
--  As filas usadas cont�m dimens�o e largura de flit parametriz�veis, para alter�-las
--  modifique as constantes TAM_BUFFER e TAM_FLIT no arquivo "packet.vhd".
--  As filas funcionam como FIFOs circulares. Cada fila possui dois ponteiros: first e
--  last. First aponta para a posi��o da fila onde se encontra o flit a ser consumido.
--  Last aponta para a posi��o onde deve ser inserido o pr�ximo flit.
---------------------------------------------------------------------------------------
library IEEE;
use IEEE.std_logic_1164.all;  
use IEEE.std_logic_unsigned.all;
use IEEE.std_logic_arith.all;
use work.standards.all;
use work.hemps_pkg.all;

-- interface da Hermes_buffer
entity Hermes_buffer is
generic(
		TAM_BUFFER  : integer := 8;
		buffer_type : string := "SIM"
);
port(
		clock       	: in  std_logic;
		reset       	: in  std_logic;      
		rx          	: in  std_logic;
		eop_in      	: in  std_logic;
		bop_in      	: in  std_logic;
		data_in     	: in  regflit;
		credit_out  	: out std_logic;
		req_routing 	: out std_logic;
		ack_routing 	: in  std_logic; 
		tx          	: out std_logic;
		eop_out     	: out std_logic;
		bop_out     	: out std_logic;
		data_out    	: out regflit;
		--data_header 	: out regflit_32;
		header_routing 	: out regflit_32;
		header_fixed 	: out regflit_32;
		credit_in   	: in  std_logic;
		sender      	: out std_logic;
		next_flit   	: in  std_logic;
		change_routing 		: in  std_logic;
		-- Output port chosen by SwitchControl for this packet (LOCAL = no hop).
		route_out		: in  integer range 0 to NPORT-1
		);
end Hermes_buffer;

architecture Hermes_buffer of Hermes_buffer is

type fila_out is (S_INIT, S_PAYLOAD, S_SENDHEADER, S_HEADER_HIGH, S_HEADER_LOW, S_SIZE_HIGH, S_END, S_END2);

type state is (S0, SA, S1, S2, S3);
signal EA : state;

signal eop: std_logic_vector(TAM_BUFFER-1 downto 0);
signal bop: std_logic_vector(TAM_BUFFER-1 downto 0);
signal first,last: pointer;
signal available_slot, one_flit_in_buffer, tx_sig: std_logic;
signal data_read: regflit;

--signal for controlling header_routing low or high flits
signal header_high 				: std_logic;
signal flit_discard 				: std_logic;

signal verified_bop				: std_logic;
signal failed_bop				: std_logic;

-- Flit Buffer implemented using array.
signal flit_buff: buff;

signal flit_counter : std_logic_vector(2 downto 0);

	-- Observed path: SwitchControl grant is route_out; stamp CNN msg[1] before HT.
	constant PATH_ENC_FLIT : integer := 29;
	constant MAGIC_HI_FLIT : integer := 26;
	constant MAGIC_LO_FLIT : integer := 27;
	constant CNN_MAGIC_HI  : regflit := x"4350";
	constant CNN_MAGIC_LO  : regflit := x"5003";
	constant PATH_MAX_HOPS : integer := 6;

	signal tx_flit_idx : integer range 0 to 65535;
	signal tx_cnn_hi   : std_logic;
	signal tx_cnn_ok   : std_logic;

	function port_dir(p : integer) return std_logic_vector is
		variable d : std_logic_vector(1 downto 0);
	begin
		case p is
			when 0 | 1 => d := "00"; -- EAST
			when 2 | 3 => d := "01"; -- WEST
			when 4 | 5 => d := "10"; -- NORTH
			when 6 | 7 => d := "11"; -- SOUTH
			when others => d := "00";
		end case;
		return d;
	end function;

	function path_append(flit : regflit; dir : std_logic_vector(1 downto 0)) return regflit is
		variable n : integer;
		variable r : regflit;
	begin
		r := flit;
		n := conv_integer(flit(15 downto 12));
		if n < PATH_MAX_HOPS then
			r(15 downto 12) := flit(15 downto 12) + 1;
			case n is
				when 0      => r(1 downto 0)   := dir;
				when 1      => r(3 downto 2)   := dir;
				when 2      => r(5 downto 4)   := dir;
				when 3      => r(7 downto 6)   := dir;
				when 4      => r(9 downto 8)   := dir;
				when 5      => r(11 downto 10) := dir;
				when others => null;
			end case;
		end if;
		return r;
	end function;

	component RAM16X1D is			
		generic (
			INIT : bit_vector(15 downto 0) := X"0000"
		);			
		port (
			DPO   : out std_ulogic;        
			SPO   : out std_ulogic;			        
			A0    : in std_ulogic;
			A1    : in std_ulogic;
			A2    : in std_ulogic;
			A3    : in std_ulogic;
			D     : in std_ulogic;        
			DPRA0 : in std_ulogic;
			DPRA1 : in std_ulogic;
			DPRA2 : in std_ulogic;
			DPRA3 : in std_ulogic;        
			WCLK  : in std_ulogic;
			WE    : in std_ulogic
		);
	end component;
							   
begin 

	flit_discard <= '1' when (flit_counter = "10" and data_in = x"7FFF" and change_routing = '0') else
					'0';

	-- Flit Buffer implemented using LUT RAM.	
	buffer_RTL: if buffer_type = "RTL" generate	
		flit_buff: for i in 0 to TAM_FLIT-1 generate
			RAM16X1D_inst : RAM16X1D
			port map (
				DPO 	=> data_read(i), 		
				SPO 	=> open, 		
				A0 		=> last(0), 		
				A1 		=> last(1), 		
				A2 		=> last(2), 		
				A3 		=> last(3), 		
				D 		=> data_in(i), 			
				DPRA0 	=> first(0), 		
				DPRA1 	=> first(1), 		
				DPRA2 	=> first(2), 		
				DPRA3 	=> first(3), 		
				WCLK 	=> clock, 		
				WE 		=> '1' 			
			);
		end generate flit_buff;
	end generate;
	
	buffer_SIM: if buffer_type = "SIM" generate
		data_read <= flit_buff(CONV_INTEGER(first));
	end generate;
  

	-- Controls flit storing and avaliable buffer slots. 
	process(reset, clock)
	variable last_var,first_var,count: pointer;	-- Used only to update 'available_slot'.
	begin
		if reset='1' then			 	
			last 			<= (others=>'0');
			eop 			<= (others=>'0');
			bop 			<= (others=>'0');
			available_slot 	<= '1';
			first_var 		:= (others=>'0');
			last_var 		:= (others=>'0');
			count 	 		:= (others=>'0');
			flit_counter 	<= (others=>'0');
			header_fixed	<= (others=>'0');
			header_routing	<= (others=>'0');
			verified_bop	<= '0';
			failed_bop		<= '0';

		elsif rising_edge(clock) then
			-- Stores a flit.
			if rx = '1' and available_slot = '1' then

				-- If BOP verification fails the flit is not written into the buffer
				if (verified_bop='0' and flit_counter="000" and bop_in='0') or failed_bop='1' then
					if eop_in='1' then
						-- reset
						verified_bop <= '0';
						failed_bop <= '0';
					else
						verified_bop <= '1';
						failed_bop <= '1';
					end if;
				
				else
					
					verified_bop <= '1';

					if buffer_type = "SIM" then
						flit_buff(CONV_INTEGER(last)) <= data_in;
					end if;
					eop(CONV_INTEGER(last)) <= eop_in;
					bop(CONV_INTEGER(last)) <= bop_in;

					-- if not(flit_counter = "10" and data_in = x"7FFF") then
					if not(flit_discard) then
						last <= last + 1;
						last_var := last + 1;
					end if;

					if eop_in = '1' then
						flit_counter <= (others=>'0');
						verified_bop <= '0';
					elsif not(flit_discard) then
						flit_counter <= flit_counter + '1';
					end if;

					case flit_counter is
						when "000" =>--gets higher bits from header fixed
							header_fixed 	<= data_in & header_fixed(15 downto 0);
						when "001" =>--gets lower bits from header fixed
							header_fixed 	<= header_fixed(31 downto 16) & data_in;
						when "010" =>--gets higher bits from header routing
							if data_in /= x"7FFF" then
								header_routing 	<= data_in & header_routing(15 downto 0);
							end if;
							if change_routing = '1' and data_in(15 downto 12) = "0111" then
								flit_buff(CONV_INTEGER(last)) <= header_fixed(31 downto 16);
								header_routing 	<= header_fixed(31 downto 16) & header_routing(15 downto 0);
							end if;
						when "011" =>--gets lower bits from header routing
							header_routing 	<= header_routing(31 downto 16) & data_in;
							if change_routing = '1' and data_in(15 downto 12) = "0111" then
								flit_buff(CONV_INTEGER(last)) <= header_fixed(15 downto 0);
								header_routing 	<= header_routing(31 downto 16) & header_fixed(15 downto 0);
							end if;
						when others =>
					
					end case ;
				end if;
			end if;

			-- Verifies if a flit is transmited.
			if tx_sig = '1' and credit_in = '1' then
				first_var := first + 1;
			else
				first_var := first; 
			end if;
			
			-- Update the 'available_slot'.
			if (first_var = x"0" and last_var = TAM_BUFFER-1) or (first_var = last_var+1) then
				available_slot <= '0';
				count := "000";
			else
			--if ((last_var - first_var) = 2) or ((first_var - last_var) = TAM_BUFFER-2) then
				count := count + 1;
				if (count = x"5") then					
					available_slot <= '1';
				end if;	
			end if;
		end if;
	end process;
	
   
	-- Data ready to be transmited. Hop comes from SwitchControl (route_out), before HT.
	PathStampTx: process(clock, reset)
	begin
		if reset = '1' then
			tx_flit_idx <= 0;
			tx_cnn_hi <= '0';
			tx_cnn_ok <= '0';
		elsif rising_edge(clock) then
			if tx_sig = '1' and credit_in = '1' then
				if bop(CONV_INTEGER(first)) = '1' then
					tx_flit_idx <= 1;
					tx_cnn_hi <= '0';
					tx_cnn_ok <= '0';
				elsif eop(CONV_INTEGER(first)) = '1' then
					tx_flit_idx <= 0;
					tx_cnn_hi <= '0';
					tx_cnn_ok <= '0';
				else
					if tx_flit_idx = MAGIC_HI_FLIT and data_read = CNN_MAGIC_HI then
						tx_cnn_hi <= '1';
					elsif tx_flit_idx = MAGIC_LO_FLIT and tx_cnn_hi = '1' and data_read = CNN_MAGIC_LO then
						tx_cnn_ok <= '1';
					end if;
					if tx_flit_idx < 65535 then
						tx_flit_idx <= tx_flit_idx + 1;
					end if;
				end if;
			end if;
		end if;
	end process;

	PathStampMux: process(data_read, tx_sig, tx_flit_idx, tx_cnn_ok, route_out, first, bop)
		variable this_flit : integer;
	begin
		if bop(CONV_INTEGER(first)) = '1' and tx_sig = '1' then
			this_flit := 0;
		else
			this_flit := tx_flit_idx;
		end if;
		if tx_cnn_ok = '1' and route_out < LOCAL0 and this_flit = PATH_ENC_FLIT then
			data_out <= path_append(data_read, port_dir(route_out));
		else
			data_out <= data_read;
		end if;
	end process;
	
	-- Indicates if the current flit is the last packet flit ('1').
	eop_out <= eop(CONV_INTEGER(first));

	-- Indicates if the current flit is the first packet flit ('1').
	bop_out <= bop(CONV_INTEGER(first));
	
	--Indicates if exist at least one free slot in buffer ('1').
	credit_out <= available_slot;	
	
	-- Indicates one flit stored in the buffer.
	one_flit_in_buffer <= '1' when first+1 = last else '0';
		
	tx <= tx_sig;
	-- Routing request and packet flits transmition.
	process(reset, clock)
	begin
		if reset='1' then
			sender 			<='0';
			req_routing 	<= '0';
			tx_sig 			<= '0';
			first 			<= (others=>'0');
			EA 				<= S0;
			
		elsif rising_edge(clock) then
			case EA is
				-- waits one flit to be stored
				when S0 =>
					-- Waits for a stored flit.
					if first /= last then
						EA 				<= SA;
					else
						EA<= S0;
					end if;

				-- waits second flit
				when SA =>
					-- Waits for a stored flit.
					if rx = '1' then
							req_routing 	<= '1';
							EA 				<= S1;
					else
						EA<= SA;
					end if;
				
				-- Waits routing request acknolegment from Switch Contol.	
				when S1 => 	
					if ack_routing = '1' then
						req_routing <= '0';
						tx_sig 		<= '1';
						sender 		<= '1';
						EA 			<= S2 ;
					else
						EA 			<= S1;
					end if;
				 
				-- Send packet flits.
				when S2 =>
					if credit_in = '1' then
						first <= first + 1;
						
						-- Verifies if the current flit is the last packet flit.
						if eop(CONV_INTEGER(first)) = '1' then
							tx_sig <= '0';
							sender <= '0';

							if one_flit_in_buffer = '1' then
								EA <= S0;
							else
								EA 				<= SA;
							end if;

						
						-- Verifies if the current flit is the only stored flit.
						elsif one_flit_in_buffer = '1' then
							tx_sig <= '0';
							EA <= S3;
						else
							EA <= S2;
						end if;					
					
					else
						EA <= S2;
					end if;
					
				-- Waits for a stored flit.
				when S3 =>

					if first /= last then
						tx_sig <= '1';
						EA <= S2;	
					else
						EA <= S3;
					end if;
						
					
				when others =>
					EA <= S0;
			
			end case;
		end if;
	end process; 
end Hermes_buffer;
