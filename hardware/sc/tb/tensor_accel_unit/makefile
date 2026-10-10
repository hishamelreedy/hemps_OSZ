# Standalone SystemC unit TB for tensor_accel (no NoC).
# Must run under WSL/Linux (binary is ELF), not Windows CMD:
#   wsl
#   cd ".../hardware/sc/tb/tensor_accel_unit"
#   make run
#
# SystemC include + lib MUST be the same install. Mismatch causes:
#   undefined reference to sc_api_version_2_3_3_cxx201703L
# Prefer local 2.3.3 if present, else distro 2.3.4.
#
# Override explicitly if needed:
#   make SYSTEMC_HOME=/usr/local/systemc-2.3.3 run

COR  ='\033[0;32m'
NC   ='\033[0m'

SC_ROOT   := ../..
PE_ACCEL  := $(SC_ROOT)/pe/tensor_accel
INC_PKG   := ../../../../include
SW_INC    := ../../../../software/include

# Prefer /usr/local/systemc-2.3.3 when both header and lib exist.
ifeq ($(origin SYSTEMC_HOME),undefined)
  ifneq ($(wildcard /usr/local/systemc-2.3.3/lib-linux64/libsystemc.so),)
    SYSTEMC_HOME := /usr/local/systemc-2.3.3
  else ifneq ($(wildcard /usr/local/systemc-2.3.3/lib/libsystemc.so),)
    SYSTEMC_HOME := /usr/local/systemc-2.3.3
  endif
endif

ifdef SYSTEMC_HOME
  SYSTEMC_INC ?= $(SYSTEMC_HOME)/include
  ifneq ($(wildcard $(SYSTEMC_HOME)/lib-linux64/libsystemc.so),)
    SYSTEMC_LIB ?= $(SYSTEMC_HOME)/lib-linux64
  else
    SYSTEMC_LIB ?= $(SYSTEMC_HOME)/lib
  endif
else
  SYSTEMC_INC ?= /usr/include
  SYSTEMC_LIB ?= /usr/lib/x86_64-linux-gnu
endif

CXX      ?= g++
# Match typical SystemC builds (g++ 13 defaults to C++17; keep explicit).
CXXSTD   ?= -std=c++17
CXXFLAGS := -g -Wall -O2 -fpermissive -Wno-write-strings $(CXXSTD) \
	-DTENSOR_ACCEL_UNIT_TB \
	-I. -I$(SC_ROOT) -I$(INC_PKG) -I$(SW_INC) -I$(SYSTEMC_INC)
LDFLAGS  := -L$(SYSTEMC_LIB) -Wl,-rpath,$(SYSTEMC_LIB) -lsystemc -lm

TARGET := tensor_accel_unit_tb
LOG    := $(TARGET).log
BATCH_H := ../../../../software/include/lenet5_tb_batch.h
BATCH_N ?= 100

.PHONY: all run clean gen-batch print-sc

all: $(TARGET)

print-sc:
	@echo "SYSTEMC_HOME=$(SYSTEMC_HOME)"
	@echo "SYSTEMC_INC=$(SYSTEMC_INC)"
	@echo "SYSTEMC_LIB=$(SYSTEMC_LIB)"

gen-batch:
	python3 ../../../../tools/gen_lenet5_tb_batch.py --batch $(BATCH_N) -o $(BATCH_H)

$(BATCH_H):
	$(MAKE) gen-batch

$(TARGET): sc_main.o tensor_accel.o
	@printf "${COR}Linking %s (lib=$(SYSTEMC_LIB)) ...${NC}\n" "$@"
	$(CXX) -o $@ $^ $(LDFLAGS)

sc_main.o: sc_main.cpp tb_ram.h $(PE_ACCEL)/tensor_accel.h $(BATCH_H)
	@printf "${COR}Compiling sc_main.cpp (inc=$(SYSTEMC_INC)) ...${NC}\n"
	$(CXX) $(CXXFLAGS) -c sc_main.cpp -o $@

tensor_accel.o: $(PE_ACCEL)/tensor_accel.cpp $(PE_ACCEL)/tensor_accel.h tb_ram.h
	@printf "${COR}Compiling tensor_accel.cpp (unit TB) ...${NC}\n"
	$(CXX) $(CXXFLAGS) -c $(PE_ACCEL)/tensor_accel.cpp -o $@

run: $(BATCH_H) $(TARGET)
	@uname -s >/dev/null 2>&1 || { echo "ERROR: run this under WSL/Linux, not Windows CMD"; exit 1; }
	@echo "Running $(TARGET) (log -> $(LOG)) ..."
	./$(TARGET) 2>&1 | tee $(LOG)

clean:
	rm -f *.o $(TARGET) $(LOG)
