GREEN  ='\033[0;32m'
NC   ='\033[0m' # No Color

INCLUDE           = ../../software/include

CFLAGS            = -O2 -Wall -fms-extensions -c -s -std=c99 -G 0
# ACCEL_MODE: software (default) | sw | stub_ni | stub_macs | hw
ifeq ($(ACCEL_MODE),hw)
ACCEL_CFLAGS      = -DCNN_ACCEL_MODE=ACCEL_MODE_STUB_MACS -DCNN_ACCEL_BACKEND=1 -DMLP_USE_ACCEL=1
else ifeq ($(ACCEL_MODE),stub_ni)
ACCEL_CFLAGS      = -DCNN_ACCEL_MODE=ACCEL_MODE_STUB_NI -DCNN_ACCEL_BACKEND=0 -DMLP_USE_ACCEL=1
else ifeq ($(ACCEL_MODE),stub_macs)
ACCEL_CFLAGS      = -DCNN_ACCEL_MODE=ACCEL_MODE_STUB_MACS -DCNN_ACCEL_BACKEND=0 -DMLP_USE_ACCEL=1
else ifeq ($(ACCEL_MODE),sw)
ACCEL_CFLAGS      = -DCNN_ACCEL_MODE=ACCEL_MODE_STUB_MACS -DCNN_ACCEL_BACKEND=0 -DMLP_USE_ACCEL=1
else
ACCEL_CFLAGS      = -DCNN_ACCEL_MODE=ACCEL_MODE_SOFTWARE -DCNN_ACCEL_BACKEND=0 -DMLP_USE_ACCEL=0
endif
# Regenerate real weights/goldens:
#   python tools/export_mlp784_assets.py
#   python tools/run_mlp784_campaign.py
ifdef MLP784_ACCEL_UNIT
CFLAGS            += -DMLP784_ACCEL_UNIT=1
endif
CFLAGS            += $(ACCEL_CFLAGS)
GCC_MIPS          = mips-elf-gcc $(CFLAGS)
AS_MIPS           = mips-elf-as
LD_MIPS           = mips-elf-ld
DUMP_MIPS         = mips-elf-objdump
COPY_MIPS         = mips-elf-objcopy -I elf32-bigmips -O binary

BOOT_TASK         = boot_task
BOOT_TASK_SRC     = ../../software/boot_task/boot_task.asm

TASK_SRC = $(wildcard *.c)
TASK_OBJ = $(TASK_SRC:.c=.o)
NILES_HDRS = $(INCLUDE)/mlp784_niles_config.h $(INCLUDE)/niles_app.h $(INCLUDE)/mlp784_input.h

default: $(BOOT_TASK).o $(TASK_OBJ) 

boot_task.o: $(BOOT_TASK_SRC)
	@printf "${GREEN}Compiling boot task: %s ...${NC}\n" "$*.c"
	$(AS_MIPS) --defsym sp_addr=$(PAGE_SP_INIT) -o $@ $^

$(TASK_OBJ): $(TASK_SRC) $(NILES_HDRS)
	@printf "${GREEN}Compiling task: %s ...${NC}\n" "$*.c"
	$(GCC_MIPS) $(CFLAGS) $*.c -o $*.o --include id_tasks.h -I $(INCLUDE)
	$(LD_MIPS) -Ttext 0 -eentry -Map $*.map -s -N -o $*.bin $(BOOT_TASK).o $*.o
	$(LD_MIPS) -Ttext 0 -eentry -Map $*_debug.map -o $*_debug.bin $(BOOT_TASK).o $*.o
	$(DUMP_MIPS) -S $*_debug.bin > $*.lst
	$(COPY_MIPS) $*.bin $*.dump
	hexdump -v -e '1/1 "%02x" 1/1 "%02x" 1/1 "%02x" 1/1 "%02x" "\n"' $*.dump > $*.txt

clean:
	@printf "Cleaning up\n"
	@rm -rf *.o
	@rm -rf *.bin
	@rm -rf *.map
	@rm -rf *.lst
	@rm -rf *.txt
	@rm -rf *.dump
