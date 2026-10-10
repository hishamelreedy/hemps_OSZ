vsim -t ps +notimingchecks -voptargs=+acc hardware/work.test_bench

# wave.do touches unbound IO ports in batch mode; use GUI vsim without -c to open waves
# do wave.do
onerror {resume}
onbreak {quit}
radix hex
set NumericStdNoWarnings 1
set StdArithNoWarnings 1

