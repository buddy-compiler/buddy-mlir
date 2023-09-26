""" we need env: buddy tools, you can get them in buddy-mlir"""
import os
import subprocess

filepath = " ./test/matmul-os.mlir"
passes = ["-lower-gemmini"]

# build cmd
activate_cmd = "source ~/.zshrc && conda activate gemmini && "
opt_cmd = 'buddy-opt ' + filepath + ' {} | '.format(' '.join(passes))
translate_cmd = f'buddy-translate --buddy-to-llvmir | '
llc_cmd = f'buddy-llc -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D -float-abi=hard -o log.o && '
riscv_cmd = f'riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out'

# build and run
spike_cmd = f'spike --extension=gemmini pk a.out'
build_cmd = activate_cmd + opt_cmd + translate_cmd + llc_cmd + riscv_cmd

proc = subprocess.Popen(build_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, executable=os.environ["SHELL"])
(out, err) = proc.communicate()

if proc.returncode != 0:
	msg = "buddy build error:\n"
	py_str = lambda x: x.decode("utf-8")
	msg += py_str(out)
	raise RuntimeError(msg)
else:
    print(out.decode("utf-8"))

# build_cmd = opt_cmd + translate_cmd + llc_cmd
# os.system(build_cmd)
# os.system(riscv_cmd)
# os.system(spike_cmd)