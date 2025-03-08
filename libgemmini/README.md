## Build libgemmini

Make sure that you have chipyard installed and before you compile this library, please activate the specify conda environment.

```shell
conda activate $(env_chipyard)
```

Please specify your own path of chipyard. You can type this command to compile:

```shell
$ make abs_top_srcdir=/your/path/chipyard/generators/gemmini/software/gemmini-rocc-tests
```

## Usage

We follow this compilation flow `Linalg -> Gemmini -> Func -> LLVM IR`. And we should use new passes and compilation commands, here is an example how we modify the Makefile in `examples/GemminiDialect`:

```makefile
mvin-mvout-run:
	@${BUDDY_OPT} ./mvin-mvout.mlir \
	-lower-gemmini-to-func \
	-lower-gemmini-to-llvm | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -L /path/to/buddy-mlir/libgemmini -l gemmini -O2 -o a.out
	@spike --extension=gemmini pk a.out
```
