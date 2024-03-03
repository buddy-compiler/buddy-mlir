abs_top_srcdir = /home/qzylalala/work_space/chipyard/generators/gemmini/software/gemmini-rocc-tests

GEMMINI_HEADERS = $(abs_top_srcdir)/include $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_nn.h $(abs_top_srcdir)/include/gemmini_testutils.h

CC_LINUX := riscv64-unknown-linux-gnu-gcc
AR_LINUX := riscv64-unknown-linux-gnu-ar

CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-fno-tree-loop-distribute-patterns \
	-march=rv64gc -Wa,-march=rv64gc \
	-lm \
	-lgcc \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(abs_top_srcdir)/include \
	-DID_STRING=$(ID_STRING) \
	-Wno-incompatible-pointer-types

CFLAGS_PK := \
	$(CFLAGS) \
	-static \
	-DBAREMETAL=1 \

gemmini.o: gemmini.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS_PK) $< $(LFLAGS) -c

libgemmini.a: gemmini.o
	$(AR_LINUX) -r $@ $^
