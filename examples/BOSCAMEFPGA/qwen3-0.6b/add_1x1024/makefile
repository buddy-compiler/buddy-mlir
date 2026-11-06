# add_1x1024: f32 residual add, memref<1x1024xf32>.
#
# The directory name is the operator identity. ../common.mk sets
# NAME=$(notdir $(CURDIR)), stamps target=nr-fpga, and lowers this kernel
# with convert-linalg-to-loops (scalar RA, no AME, no RVV). Do not add
# per-operator rules here.
include ../common.mk
