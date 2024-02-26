#include "gemmini.h"
#include <stdio.h>
#include <stdlib.h>

typedef uint64_t reg_t;

static void _mlir_ciface_mvin(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_MVIN);
}

static void _mlir_ciface_mvin2(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_MVIN2);
}

static void _mlir_ciface_mvin3(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_MVIN3);
}

static void _mlir_ciface_mvout(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_MVOUT);
}

static void _mlir_ciface_flush(reg_t rs1, reg_t rs2 = 0) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_FLUSH);
}

static void _mlir_ciface_config_ld(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_CONFIG);
}

static void _mlir_ciface_config_st(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_CONFIG);
}

static void _mlir_ciface_config_ex(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_CONFIG);
}

static void _mlir_ciface_config_norm(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_CONFIG);
}

static void _mlir_ciface_preload(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_PRELOAD);
}

static void _mlir_ciface_compute_preloaded(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_COMPUTE_PRELOADED);
}

static void _mlir_ciface_compute_accumulated(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_COMPUTE_ACCUMULATE);
}

static void _mlir_ciface_loop_ws_config_bounds(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_WS_CONFIG_BOUNDS);
}

static void _mlir_ciface_loop_ws_config_addrs_ab(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_WS_CONFIG_ADDRS_AB);
}

static void _mlir_ciface_loop_ws_config_addrs_dc(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_WS_CONFIG_ADDRS_DC);
}

static void _mlir_ciface_loop_ws_config_strides_ab(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_WS_CONFIG_STRIDES_AB);
}

static void _mlir_ciface_loop_ws_config_strides_dc(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_WS_CONFIG_STRIDES_DC);
}

static void _mlir_ciface_loop_ws(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_WS);
}

static void _mlir_ciface_loop_conv_ws(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_CONV_WS);
}

static void _mlir_ciface_loop_conv_ws_config1(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_CONV_WS_CONFIG_1);
}

static void _mlir_ciface_loop_conv_ws_config2(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_CONV_WS_CONFIG_2);
}

static void _mlir_ciface_loop_conv_ws_config3(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_CONV_WS_CONFIG_3);
}

static void _mlir_ciface_loop_conv_ws_config4(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_CONV_WS_CONFIG_4);
}

static void _mlir_ciface_loop_conv_ws_config5(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_CONV_WS_CONFIG_5);
}

static void _mlir_ciface_loop_conv_ws_config6(reg_t rs1, reg_t rs2) {
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_LOOP_CONV_WS_CONFIG_6);
}