// RUN: not buddy-opt %s --lower-linalg-to-boscame=target=upstream --lower-qwen-w8a8-to-boscame=target=qwen3-fpga 2>&1 | FileCheck %s
// RUN: not buddy-opt %s --lower-linalg-to-boscame --lower-bosc-ame --lower-qwen-w8a8-to-boscame=target=qwen3-fpga 2>&1 | FileCheck %s
// CHECK: upstream AME configuration cannot be mixed with the qwen3-fpga target
module {
func.func @mixed(%A: memref<4x32xi8>, %B: memref<32x4xi8>, %C: memref<4x4xi32>) {
 linalg.matmul ins(%A, %B : memref<4x32xi8>, memref<32x4xi8>) outs(%C : memref<4x4xi32>)
 return
}}
