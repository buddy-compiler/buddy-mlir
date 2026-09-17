// FP32 attention, no integer quantization of probabilities or Q/K/V.
module {
  func.func @kernel_attention_pv_16x1x128x17(%a: memref<16x1x17xf32>, %b: memref<16x17x128xf32>, %c: memref<16x1x128xf32>) attributes {llvm.emit_c_interface} {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<16x1x128xf32>)
    linalg.batch_matmul ins(%a, %b : memref<16x1x17xf32>, memref<16x17x128xf32>) outs(%c : memref<16x1x128xf32>)
    return
  }
}
