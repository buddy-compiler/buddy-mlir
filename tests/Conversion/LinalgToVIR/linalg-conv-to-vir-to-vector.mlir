// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s --check-prefix=VIR
// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse --convert-vector-to-scf --lower-affine --expand-strided-metadata --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

// The 2-D convolutions over NHWC inputs: vectorized along the output
// channels, the rows of the output in parallel, an FHWC filter copied into
// HWCF first.

// VIR-LABEL: func.func @conv_fhwc
// VIR: memref.alloc() : memref<3x3x3x5xf32>
// VIR: scf.parallel
// VIR: memref.store
// VIR: scf.parallel
// VIR: vir.set_vl %{{.*}} : index {
// VIR: vir.load %{{.*}} : memref<1x3x5x5xf32> -> !vir.vec<?xf32>
// VIR: vir.load %{{.*}} : memref<3x3x3x5xf32> -> !vir.vec<?xf32>
// VIR: memref.load %{{.*}} : memref<1x5x7x3xf32>
// VIR: vir.broadcast
// VIR: vir.fma
// VIR: vir.store
// VIR: memref.dealloc
// VIR-NOT: linalg.conv
// VIR-LABEL: func.func @conv_hwcf_s2
// VIR-NOT: memref.alloc() : memref<
// VIR: vir.set_vl

memref.global "private" constant @in : memref<1x5x7x3xf32> = dense<[[[[-2.0, -1.0, 0.0], [1.0, 2.0, -2.0], [-1.0, 0.0, 1.0], [2.0, -2.0, -1.0], [0.0, 1.0, 2.0], [-2.0, -1.0, 0.0], [1.0, 2.0, -2.0]], [[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0], [1.0, 2.0, -2.0], [-1.0, 0.0, 1.0], [2.0, -2.0, -1.0], [0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]], [[2.0, -2.0, -1.0], [0.0, 1.0, 2.0], [-2.0, -1.0, 0.0], [1.0, 2.0, -2.0], [-1.0, 0.0, 1.0], [2.0, -2.0, -1.0], [0.0, 1.0, 2.0]], [[-1.0, 0.0, 1.0], [2.0, -2.0, -1.0], [0.0, 1.0, 2.0], [-2.0, -1.0, 0.0], [1.0, 2.0, -2.0], [-1.0, 0.0, 1.0], [2.0, -2.0, -1.0]], [[1.0, 2.0, -2.0], [-1.0, 0.0, 1.0], [2.0, -2.0, -1.0], [0.0, 1.0, 2.0], [-2.0, -1.0, 0.0], [1.0, 2.0, -2.0], [-1.0, 0.0, 1.0]]]]>
memref.global "private" constant @w1 : memref<5x3x3x3xf32> = dense<[[[[-3.0, 2.0, 0.0], [-2.0, 3.0, 1.0], [-1.0, -3.0, 2.0]], [[-1.0, -3.0, 2.0], [0.0, -2.0, 3.0], [1.0, -1.0, -3.0]], [[1.0, -1.0, -3.0], [2.0, 0.0, -2.0], [3.0, 1.0, -1.0]]], [[[0.0, -2.0, 3.0], [1.0, -1.0, -3.0], [2.0, 0.0, -2.0]], [[2.0, 0.0, -2.0], [3.0, 1.0, -1.0], [-3.0, 2.0, 0.0]], [[-3.0, 2.0, 0.0], [-2.0, 3.0, 1.0], [-1.0, -3.0, 2.0]]], [[[3.0, 1.0, -1.0], [-3.0, 2.0, 0.0], [-2.0, 3.0, 1.0]], [[-2.0, 3.0, 1.0], [-1.0, -3.0, 2.0], [0.0, -2.0, 3.0]], [[0.0, -2.0, 3.0], [1.0, -1.0, -3.0], [2.0, 0.0, -2.0]]], [[[-1.0, -3.0, 2.0], [0.0, -2.0, 3.0], [1.0, -1.0, -3.0]], [[1.0, -1.0, -3.0], [2.0, 0.0, -2.0], [3.0, 1.0, -1.0]], [[3.0, 1.0, -1.0], [-3.0, 2.0, 0.0], [-2.0, 3.0, 1.0]]], [[[2.0, 0.0, -2.0], [3.0, 1.0, -1.0], [-3.0, 2.0, 0.0]], [[-3.0, 2.0, 0.0], [-2.0, 3.0, 1.0], [-1.0, -3.0, 2.0]], [[-1.0, -3.0, 2.0], [0.0, -2.0, 3.0], [1.0, -1.0, -3.0]]]]>
memref.global "private" constant @w2 : memref<3x3x3x6xf32> = dense<[[[[-3.0, 2.0, 0.0, -2.0, 3.0, 1.0], [-2.0, 3.0, 1.0, -1.0, -3.0, 2.0], [-1.0, -3.0, 2.0, 0.0, -2.0, 3.0]], [[-1.0, -3.0, 2.0, 0.0, -2.0, 3.0], [0.0, -2.0, 3.0, 1.0, -1.0, -3.0], [1.0, -1.0, -3.0, 2.0, 0.0, -2.0]], [[1.0, -1.0, -3.0, 2.0, 0.0, -2.0], [2.0, 0.0, -2.0, 3.0, 1.0, -1.0], [3.0, 1.0, -1.0, -3.0, 2.0, 0.0]]], [[[0.0, -2.0, 3.0, 1.0, -1.0, -3.0], [1.0, -1.0, -3.0, 2.0, 0.0, -2.0], [2.0, 0.0, -2.0, 3.0, 1.0, -1.0]], [[2.0, 0.0, -2.0, 3.0, 1.0, -1.0], [3.0, 1.0, -1.0, -3.0, 2.0, 0.0], [-3.0, 2.0, 0.0, -2.0, 3.0, 1.0]], [[-3.0, 2.0, 0.0, -2.0, 3.0, 1.0], [-2.0, 3.0, 1.0, -1.0, -3.0, 2.0], [-1.0, -3.0, 2.0, 0.0, -2.0, 3.0]]], [[[3.0, 1.0, -1.0, -3.0, 2.0, 0.0], [-3.0, 2.0, 0.0, -2.0, 3.0, 1.0], [-2.0, 3.0, 1.0, -1.0, -3.0, 2.0]], [[-2.0, 3.0, 1.0, -1.0, -3.0, 2.0], [-1.0, -3.0, 2.0, 0.0, -2.0, 3.0], [0.0, -2.0, 3.0, 1.0, -1.0, -3.0]], [[0.0, -2.0, 3.0, 1.0, -1.0, -3.0], [1.0, -1.0, -3.0, 2.0, 0.0, -2.0], [2.0, 0.0, -2.0, 3.0, 1.0, -1.0]]]]>
memref.global "private" constant @w3 : memref<5x2x2x3xf32> = dense<[[[[-3.0, 2.0, 0.0], [-2.0, 3.0, 1.0]], [[-1.0, -3.0, 2.0], [0.0, -2.0, 3.0]]], [[[0.0, -2.0, 3.0], [1.0, -1.0, -3.0]], [[2.0, 0.0, -2.0], [3.0, 1.0, -1.0]]], [[[3.0, 1.0, -1.0], [-3.0, 2.0, 0.0]], [[-2.0, 3.0, 1.0], [-1.0, -3.0, 2.0]]], [[[-1.0, -3.0, 2.0], [0.0, -2.0, 3.0]], [[1.0, -1.0, -3.0], [2.0, 0.0, -2.0]]], [[[2.0, 0.0, -2.0], [3.0, 1.0, -1.0]], [[-3.0, 2.0, 0.0], [-2.0, 3.0, 1.0]]]]>

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

// 3 x 3, stride 1: 5 output pixels a row, 5 output
// channels (a vector of 4, then 1), accumulated into the output
// (out[..., f] = f before).
func.func @conv_fhwc(%in: memref<1x5x7x3xf32>, %w: memref<5x3x3x3xf32>,
                     %out: memref<1x3x5x5xf32>) {
  linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                             strides = dense<1> : tensor<2xi64>}
    ins(%in, %w : memref<1x5x7x3xf32>, memref<5x3x3x3xf32>)
    outs(%out : memref<1x3x5x5xf32>)
  return
}

// 3 x 3, stride 2, an HWCF filter of 6 output channels.
func.func @conv_hwcf_s2(%in: memref<1x5x7x3xf32>, %w: memref<3x3x3x6xf32>,
                        %out: memref<1x2x3x6xf32>) {
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                             strides = dense<2> : tensor<2xi64>}
    ins(%in, %w : memref<1x5x7x3xf32>, memref<3x3x3x6xf32>)
    outs(%out : memref<1x2x3x6xf32>)
  return
}

// 2 x 2, dilation 2.
func.func @conv_fhwc_d2(%in: memref<1x5x7x3xf32>, %w: memref<5x2x2x3xf32>,
                        %out: memref<1x3x5x5xf32>) {
  linalg.conv_2d_nhwc_fhwc {dilations = dense<2> : tensor<2xi64>,
                             strides = dense<1> : tensor<2xi64>}
    ins(%in, %w : memref<1x5x7x3xf32>, memref<5x2x2x3xf32>)
    outs(%out : memref<1x3x5x5xf32>)
  return
}

func.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %in = memref.get_global @in : memref<1x5x7x3xf32>
  %w1 = memref.get_global @w1 : memref<5x3x3x3xf32>
  %w2 = memref.get_global @w2 : memref<3x3x3x6xf32>
  %w3 = memref.get_global @w3 : memref<5x2x2x3xf32>
  %zero = arith.constant 0.0 : f32
  %minus1 = arith.constant -1.0 : f32

  %o1 = memref.alloc() : memref<1x3x5x5xf32>
  scf.for %h = %c0 to %c3 step %c1 {
    scf.for %w = %c0 to %c5 step %c1 {
      scf.for %f = %c0 to %c5 step %c1 {
        %fi = arith.index_cast %f : index to i32
        %ff = arith.sitofp %fi : i32 to f32
        memref.store %ff, %o1[%c0, %h, %w, %f] : memref<1x3x5x5xf32>
      }
    }
  }
  call @conv_fhwc(%in, %w1, %o1)
    : (memref<1x5x7x3xf32>, memref<5x3x3x3xf32>, memref<1x3x5x5xf32>) -> ()
  %p1 = memref.cast %o1 : memref<1x3x5x5xf32> to memref<*xf32>
  call @printMemrefF32(%p1) : (memref<*xf32>) -> ()

  %o2 = memref.alloc() : memref<1x2x3x6xf32>
  linalg.fill ins(%minus1 : f32) outs(%o2 : memref<1x2x3x6xf32>)
  call @conv_hwcf_s2(%in, %w2, %o2)
    : (memref<1x5x7x3xf32>, memref<3x3x3x6xf32>, memref<1x2x3x6xf32>) -> ()
  %p2 = memref.cast %o2 : memref<1x2x3x6xf32> to memref<*xf32>
  call @printMemrefF32(%p2) : (memref<*xf32>) -> ()

  %o3 = memref.alloc() : memref<1x3x5x5xf32>
  linalg.fill ins(%zero : f32) outs(%o3 : memref<1x3x5x5xf32>)
  call @conv_fhwc_d2(%in, %w3, %o3)
    : (memref<1x5x7x3xf32>, memref<5x2x2x3xf32>, memref<1x3x5x5xf32>) -> ()
  %p3 = memref.cast %o3 : memref<1x3x5x5xf32> to memref<*xf32>
  call @printMemrefF32(%p3) : (memref<*xf32>) -> ()

  memref.dealloc %o1 : memref<1x3x5x5xf32>
  memref.dealloc %o2 : memref<1x2x3x6xf32>
  memref.dealloc %o3 : memref<1x3x5x5xf32>

  // The expected outputs are those of numpy (the same sums in another order:
  // small integers, exact in f32).
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 4 offset = 0 sizes = \[1, 3, 5, 5\]}}
  // CHECK-NEXT{LITERAL}: [[[[13, -6, -11, -2, 7],
  // CHECK-NEXT{LITERAL}: [-7, -9, 3, 8, 13],
  // CHECK-NEXT{LITERAL}: [-7, 8, 2, 3, 4],
  // CHECK-NEXT{LITERAL}: [3, 0, 11, 8, 5],
  // CHECK-NEXT{LITERAL}: [-2, 12, 5, -2, -9]],
  // CHECK-NEXT{LITERAL}: [[-2, 12, 5, -2, -9],
  // CHECK-NEXT{LITERAL}: [13, -6, -11, -2, 7],
  // CHECK-NEXT{LITERAL}: [-7, -9, 3, 8, 13],
  // CHECK-NEXT{LITERAL}: [-7, 8, 2, 3, 4],
  // CHECK-NEXT{LITERAL}: [3, 0, 11, 8, 5]],
  // CHECK-NEXT{LITERAL}: [[3, 0, 11, 8, 5],
  // CHECK-NEXT{LITERAL}: [-2, 12, 5, -2, -9],
  // CHECK-NEXT{LITERAL}: [13, -6, -11, -2, 7],
  // CHECK-NEXT{LITERAL}: [-7, -9, 3, 8, 13],
  // CHECK-NEXT{LITERAL}: [-7, 8, 2, 3, 4]]]]
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 4 offset = 0 sizes = \[1, 2, 3, 6\]}}
  // CHECK-NEXT{LITERAL}: [[[[11, -6, 12, -5, -8, -4],
  // CHECK-NEXT{LITERAL}: [-7, -7, -7, -7, 0, 21],
  // CHECK-NEXT{LITERAL}: [-5, 12, -6, 11, -7, -4]],
  // CHECK-NEXT{LITERAL}: [[-6, -15, 11, 2, 14, -9],
  // CHECK-NEXT{LITERAL}: [11, -6, 12, -5, -8, -4],
  // CHECK-NEXT{LITERAL}: [-7, -7, -7, -7, 0, 21]]]]
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 4 offset = 0 sizes = \[1, 3, 5, 5\]}}
  // CHECK-NEXT{LITERAL}: [[[[11, -3, -10, 11, -17],
  // CHECK-NEXT{LITERAL}: [-19, 0, -9, -11, 15],
  // CHECK-NEXT{LITERAL}: [16, -12, 2, 2, 2],
  // CHECK-NEXT{LITERAL}: [-19, 11, 13, 15, -11],
  // CHECK-NEXT{LITERAL}: [11, 4, 4, -17, 11]],
  // CHECK-NEXT{LITERAL}: [[11, 4, 4, -17, 11],
  // CHECK-NEXT{LITERAL}: [11, -3, -10, 11, -17],
  // CHECK-NEXT{LITERAL}: [-19, 0, -9, -11, 15],
  // CHECK-NEXT{LITERAL}: [16, -12, 2, 2, 2],
  // CHECK-NEXT{LITERAL}: [-19, 11, 13, 15, -11]],
  // CHECK-NEXT{LITERAL}: [[-19, 11, 13, 15, -11],
  // CHECK-NEXT{LITERAL}: [11, 4, 4, -17, 11],
  // CHECK-NEXT{LITERAL}: [11, -3, -10, 11, -17],
  // CHECK-NEXT{LITERAL}: [-19, 0, -9, -11, 15],
  // CHECK-NEXT{LITERAL}: [16, -12, 2, 2, 2]]]]
  %ret = arith.constant 0 : i32
  return %ret : i32
}
