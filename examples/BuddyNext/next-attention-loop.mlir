// RUN: buddy-opt %s \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @rtclock() -> f64
  memref.global "private" constant @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32> = dense<8.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x1x40x40xf32 : memref<1x1x40x40xf32> = dense<4.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_32x128x40xf32 : memref<32x128x40xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_32x40x128xf32 : memref<32x40x128xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x32x40x40xf32 : memref<1x32x40x40xf32> = dense<11.3137083> {alignment = 64 : i64}
  func.func @kenerl(%arg0: tensor<32x40x128xf32>, %arg1: tensor<32x128x40xf32>, %arg2: tensor<1x1x40x40xf32>, %arg3: tensor<1x32x40x128xf32>) {
    %t_start = call @rtclock() : () -> f64
    %cst = arith.constant 0.0883883461 : f32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant -3.40282347E+38 : f32
    %0 = bufferization.to_memref %arg3 : tensor<1x32x40x128xf32> to memref<1x32x40x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg2 : tensor<1x1x40x40xf32> to memref<1x1x40x40xf32, strided<[?, ?, ?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg1 : tensor<32x128x40xf32> to memref<32x128x40xf32, strided<[?, ?, ?], offset: ?>>
    %3 = bufferization.to_memref %arg0 : tensor<32x40x128xf32> to memref<32x40x128xf32, strided<[?, ?, ?], offset: ?>>

    // MatMul
    // %0 = tosa.matmul %t0, %t1 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    // Initialize MatMul Output.
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x40x40xf32>
    affine.for %arg4 = 0 to 32 {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 40 {
          affine.store %cst_0, %alloc[%arg4, %arg5, %arg6] : memref<32x40x40xf32>
        }
      }
    }
    // Perform MatMul core operations: multiplication and addition.
    affine.for %arg4 = 0 to 32 {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 128 {
            %5 = affine.load %3[%arg4, %arg5, %arg7] : memref<32x40x128xf32, strided<[?, ?, ?], offset: ?>>
            %6 = affine.load %2[%arg4, %arg7, %arg6] : memref<32x128x40xf32, strided<[?, ?, ?], offset: ?>>
            %7 = affine.load %alloc[%arg4, %arg5, %arg6] : memref<32x40x40xf32>
            %8 = arith.mulf %5, %6 : f32
            %9 = arith.addf %7, %8 : f32
            affine.store %9, %alloc[%arg4, %arg5, %arg6] : memref<32x40x40xf32>
          }
        }
      }
    }

    // Reshape + Constant + Reciprocal
    // %1 = tosa.reshape %0 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    // %2 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    // %3 = tosa.reciprocal %2 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2], [3]] output_shape [1, 32, 40, 40]: memref<32x40x40xf32> into memref<1x32x40x40xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            affine.store %cst, %alloc_3[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }

    // Multiplication
    // %4 = tosa.mul %1, %3 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %expand_shape[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = affine.load %alloc_3[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %7 = arith.mulf %5, %6 : f32
            affine.store %7, %alloc_4[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }

    // Addition
    // %5 = tosa.add %4, %t2 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %alloc_4[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = affine.load %1[%c0, %c0, %arg6, %arg7] : memref<1x1x40x40xf32, strided<[?, ?, ?, ?], offset: ?>>
            %7 = arith.addf %5, %6 : f32
            affine.store %7, %alloc_5[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }

    // Reduce Max
    // %6 = tosa.reduce_max %5 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    // Initialize reduce max operation output.
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.store %cst_2, %alloc_6[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
        }
      }
    }
    // Perform reduce max operation.
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %alloc_5[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = affine.load %alloc_6[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
            %7 = arith.cmpf ugt, %5, %6 : f32
            %8 = arith.select %7, %5, %6 : f32
            %9 = arith.cmpf uno, %6, %6 : f32
            %10 = arith.select %9, %6, %8 : f32
            affine.store %10, %alloc_6[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
          }
        }
      }
    }

    // Subtraction
    // %7 = tosa.sub %5, %6 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    // Allocate space and perform subtraction.
    %expand_shape_7 = memref.expand_shape %alloc_6 [[0], [1], [2, 3]] output_shape [1, 32, 40, 1]: memref<1x32x40xf32> into memref<1x32x40x1xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %alloc_5[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = affine.load %expand_shape_7[%c0, %arg5, %arg6, %c0] : memref<1x32x40x1xf32>
            %7 = arith.subf %5, %6 : f32
            affine.store %7, %alloc_8[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }

    // Exponentiation
    // %8 = tosa.exp %7 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    // Allocate space and perform exponentiation.
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %alloc_8[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = math.exp %5 : f32
            affine.store %6, %alloc_9[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }

    // Reduce Sum
    // %9 = tosa.reduce_sum %8 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    // Allocate space and initialize the output.
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.store %cst_0, %alloc_10[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
        }
      }
    }
    // Perform reduce sum operation.
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %alloc_9[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = affine.load %alloc_10[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
            %7 = arith.addf %5, %6 : f32
            affine.store %7, %alloc_10[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
          }
        }
      }
    }

    // Reciprocal
    // %10 = tosa.reciprocal %9 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %expand_shape_11 = memref.expand_shape %alloc_10 [[0], [1], [2, 3]] output_shape [1, 32, 40, 1]: memref<1x32x40xf32> into memref<1x32x40x1xf32>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x1xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 1 {
            %5 = affine.load %expand_shape_11[%c0, %arg5, %arg6, %c0] : memref<1x32x40x1xf32>
            %6 = arith.divf %cst_1, %5 : f32
            affine.store %6, %alloc_12[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x1xf32>
          }
        }
      }
    }

    // Multiplication
    // %11 = tosa.mul %8, %10 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %alloc_9[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %6 = affine.load %alloc_12[%c0, %arg5, %arg6, %c0] : memref<1x32x40x1xf32>
            %7 = arith.mulf %5, %6 : f32
            affine.store %7, %alloc_13[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }

    // Prepare MatMul input memref.
    // %12 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    // %13 = tosa.add %11, %12 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    // %14 = tosa.reshape %13 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    // %15 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    // %16 = tosa.add %t3, %15 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    // %17 = tosa.reshape %16 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %collapse_shape = memref.collapse_shape %alloc_13 [[0, 1], [2], [3]] : memref<1x32x40x40xf32> into memref<32x40x40xf32>
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x128xf32>
    // SSA value %0 is from %arg3
    memref.copy %0, %alloc_14 : memref<1x32x40x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x32x40x128xf32>
    %collapse_shape_15 = memref.collapse_shape %alloc_14 [[0, 1], [2], [3]] : memref<1x32x40x128xf32> into memref<32x40x128xf32>

    // MatMul
    // %18 = tosa.matmul %14, %17 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    // Allocate space and initialize output.
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32x40x128xf32>
    affine.for %arg4 = 0 to 32 {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 128 {
          affine.store %cst_0, %alloc_16[%arg4, %arg5, %arg6] : memref<32x40x128xf32>
        }
      }
    }
    // Perform MatMul core operations: multiplication and addition.
    affine.for %arg4 = 0 to 32 {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 128 {
          affine.for %arg7 = 0 to 40 {
            %5 = affine.load %collapse_shape[%arg4, %arg5, %arg7] : memref<32x40x40xf32>
            %6 = affine.load %collapse_shape_15[%arg4, %arg7, %arg6] : memref<32x40x128xf32>
            %7 = affine.load %alloc_16[%arg4, %arg5, %arg6] : memref<32x40x128xf32>
            %8 = arith.mulf %5, %6 : f32
            %9 = arith.addf %7, %8 : f32
            affine.store %9, %alloc_16[%arg4, %arg5, %arg6] : memref<32x40x128xf32>
          }
        }
      }
    }

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64

    %cast = memref.cast %alloc_16 : memref<32x40x128xf32> to memref<*xf32>
    // %4 = bufferization.to_tensor %cast : memref<*xf32> to tensor<*xf32>

    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [32, 40, 128] strides = [5120, 128, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SAME: [8{{(, 8)*}}],

    // Print results.
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    // Print timings.
    vector.print %time : f64

    return
  }
  func.func @main() {
    %0 = memref.get_global @__constant_32x40x128xf32 : memref<32x40x128xf32>
    %1 = bufferization.to_tensor %0 restrict: memref<32x40x128xf32> to tensor<32x40x128xf32>
    %2 = memref.get_global @__constant_32x128x40xf32 : memref<32x128x40xf32>
    %3 = bufferization.to_tensor %2 restrict: memref<32x128x40xf32> to tensor<32x128x40xf32>
    %4 = memref.get_global @__constant_1x1x40x40xf32 : memref<1x1x40x40xf32>
    %5 = bufferization.to_tensor %4 restrict: memref<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %6 = memref.get_global @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32>
    %7 = bufferization.to_tensor %6 restrict: memref<1x32x40x128xf32> to tensor<1x32x40x128xf32>
    call @kenerl(%1, %3, %5, %7) : (tensor<32x40x128xf32>, tensor<32x128x40xf32>, tensor<1x1x40x40xf32>, tensor<1x32x40x128xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>)
}
