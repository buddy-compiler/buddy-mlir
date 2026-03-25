#map = affine_map<(d0) -> (d0)>
module {
  func.func @_layer_norm_fwd_fused(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: memref<*xf32> {tt.divisibility = 16 : i32}, %arg5: memref<*xf32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c512_i32 = arith.constant 512 : i32
    %c512 = arith.constant 512 : index
    %cst_1 = arith.constant 0.000000e+00 : f16
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<512xf32>)
    %0 = arith.muli %arg12, %arg6 : i32
    %1 = arith.index_cast %0 : i32 to index
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<512xi32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<512xi32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_4 : memref<512xi32>) {
    ^bb0(%out: i32):
      %13 = linalg.index 0 : index
      %14 = arith.index_cast %13 : index to i32
      linalg.yield %14 : i32
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<512xi32>
    linalg.fill ins(%arg7 : i32) outs(%alloc_5 : memref<512xi32>)
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    memref.copy %alloc_2, %alloc_6 : memref<512xf32> to memref<512xf32>
    %2 = scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32 iter_args(%arg16 = %alloc_6) -> (memref<512xf32>)  : i32 {
      %13 = arith.index_cast %arg15 : i32 to index
      %14 = arith.addi %1, %13 : index
      %reinterpret_cast_13 = memref.reinterpret_cast %arg0 to offset: [%14], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %15 = arith.addi %13, %c512 : index
      %16 = arith.index_cast %arg7 : i32 to index
      %17 = arith.minsi %15, %16 : index
      %18 = arith.maxsi %17, %13 : index
      %19 = arith.subi %18, %13 : index
      %alloc_14 = memref.alloc() : memref<512xf16>
      %20 = arith.cmpi slt, %19, %c512 : index
      scf.if %20 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc_14 : memref<512xf16>)
      }
      %subview = memref.subview %reinterpret_cast_13[0] [%19] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_15 = memref.subview %alloc_14[0] [%19] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_15 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_14 : memref<512xf16>) outs(%alloc : memref<512xf32>) {
      ^bb0(%in: f16, %out: f32):
        %21 = arith.extf %in : f16 to f32
        linalg.yield %21 : f32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %alloc : memref<512xf32>, memref<512xf32>) outs(%arg16 : memref<512xf32>) {
      ^bb0(%in: f32, %in_16: f32, %out: f32):
        %21 = arith.addf %in, %in_16 : f32
        linalg.yield %21 : f32
      }
      scf.yield %arg16 : memref<512xf32>
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_7[] : memref<f32>
    linalg.reduce ins(%2 : memref<512xf32>) outs(%alloc_7 : memref<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %3 = memref.load %alloc_7[] : memref<f32>
    %4 = arith.sitofp %arg7 : i32 to f32
    %5 = arith.divf %3, %4 : f32
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    linalg.fill ins(%5 : f32) outs(%alloc_8 : memref<512xf32>)
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    memref.copy %alloc_2, %alloc_9 : memref<512xf32> to memref<512xf32>
    %6 = scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32 iter_args(%arg16 = %alloc_9) -> (memref<512xf32>)  : i32 {
      linalg.fill ins(%arg15 : i32) outs(%alloc_3 : memref<512xi32>)
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc_3, %alloc_4 : memref<512xi32>, memref<512xi32>) outs(%alloc_3 : memref<512xi32>) {
      ^bb0(%in: i32, %in_17: i32, %out: i32):
        %21 = arith.addi %in, %in_17 : i32
        linalg.yield %21 : i32
      }
      %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<512xi1>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc_3, %alloc_5 : memref<512xi32>, memref<512xi32>) outs(%alloc_13 : memref<512xi1>) {
      ^bb0(%in: i32, %in_17: i32, %out: i1):
        %21 = arith.cmpi slt, %in, %in_17 : i32
        linalg.yield %21 : i1
      }
      %13 = arith.index_cast %arg15 : i32 to index
      %14 = arith.addi %1, %13 : index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg0 to offset: [%14], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %15 = arith.addi %13, %c512 : index
      %16 = arith.index_cast %arg7 : i32 to index
      %17 = arith.minsi %15, %16 : index
      %18 = arith.maxsi %17, %13 : index
      %19 = arith.subi %18, %13 : index
      %alloc_15 = memref.alloc() : memref<512xf16>
      %20 = arith.cmpi slt, %19, %c512 : index
      scf.if %20 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc_15 : memref<512xf16>)
      }
      %subview = memref.subview %reinterpret_cast_14[0] [%19] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_16 = memref.subview %alloc_15[0] [%19] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_16 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_15 : memref<512xf16>) outs(%alloc : memref<512xf32>) {
      ^bb0(%in: f16, %out: f32):
        %21 = arith.extf %in : f16 to f32
        linalg.yield %21 : f32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc, %alloc_8 : memref<512xf32>, memref<512xf32>) outs(%alloc : memref<512xf32>) {
      ^bb0(%in: f32, %in_17: f32, %out: f32):
        %21 = arith.subf %in, %in_17 : f32
        linalg.yield %21 : f32
      }
      linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%alloc_13, %alloc, %alloc_2 : memref<512xi1>, memref<512xf32>, memref<512xf32>) outs(%alloc : memref<512xf32>) {
      ^bb0(%in: i1, %in_17: f32, %in_18: f32, %out: f32):
        %21 = arith.select %in, %in_17, %in_18 : f32
        linalg.yield %21 : f32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc, %alloc : memref<512xf32>, memref<512xf32>) outs(%alloc : memref<512xf32>) {
      ^bb0(%in: f32, %in_17: f32, %out: f32):
        %21 = arith.mulf %in, %in_17 : f32
        linalg.yield %21 : f32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %alloc : memref<512xf32>, memref<512xf32>) outs(%arg16 : memref<512xf32>) {
      ^bb0(%in: f32, %in_17: f32, %out: f32):
        %21 = arith.addf %in, %in_17 : f32
        linalg.yield %21 : f32
      }
      scf.yield %arg16 : memref<512xf32>
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_10[] : memref<f32>
    linalg.reduce ins(%6 : memref<512xf32>) outs(%alloc_10 : memref<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %7 = memref.load %alloc_10[] : memref<f32>
    %8 = arith.divf %7, %4 : f32
    %9 = arith.addf %8, %arg8 : f32
    %10 = math.sqrt %9 : f32
    %11 = arith.divf %cst_0, %10 : f32
    %12 = arith.index_cast %arg12 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%12], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %5, %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
    %reinterpret_cast_11 = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %11, %reinterpret_cast_11[0] : memref<1xf32, strided<[1], offset: ?>>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    linalg.fill ins(%11 : f32) outs(%alloc_12 : memref<512xf32>)
    scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32  : i32 {
      %13 = arith.index_cast %arg15 : i32 to index
      %reinterpret_cast_13 = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %14 = arith.addi %13, %c512 : index
      %15 = arith.index_cast %arg7 : i32 to index
      %16 = arith.minsi %14, %15 : index
      %17 = arith.maxsi %16, %13 : index
      %18 = arith.subi %17, %13 : index
      %alloc_14 = memref.alloc() : memref<512xf16>
      %subview = memref.subview %reinterpret_cast_13[0] [%18] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_15 = memref.subview %alloc_14[0] [%18] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_15 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %reinterpret_cast_16 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %alloc_17 = memref.alloc() : memref<512xf16>
      %subview_18 = memref.subview %reinterpret_cast_16[0] [%18] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_19 = memref.subview %alloc_17[0] [%18] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview_18, %subview_19 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %19 = arith.addi %1, %13 : index
      %reinterpret_cast_20 = memref.reinterpret_cast %arg0 to offset: [%19], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %alloc_21 = memref.alloc() : memref<512xf16>
      %20 = arith.cmpi slt, %18, %c512 : index
      scf.if %20 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc_21 : memref<512xf16>)
      }
      %subview_22 = memref.subview %reinterpret_cast_20[0] [%18] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_23 = memref.subview %alloc_21[0] [%18] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview_22, %subview_23 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_21 : memref<512xf16>) outs(%alloc_24 : memref<512xf32>) {
      ^bb0(%in: f16, %out: f32):
        %21 = arith.extf %in : f16 to f32
        linalg.yield %21 : f32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc_24, %alloc_8 : memref<512xf32>, memref<512xf32>) outs(%alloc_24 : memref<512xf32>) {
      ^bb0(%in: f32, %in_29: f32, %out: f32):
        %21 = arith.subf %in, %in_29 : f32
        linalg.yield %21 : f32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc_24, %alloc_12 : memref<512xf32>, memref<512xf32>) outs(%alloc_24 : memref<512xf32>) {
      ^bb0(%in: f32, %in_29: f32, %out: f32):
        %21 = arith.mulf %in, %in_29 : f32
        linalg.yield %21 : f32
      }
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_14 : memref<512xf16>) outs(%alloc : memref<512xf32>) {
      ^bb0(%in: f16, %out: f32):
        %21 = arith.extf %in : f16 to f32
        linalg.yield %21 : f32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc_24, %alloc : memref<512xf32>, memref<512xf32>) outs(%alloc_24 : memref<512xf32>) {
      ^bb0(%in: f32, %in_29: f32, %out: f32):
        %21 = arith.mulf %in, %in_29 : f32
        linalg.yield %21 : f32
      }
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_17 : memref<512xf16>) outs(%alloc : memref<512xf32>) {
      ^bb0(%in: f16, %out: f32):
        %21 = arith.extf %in : f16 to f32
        linalg.yield %21 : f32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc_24, %alloc : memref<512xf32>, memref<512xf32>) outs(%alloc_24 : memref<512xf32>) {
      ^bb0(%in: f32, %in_29: f32, %out: f32):
        %21 = arith.addf %in, %in_29 : f32
        linalg.yield %21 : f32
      }
      %reinterpret_cast_25 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<512xf16>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_24 : memref<512xf32>) outs(%alloc_26 : memref<512xf16>) {
      ^bb0(%in: f32, %out: f16):
        %21 = arith.truncf %in : f32 to f16
        linalg.yield %21 : f16
      }
      %subview_27 = memref.subview %alloc_26[0] [%18] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      %subview_28 = memref.subview %reinterpret_cast_25[0] [%18] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      memref.copy %subview_27, %subview_28 : memref<?xf16, strided<[1]>> to memref<?xf16, strided<[1], offset: ?>>
    }
    return
  }
  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %cst = arith.constant 9.99999974E-6 : f32
    %alloc = memref.alloc() : memref<512xf16>
    %alloc_0 = memref.alloc() : memref<512xf16>
    %alloc_1 = memref.alloc() : memref<512xf16>
    %alloc_2 = memref.alloc() : memref<512xf16>
    %alloc_3 = memref.alloc() : memref<1xf32>
    %alloc_4 = memref.alloc() : memref<1xf32>
    %cast = memref.cast %alloc : memref<512xf16> to memref<*xf16>
    %cast_5 = memref.cast %alloc_0 : memref<512xf16> to memref<*xf16>
    %cast_6 = memref.cast %alloc_1 : memref<512xf16> to memref<*xf16>
    %cast_7 = memref.cast %alloc_2 : memref<512xf16> to memref<*xf16>
    %cast_8 = memref.cast %alloc_3 : memref<1xf32> to memref<*xf32>
    %cast_9 = memref.cast %alloc_4 : memref<1xf32> to memref<*xf32>
    call @_layer_norm_fwd_fused(%cast, %cast_5, %cast_6, %cast_7, %cast_8, %cast_9, %c512_i32, %c512_i32, %cst, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : (memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf32>, memref<*xf32>, i32, i32, f32, i32, i32, i32, i32, i32, i32) -> ()
    memref.dealloc %alloc : memref<512xf16>
    memref.dealloc %alloc_0 : memref<512xf16>
    memref.dealloc %alloc_1 : memref<512xf16>
    memref.dealloc %alloc_2 : memref<512xf16>
    memref.dealloc %alloc_3 : memref<1xf32>
    memref.dealloc %alloc_4 : memref<1xf32>
    return %c0_i32 : i32
  }
}

