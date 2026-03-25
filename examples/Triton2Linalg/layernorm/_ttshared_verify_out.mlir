#map = affine_map<(d0) -> (d0)>
module {
  func.func @_layer_norm_fwd_fused(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: memref<*xf32> {tt.divisibility = 16 : i32}, %arg5: memref<*xf32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c512_i32 = arith.constant 512 : i32
    %c512 = arith.constant 512 : index
    %cst_1 = arith.constant 0.000000e+00 : f16
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<512xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    %2 = arith.muli %arg12, %arg6 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = tensor.empty() : tensor<512xi32>
    %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%4 : tensor<512xi32>) {
    ^bb0(%out: i32):
      %20 = linalg.index 0 : index
      %21 = arith.index_cast %20 : index to i32
      linalg.yield %21 : i32
    } -> tensor<512xi32>
    %6 = linalg.fill ins(%arg7 : i32) outs(%4 : tensor<512xi32>) -> tensor<512xi32>
    %7 = scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32 iter_args(%arg16 = %1) -> (tensor<512xf32>)  : i32 {
      %20 = arith.index_cast %arg15 : i32 to index
      %21 = arith.addi %3, %20 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %22 = arith.addi %20, %c512 : index
      %23 = arith.index_cast %arg7 : i32 to index
      %24 = arith.minsi %22, %23 : index
      %25 = arith.maxsi %24, %20 : index
      %26 = arith.subi %25, %20 : index
      %alloc = memref.alloc() : memref<512xf16>
      %27 = arith.cmpi slt, %26, %c512 : index
      scf.if %27 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc : memref<512xf16>)
      }
      %subview = memref.subview %reinterpret_cast_6[0] [%26] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_7 = memref.subview %alloc[0] [%26] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_7 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %28 = bufferization.to_tensor %alloc restrict writable : memref<512xf16> to tensor<512xf16>
      %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%28 : tensor<512xf16>) outs(%0 : tensor<512xf32>) {
      ^bb0(%in: f16, %out: f32):
        %31 = arith.extf %in : f16 to f32
        linalg.yield %31 : f32
      } -> tensor<512xf32>
      %30 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %29 : tensor<512xf32>, tensor<512xf32>) outs(%arg16 : tensor<512xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %31 = arith.addf %in, %in_8 : f32
        linalg.yield %31 : f32
      } -> tensor<512xf32>
      scf.yield %30 : tensor<512xf32>
    }
    %8 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %8[] : tensor<f32>
    %reduced = linalg.reduce ins(%7 : tensor<512xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %20 = arith.addf %in, %init : f32
        linalg.yield %20 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %9 = arith.sitofp %arg7 : i32 to f32
    %10 = arith.divf %extracted, %9 : f32
    %11 = linalg.fill ins(%10 : f32) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    %12 = scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32 iter_args(%arg16 = %1) -> (tensor<512xf32>)  : i32 {
      %20 = linalg.fill ins(%arg15 : i32) outs(%4 : tensor<512xi32>) -> tensor<512xi32>
      %21 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%20, %5 : tensor<512xi32>, tensor<512xi32>) outs(%20 : tensor<512xi32>) {
      ^bb0(%in: i32, %in_8: i32, %out: i32):
        %38 = arith.addi %in, %in_8 : i32
        linalg.yield %38 : i32
      } -> tensor<512xi32>
      %22 = tensor.empty() : tensor<512xi1>
      %23 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%21, %6 : tensor<512xi32>, tensor<512xi32>) outs(%22 : tensor<512xi1>) {
      ^bb0(%in: i32, %in_8: i32, %out: i1):
        %38 = arith.cmpi slt, %in, %in_8 : i32
        linalg.yield %38 : i1
      } -> tensor<512xi1>
      %24 = arith.index_cast %arg15 : i32 to index
      %25 = arith.addi %3, %24 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %26 = arith.addi %24, %c512 : index
      %27 = arith.index_cast %arg7 : i32 to index
      %28 = arith.minsi %26, %27 : index
      %29 = arith.maxsi %28, %24 : index
      %30 = arith.subi %29, %24 : index
      %alloc = memref.alloc() : memref<512xf16>
      %31 = arith.cmpi slt, %30, %c512 : index
      scf.if %31 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc : memref<512xf16>)
      }
      %subview = memref.subview %reinterpret_cast_6[0] [%30] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_7 = memref.subview %alloc[0] [%30] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_7 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %32 = bufferization.to_tensor %alloc restrict writable : memref<512xf16> to tensor<512xf16>
      %33 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%32 : tensor<512xf16>) outs(%0 : tensor<512xf32>) {
      ^bb0(%in: f16, %out: f32):
        %38 = arith.extf %in : f16 to f32
        linalg.yield %38 : f32
      } -> tensor<512xf32>
      %34 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%33, %11 : tensor<512xf32>, tensor<512xf32>) outs(%33 : tensor<512xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %38 = arith.subf %in, %in_8 : f32
        linalg.yield %38 : f32
      } -> tensor<512xf32>
      %35 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%23, %34, %1 : tensor<512xi1>, tensor<512xf32>, tensor<512xf32>) outs(%34 : tensor<512xf32>) {
      ^bb0(%in: i1, %in_8: f32, %in_9: f32, %out: f32):
        %38 = arith.select %in, %in_8, %in_9 : f32
        linalg.yield %38 : f32
      } -> tensor<512xf32>
      %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%35, %35 : tensor<512xf32>, tensor<512xf32>) outs(%35 : tensor<512xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %38 = arith.mulf %in, %in_8 : f32
        linalg.yield %38 : f32
      } -> tensor<512xf32>
      %37 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %36 : tensor<512xf32>, tensor<512xf32>) outs(%arg16 : tensor<512xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %38 = arith.addf %in, %in_8 : f32
        linalg.yield %38 : f32
      } -> tensor<512xf32>
      scf.yield %37 : tensor<512xf32>
    }
    %13 = bufferization.alloc_tensor() : tensor<f32>
    %inserted_2 = tensor.insert %cst into %13[] : tensor<f32>
    %reduced_3 = linalg.reduce ins(%12 : tensor<512xf32>) outs(%inserted_2 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %20 = arith.addf %in, %init : f32
        linalg.yield %20 : f32
      }
    %extracted_4 = tensor.extract %reduced_3[] : tensor<f32>
    %14 = arith.divf %extracted_4, %9 : f32
    %15 = arith.addf %14, %arg8 : f32
    %16 = math.sqrt %15 : f32
    %17 = arith.divf %cst_0, %16 : f32
    %18 = arith.index_cast %arg12 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%18], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %10, %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%18], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %17, %reinterpret_cast_5[0] : memref<1xf32, strided<[1], offset: ?>>
    %19 = linalg.fill ins(%17 : f32) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32  : i32 {
      %20 = arith.index_cast %arg15 : i32 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %21 = arith.addi %20, %c512 : index
      %22 = arith.index_cast %arg7 : i32 to index
      %23 = arith.minsi %21, %22 : index
      %24 = arith.maxsi %23, %20 : index
      %25 = arith.subi %24, %20 : index
      %alloc = memref.alloc() : memref<512xf16>
      %subview = memref.subview %reinterpret_cast_6[0] [%25] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_7 = memref.subview %alloc[0] [%25] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_7 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %26 = bufferization.to_tensor %alloc restrict writable : memref<512xf16> to tensor<512xf16>
      %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%20], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %alloc_9 = memref.alloc() : memref<512xf16>
      %subview_10 = memref.subview %reinterpret_cast_8[0] [%25] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_11 = memref.subview %alloc_9[0] [%25] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview_10, %subview_11 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %27 = bufferization.to_tensor %alloc_9 restrict writable : memref<512xf16> to tensor<512xf16>
      %28 = arith.addi %3, %20 : index
      %reinterpret_cast_12 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %alloc_13 = memref.alloc() : memref<512xf16>
      %29 = arith.cmpi slt, %25, %c512 : index
      scf.if %29 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc_13 : memref<512xf16>)
      }
      %subview_14 = memref.subview %reinterpret_cast_12[0] [%25] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_15 = memref.subview %alloc_13[0] [%25] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview_14, %subview_15 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %30 = bufferization.to_tensor %alloc_13 restrict writable : memref<512xf16> to tensor<512xf16>
      %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%30 : tensor<512xf16>) outs(%0 : tensor<512xf32>) {
      ^bb0(%in: f16, %out: f32):
        %40 = arith.extf %in : f16 to f32
        linalg.yield %40 : f32
      } -> tensor<512xf32>
      %32 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%31, %11 : tensor<512xf32>, tensor<512xf32>) outs(%31 : tensor<512xf32>) {
      ^bb0(%in: f32, %in_18: f32, %out: f32):
        %40 = arith.subf %in, %in_18 : f32
        linalg.yield %40 : f32
      } -> tensor<512xf32>
      %33 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%32, %19 : tensor<512xf32>, tensor<512xf32>) outs(%32 : tensor<512xf32>) {
      ^bb0(%in: f32, %in_18: f32, %out: f32):
        %40 = arith.mulf %in, %in_18 : f32
        linalg.yield %40 : f32
      } -> tensor<512xf32>
      %34 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%26 : tensor<512xf16>) outs(%0 : tensor<512xf32>) {
      ^bb0(%in: f16, %out: f32):
        %40 = arith.extf %in : f16 to f32
        linalg.yield %40 : f32
      } -> tensor<512xf32>
      %35 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%33, %34 : tensor<512xf32>, tensor<512xf32>) outs(%33 : tensor<512xf32>) {
      ^bb0(%in: f32, %in_18: f32, %out: f32):
        %40 = arith.mulf %in, %in_18 : f32
        linalg.yield %40 : f32
      } -> tensor<512xf32>
      %36 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%27 : tensor<512xf16>) outs(%0 : tensor<512xf32>) {
      ^bb0(%in: f16, %out: f32):
        %40 = arith.extf %in : f16 to f32
        linalg.yield %40 : f32
      } -> tensor<512xf32>
      %37 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%35, %36 : tensor<512xf32>, tensor<512xf32>) outs(%35 : tensor<512xf32>) {
      ^bb0(%in: f32, %in_18: f32, %out: f32):
        %40 = arith.addf %in, %in_18 : f32
        linalg.yield %40 : f32
      } -> tensor<512xf32>
      %reinterpret_cast_16 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %38 = tensor.empty() : tensor<512xf16>
      %39 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%37 : tensor<512xf32>) outs(%38 : tensor<512xf16>) {
      ^bb0(%in: f32, %out: f16):
        %40 = arith.truncf %in : f32 to f16
        linalg.yield %40 : f16
      } -> tensor<512xf16>
      %extracted_slice = tensor.extract_slice %39[0] [%25] [1] : tensor<512xf16> to tensor<?xf16>
      %subview_17 = memref.subview %reinterpret_cast_16[0] [%25] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_17 : (tensor<?xf16>, memref<?xf16, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

