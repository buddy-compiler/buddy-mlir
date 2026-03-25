#map = affine_map<(d0) -> (d0)>
module {
  func.func @softmax_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = arith.muli %arg8, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
    %2 = arith.index_cast %arg4 : i32 to index
    %3 = arith.minsi %2, %c256 : index
    %4 = arith.maxsi %3, %c0 : index
    %alloc = memref.alloc() : memref<256xf32>
    %5 = arith.cmpi slt, %4, %c256 : index
    scf.if %5 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<256xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%4] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_1 = memref.subview %alloc[0] [%4] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %6 = bufferization.to_tensor %alloc restrict writable : memref<256xf32> to tensor<256xf32>
    %7 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %7[] : tensor<f32>
    %reduced = linalg.reduce ins(%6 : tensor<256xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %17 = arith.maxnumf %in, %init : f32
        linalg.yield %17 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %8 = tensor.empty() : tensor<256xf32>
    %9 = linalg.fill ins(%extracted : f32) outs(%8 : tensor<256xf32>) -> tensor<256xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%6, %9 : tensor<256xf32>, tensor<256xf32>) outs(%6 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %17 = arith.subf %in, %in_7 : f32
      linalg.yield %17 : f32
    } -> tensor<256xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%10 : tensor<256xf32>) outs(%10 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = math.exp %in : f32
      linalg.yield %17 : f32
    } -> tensor<256xf32>
    %12 = bufferization.alloc_tensor() : tensor<f32>
    %inserted_2 = tensor.insert %cst_0 into %12[] : tensor<f32>
    %reduced_3 = linalg.reduce ins(%11 : tensor<256xf32>) outs(%inserted_2 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %17 = arith.addf %in, %init : f32
        linalg.yield %17 : f32
      }
    %extracted_4 = tensor.extract %reduced_3[] : tensor<f32>
    %13 = linalg.fill ins(%extracted_4 : f32) outs(%8 : tensor<256xf32>) -> tensor<256xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%11, %13 : tensor<256xf32>, tensor<256xf32>) outs(%11 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %17 = arith.divf %in, %in_7 : f32
      linalg.yield %17 : f32
    } -> tensor<256xf32>
    %15 = arith.muli %arg8, %arg3 : i32
    %16 = arith.index_cast %15 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %14[0] [%4] [1] : tensor<256xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%4] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }

  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %out = memref.alloc() : memref<256xf32>
    %inp = memref.alloc() : memref<256xf32>
    %out_u = memref.cast %out : memref<256xf32> to memref<*xf32>
    %inp_u = memref.cast %inp : memref<256xf32> to memref<*xf32>
    call @softmax_kernel(%out_u, %inp_u, %c256_i32, %c256_i32, %c256_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : (memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    memref.dealloc %out : memref<256xf32>
    memref.dealloc %inp : memref<256xf32>
    return %c0_i32 : i32
  }
}
