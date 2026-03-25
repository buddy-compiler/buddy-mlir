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
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_2[] : memref<f32>
    linalg.reduce ins(%alloc : memref<256xf32>) outs(%alloc_2 : memref<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %10 = arith.maxnumf %in, %init : f32
        linalg.yield %10 : f32
      }
    %6 = memref.load %alloc_2[] : memref<f32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
    linalg.fill ins(%6 : f32) outs(%alloc_3 : memref<256xf32>)
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc, %alloc_3 : memref<256xf32>, memref<256xf32>) outs(%alloc : memref<256xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %10 = arith.subf %in, %in_8 : f32
      linalg.yield %10 : f32
    }
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc : memref<256xf32>) outs(%alloc : memref<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = math.exp %in : f32
      linalg.yield %10 : f32
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_4[] : memref<f32>
    linalg.reduce ins(%alloc : memref<256xf32>) outs(%alloc_4 : memref<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %10 = arith.addf %in, %init : f32
        linalg.yield %10 : f32
      }
    %7 = memref.load %alloc_4[] : memref<f32>
    linalg.fill ins(%7 : f32) outs(%alloc_3 : memref<256xf32>)
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc, %alloc_3 : memref<256xf32>, memref<256xf32>) outs(%alloc : memref<256xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %10 = arith.divf %in, %in_8 : f32
      linalg.yield %10 : f32
    }
    %8 = arith.muli %arg8, %arg3 : i32
    %9 = arith.index_cast %8 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
    %subview_6 = memref.subview %alloc[0] [%4] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    %subview_7 = memref.subview %reinterpret_cast_5[0] [%4] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    memref.copy %subview_6, %subview_7 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1], offset: ?>>
    return
  }
  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %alloc = memref.alloc() : memref<256xf32>
    %alloc_0 = memref.alloc() : memref<256xf32>
    %cast = memref.cast %alloc : memref<256xf32> to memref<*xf32>
    %cast_1 = memref.cast %alloc_0 : memref<256xf32> to memref<*xf32>
    call @softmax_kernel(%cast, %cast_1, %c256_i32, %c256_i32, %c256_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : (memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    memref.dealloc %alloc : memref<256xf32>
    memref.dealloc %alloc_0 : memref<256xf32>
    return %c0_i32 : i32
  }
}

