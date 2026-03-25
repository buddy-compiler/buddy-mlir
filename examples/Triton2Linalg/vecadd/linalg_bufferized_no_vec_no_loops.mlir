#map = affine_map<(d0) -> (d0)>
module {
  func.func @add_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %0 = arith.muli %arg7, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %2 = arith.addi %1, %c1024 : index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.minsi %2, %3 : index
    %5 = arith.maxsi %4, %1 : index
    %6 = arith.subi %5, %1 : index
    %alloc = memref.alloc() : memref<1024xf32>
    %subview = memref.subview %reinterpret_cast[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1024xf32>
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc, %alloc_2 : memref<1024xf32>, memref<1024xf32>) outs(%alloc : memref<1024xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %7 = arith.addf %in, %in_8 : f32
      linalg.yield %7 : f32
    }
    %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %subview_6 = memref.subview %alloc[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    %subview_7 = memref.subview %reinterpret_cast_5[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    memref.copy %subview_6, %subview_7 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1], offset: ?>>
    return
  }
  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %alloc = memref.alloc() : memref<1024xf32>
    %alloc_0 = memref.alloc() : memref<1024xf32>
    %alloc_1 = memref.alloc() : memref<1024xf32>
    %cast = memref.cast %alloc : memref<1024xf32> to memref<*xf32>
    %cast_2 = memref.cast %alloc_0 : memref<1024xf32> to memref<*xf32>
    %cast_3 = memref.cast %alloc_1 : memref<1024xf32> to memref<*xf32>
    call @add_kernel(%cast, %cast_2, %cast_3, %c1024_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32, i32) -> ()
    memref.dealloc %alloc : memref<1024xf32>
    memref.dealloc %alloc_0 : memref<1024xf32>
    memref.dealloc %alloc_1 : memref<1024xf32>
    return %c0_i32 : i32
  }
}

