#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %c63_i32 = arith.constant 63 : i32
    %c15_i32 = arith.constant 15 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %2 = arith.addi %arg3, %c31_i32 : i32
    %3 = arith.divsi %2, %c32_i32 : i32
    %4 = arith.addi %arg4, %c63_i32 : i32
    %5 = arith.divsi %4, %c64_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %arg12, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.minsi %9, %c8_i32 : i32
    %11 = arith.remsi %arg12, %10 : i32
    %12 = arith.addi %8, %11 : i32
    %13 = arith.remsi %arg12, %6 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c32_i32 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.muli %14, %c64_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.index_cast %arg3 : i32 to index
    %20 = arith.index_cast %arg6 : i32 to index
    %21 = arith.muli %16, %20 : index
    %22 = arith.muli %19, %20 : index
    %23 = arith.index_cast %arg7 : i32 to index
    %24 = arith.index_cast %arg4 : i32 to index
    %25 = arith.addi %arg5, %c15_i32 : i32
    %26 = arith.divsi %25, %c16_i32 : i32
    %27 = arith.muli %arg7, %c16_i32 : i32
    %28 = arith.index_cast %27 : i32 to index
    %29:3 = scf.for %arg15 = %c0_i32 to %26 step %c1_i32 iter_args(%arg16 = %21, %arg17 = %c0, %arg18 = %1) -> (index, index, tensor<32x64xf32>)  : i32 {
      %43 = arith.addi %arg17, %18 : index
      %44 = arith.remsi %43, %24 : index
      %45 = arith.subi %43, %44 : index
      %46 = arith.addi %44, %c64 : index
      %47 = arith.minsi %46, %24 : index
      %48 = arith.subi %47, %44 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%c16, %48], strides: [%23, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %49 = arith.subi %c64, %48 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [%c16, %49], strides: [%23, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %50 = arith.remsi %arg16, %20 : index
      %51 = arith.addi %22, %50 : index
      %52 = arith.subi %51, %arg16 : index
      %53 = arith.divsi %52, %20 : index
      %54 = arith.minsi %53, %c32 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%arg16], sizes: [%54, %c16], strides: [%20, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %55 = arith.subi %c32, %54 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [%55, %c16], strides: [%20, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %56 = arith.muli %arg15, %c16_i32 : i32
      %57 = arith.subi %arg5, %56 : i32
      %58 = arith.index_cast %57 : i32 to index
      %59 = arith.minsi %58, %c16 : index
      %60 = arith.maxsi %59, %c0 : index
      %alloc = memref.alloc() : memref<32x16xf32>
      %61 = arith.cmpi slt, %60, %c16 : index
      scf.if %61 {
        linalg.fill ins(%cst : f32) outs(%alloc : memref<32x16xf32>)
      }
      %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%54, %60] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%55, %60] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_6 = memref.subview %alloc[0, 0] [%54, %60] [1, 1] : memref<32x16xf32> to memref<?x?xf32, strided<[16, 1]>>
      %subview_7 = memref.subview %alloc[%54, 0] [%55, %60] [1, 1] : memref<32x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
      memref.copy %subview_4, %subview_6 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
      memref.copy %subview_5, %subview_7 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[16, 1], offset: ?>>
      %62 = bufferization.to_tensor %alloc restrict writable : memref<32x16xf32> to tensor<32x16xf32>
      %alloc_8 = memref.alloc() : memref<16x64xf32>
      scf.if %61 {
        linalg.fill ins(%cst : f32) outs(%alloc_8 : memref<16x64xf32>)
      }
      %subview_9 = memref.subview %reinterpret_cast_0[0, 0] [%60, %48] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_10 = memref.subview %reinterpret_cast_1[0, 0] [%60, %49] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_11 = memref.subview %alloc_8[0, 0] [%60, %48] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      %subview_12 = memref.subview %alloc_8[0, %48] [%60, %49] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %subview_9, %subview_11 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_10, %subview_12 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %63 = bufferization.to_tensor %alloc_8 restrict writable : memref<16x64xf32> to tensor<16x64xf32>
      %64 = linalg.matmul ins(%62, %63 : tensor<32x16xf32>, tensor<16x64xf32>) outs(%1 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %65 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg18, %64 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%arg18 : tensor<32x64xf32>) {
      ^bb0(%in: f32, %in_13: f32, %out: f32):
        %68 = arith.addf %in, %in_13 : f32
        linalg.yield %68 : f32
      } -> tensor<32x64xf32>
      %66 = arith.addi %arg16, %c16 : index
      %67 = arith.addi %arg17, %28 : index
      scf.yield %66, %67, %65 : index, index, tensor<32x64xf32>
    }
    %30 = arith.index_cast %arg8 : i32 to index
    %31 = arith.muli %16, %30 : index
    %32 = arith.addi %31, %18 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [32, 64], strides: [%30, 1] : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
    %33 = arith.addi %16, %c32 : index
    %34 = arith.minsi %33, %19 : index
    %35 = arith.maxsi %34, %16 : index
    %36 = arith.subi %35, %16 : index
    %37 = arith.addi %18, %c64 : index
    %38 = arith.minsi %37, %24 : index
    %39 = arith.maxsi %38, %18 : index
    %40 = arith.subi %39, %18 : index
    %41 = arith.minsi %36, %c32 : index
    %42 = arith.minsi %40, %c64 : index
    %extracted_slice = tensor.extract_slice %29#2[0, 0] [%41, %42] [1, 1] : tensor<32x64xf32> to tensor<?x?xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%41, %42] [1, 1] : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}

