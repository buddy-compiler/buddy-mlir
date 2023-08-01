#map = affine_map<(d0) -> (d0 floordiv 9)>
#map1 = affine_map<(d0, d1) -> (d0 floordiv 56 + (d1 mod 9) floordiv 3)>
#map2 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 56) * 56 - (d1 floordiv 3) * 3)>

func.func private @printMemrefF32(memref<*xf32>)

// Allocate and fill the memref according to the given layout.
func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
  scf.for %arg5 = %c0 to %arg0 step %c1 {
    scf.for %arg6 = %c0 to %arg1 step %c1 {
      scf.for %arg7 = %c0 to %arg2 step %c1 {
        scf.for %arg8 = %c0 to %arg3 step %c1 {
          memref.store %arg4, %0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xf32>
        }
      }
    }
  }
  return %0 : memref<?x?x?x?xf32>
}

// Optimize GEMM using tiling and vetorization. The tile size for the innermost three loops is [4, 32, 8].
func.func @batch_matmul_optimize(%arg0: memref<1x64x576xf32>, %arg1: memref<1x576x3136xf32>, %arg2: memref<1x64x3136xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<4x8xf32>
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %cst_0 = arith.constant dense<0.000000e+00> : vector<4x32xf32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c576 = arith.constant 576 : index
  %c3136 = arith.constant 3136 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  scf.for %arg3 = %c0 to %c64 step %c4 {
    scf.for %arg4 = %c0 to %c3136 step %c32 {
      scf.for %arg5 = %c0 to %c576 step %c8 {
        %subview = memref.subview %arg0[0, %arg3, %arg5] [1, 4, 8] [1, 1, 1] : memref<1x64x576xf32> to memref<1x4x8xf32, strided<[36864, 576, 1], offset: ?>>
        %subview_1 = memref.subview %arg1[0, %arg5, %arg4] [1, 8, 32] [1, 1, 1] : memref<1x576x3136xf32> to memref<1x8x32xf32, strided<[1806336, 3136, 1], offset: ?>>
        %subview_2 = memref.subview %arg2[0, %arg3, %arg4] [1, 4, 32] [1, 1, 1] : memref<1x64x3136xf32> to memref<1x4x32xf32, strided<[200704, 3136, 1], offset: ?>>
        %0 = vector.load %subview[%c0, %c0, %c0] : memref<1x4x8xf32, strided<[36864, 576, 1], offset: ?>>, vector<8xf32>
        %1 = vector.insert %0, %cst [0] : vector<8xf32> into vector<4x8xf32>
        %2 = vector.load %subview[%c0, %c1, %c0] : memref<1x4x8xf32, strided<[36864, 576, 1], offset: ?>>, vector<8xf32>
        %3 = vector.insert %2, %1 [1] : vector<8xf32> into vector<4x8xf32>
        %4 = vector.load %subview[%c0, %c2, %c0] : memref<1x4x8xf32, strided<[36864, 576, 1], offset: ?>>, vector<8xf32>
        %5 = vector.insert %4, %3 [2] : vector<8xf32> into vector<4x8xf32>
        %6 = vector.load %subview[%c0, %c3, %c0] : memref<1x4x8xf32, strided<[36864, 576, 1], offset: ?>>, vector<8xf32>
        %7 = vector.insert %6, %5 [3] : vector<8xf32> into vector<4x8xf32>
        %8 = vector.load %subview_1[%c0, %c0, %c0] : memref<1x8x32xf32, strided<[1806336, 3136, 1], offset: ?>>, vector<32xf32>
        %9 = vector.load %subview_1[%c0, %c1, %c0] : memref<1x8x32xf32, strided<[1806336, 3136, 1], offset: ?>>, vector<32xf32>
        %10 = vector.load %subview_1[%c0, %c2, %c0] : memref<1x8x32xf32, strided<[1806336, 3136, 1], offset: ?>>, vector<32xf32>
        %11 = vector.load %subview_1[%c0, %c3, %c0] : memref<1x8x32xf32, strided<[1806336, 3136, 1], offset: ?>>, vector<32xf32>
        %12 = vector.load %subview_1[%c0, %c4, %c0] : memref<1x8x32xf32, strided<[1806336, 3136, 1], offset: ?>>, vector<32xf32>
        %13 = vector.load %subview_1[%c0, %c5, %c0] : memref<1x8x32xf32, strided<[1806336, 3136, 1], offset: ?>>, vector<32xf32>
        %14 = vector.load %subview_1[%c0, %c6, %c0] : memref<1x8x32xf32, strided<[1806336, 3136, 1], offset: ?>>, vector<32xf32>
        %15 = vector.load %subview_1[%c0, %c7, %c0] : memref<1x8x32xf32, strided<[1806336, 3136, 1], offset: ?>>, vector<32xf32>
        %16 = vector.load %subview_2[%c0, %c0, %c0] : memref<1x4x32xf32, strided<[200704, 3136, 1], offset: ?>>, vector<32xf32>
        %17 = vector.insert %16, %cst_0 [0] : vector<32xf32> into vector<4x32xf32>
        %18 = vector.load %subview_2[%c0, %c1, %c0] : memref<1x4x32xf32, strided<[200704, 3136, 1], offset: ?>>, vector<32xf32>
        %19 = vector.insert %18, %17 [1] : vector<32xf32> into vector<4x32xf32>
        %20 = vector.load %subview_2[%c0, %c2, %c0] : memref<1x4x32xf32, strided<[200704, 3136, 1], offset: ?>>, vector<32xf32>
        %21 = vector.insert %20, %19 [2] : vector<32xf32> into vector<4x32xf32>
        %22 = vector.load %subview_2[%c0, %c3, %c0] : memref<1x4x32xf32, strided<[200704, 3136, 1], offset: ?>>, vector<32xf32>
        %23 = vector.insert %22, %21 [3] : vector<32xf32> into vector<4x32xf32>
        %24 = vector.shape_cast %7 : vector<4x8xf32> to vector<32xf32>
        %25 = vector.shuffle %24, %24 [0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31] : vector<32xf32>, vector<32xf32>
        %26 = vector.shape_cast %25 : vector<32xf32> to vector<8x4xf32>
        %27 = vector.extract %26[0] : vector<8x4xf32>
        %28 = vector.outerproduct %27, %8, %23 {kind = #vector.kind<add>} : vector<4xf32>, vector<32xf32>
        %29 = vector.extract %26[1] : vector<8x4xf32>
        %30 = vector.outerproduct %29, %9, %28 {kind = #vector.kind<add>} : vector<4xf32>, vector<32xf32>
        %31 = vector.extract %26[2] : vector<8x4xf32>
        %32 = vector.outerproduct %31, %10, %30 {kind = #vector.kind<add>} : vector<4xf32>, vector<32xf32>
        %33 = vector.extract %26[3] : vector<8x4xf32>
        %34 = vector.outerproduct %33, %11, %32 {kind = #vector.kind<add>} : vector<4xf32>, vector<32xf32>
        %35 = vector.extract %26[4] : vector<8x4xf32>
        %36 = vector.outerproduct %35, %12, %34 {kind = #vector.kind<add>} : vector<4xf32>, vector<32xf32>
        %37 = vector.extract %26[5] : vector<8x4xf32>
        %38 = vector.outerproduct %37, %13, %36 {kind = #vector.kind<add>} : vector<4xf32>, vector<32xf32>
        %39 = vector.extract %26[6] : vector<8x4xf32>
        %40 = vector.outerproduct %39, %14, %38 {kind = #vector.kind<add>} : vector<4xf32>, vector<32xf32>
        %41 = vector.extract %26[7] : vector<8x4xf32>
        %42 = vector.outerproduct %41, %15, %40 {kind = #vector.kind<add>} : vector<4xf32>, vector<32xf32>
        %43 = vector.extract %42[0] : vector<4x32xf32>
        vector.store %43, %subview_2[%c0, %c0, %c0] : memref<1x4x32xf32, strided<[200704, 3136, 1], offset: ?>>, vector<32xf32>
        %44 = vector.extract %42[1] : vector<4x32xf32>
        vector.store %44, %subview_2[%c0, %c1, %c0] : memref<1x4x32xf32, strided<[200704, 3136, 1], offset: ?>>, vector<32xf32>
        %45 = vector.extract %42[2] : vector<4x32xf32>
        vector.store %45, %subview_2[%c0, %c2, %c0] : memref<1x4x32xf32, strided<[200704, 3136, 1], offset: ?>>, vector<32xf32>
        %46 = vector.extract %42[3] : vector<4x32xf32>
        vector.store %46, %subview_2[%c0, %c3, %c0] : memref<1x4x32xf32, strided<[200704, 3136, 1], offset: ?>>, vector<32xf32>
      }
    }
  }
  return
}

func.func @conv2d_nchw_fchw_im2col(%input: memref<?x?x?x?xf32>, %kernel: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
  %input_specific = memref.cast %input : memref<?x?x?x?xf32> to memref<1x64x58x58xf32>
  %kernel_specific = memref.cast %kernel : memref<?x?x?x?xf32> to memref<64x64x3x3xf32>
  %output_specific = memref.cast %output : memref<?x?x?x?xf32> to memref<1x64x56x56xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c576 = arith.constant 576 : index // 576 = 64 * 3 * 3 = kernel's c*h*w
  %c3136 = arith.constant 3136 : index // 3136 = 56 * 56 = output's h*w
  %c64 = arith.constant 64 : index
  %kernel_collapse = memref.collapse_shape %kernel_specific [[0], [1, 2, 3]] : memref<64x64x3x3xf32> into memref<64x576xf32>
  %output_collapse = memref.collapse_shape %output_specific [[0], [1], [2, 3]] : memref<1x64x56x56xf32> into memref<1x64x3136xf32>
  %input_collapse = memref.alloc() {alignment = 64 : i64} : memref<1x576x3136xf32>
  // Apply im2col.
  scf.for %idx0 = %c0 to %c1 step %c1 {
    scf.for %idx1 = %c0 to %c576 step %c1 {
      scf.for %idx2 = %c0 to %c3136 step %c1 {
        %0 = affine.apply #map(%idx1)
        %1 = affine.apply #map1(%idx2, %idx1)
        %2 = affine.apply #map2(%idx2, %idx1)
        %3 = memref.load %input_specific[%idx0, %0, %1, %2] : memref<1x64x58x58xf32>
        memref.store %3, %input_collapse[%idx0, %idx1, %idx2] : memref<1x576x3136xf32>
      }
    }
  }
  // Implement optimized GEMM.
  %kernel_expand = memref.expand_shape %kernel_collapse [[0, 1], [2]] : memref<64x576xf32> into memref<1x64x576xf32>
  func.call @batch_matmul_optimize(%kernel_expand, %input_collapse, %output_collapse) : (memref<1x64x576xf32>, memref<1x576x3136xf32>, memref<1x64x3136xf32>) -> ()
  // Apply col2im.
  %result_mem = memref.expand_shape %output_collapse [[0], [1], [2, 3]] : memref<1x64x3136xf32> into memref<1x64x56x56xf32>
  memref.dealloc %input_collapse : memref<1x576x3136xf32>
  return
}

func.func @main() {
  // Intput and kernel value.
  %cst = arith.constant 1.000000e+00 : f32
  // Output value.
  %cst_0 = arith.constant 0.000000e+00 : f32

  // Define layout.
  %input_n = arith.constant 1 : index
  %input_c = arith.constant 64 : index
  %input_h = arith.constant 58 : index
  %input_w = arith.constant 58 : index

  %kernel_n = arith.constant 64 : index
  %kernel_c = arith.constant 64 : index
  %kernel_h = arith.constant 3 : index
  %kernel_w = arith.constant 3 : index

  %output_n = arith.constant 1 : index
  %output_c = arith.constant 64 : index
  %output_h = arith.constant 56 : index
  %output_w = arith.constant 56 : index

  // Define input, kernel, and output memref.
  %input = call @alloc_f32(%input_n, %input_c, %input_h, %input_w, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  %kernel = call @alloc_f32(%kernel_n, %kernel_c, %kernel_h, %kernel_w, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  %output = call @alloc_f32(%output_n, %output_c, %output_h, %output_w, %cst_0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

  // Perform convolution
  call @conv2d_nchw_fchw_im2col(%input, %kernel, %output) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()

  // Print the output
  %print_output = memref.cast %output : memref<?x?x?x?xf32> to memref<*xf32>
  call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()

  memref.dealloc %output : memref<?x?x?x?xf32>
  memref.dealloc %input : memref<?x?x?x?xf32>
  memref.dealloc %kernel : memref<?x?x?x?xf32>
  return
}
