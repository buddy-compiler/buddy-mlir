#map = affine_map<(d0) -> (d0)>
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
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
    %c0_0 = arith.constant 0 : index
    %c32_1 = arith.constant 32 : index
    affine.for %arg15 = #map(%c0_0) to #map(%c32_1) {
      %c4 = arith.constant 4 : index
      %41 = arith.subi %c64, %c4 : index
      %c1_4 = arith.constant 1 : index
      %42 = arith.addi %41, %c1_4 : index
      %c0_5 = arith.constant 0 : index
      affine.for %arg16 = #map(%c0_5) to #map(%42) step 4 {
        %transpose = memref.transpose %alloc (d0, d1) -> (d0, d1) : memref<32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
        %45 = vector.broadcast %cst : f32 to vector<4xf32>
        vector.store %45, %transpose[%arg15, %arg16] : memref<32x64xf32, strided<[64, 1]>>, vector<4xf32>
      }
      %43 = arith.remsi %c64, %c4 : index
      %44 = arith.subi %c64, %43 : index
      affine.for %arg16 = #map(%44) to #map(%c64) {
        %transpose = memref.transpose %alloc (d0, d1) -> (d0, d1) : memref<32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
        memref.store %cst, %transpose[%arg15, %arg16] : memref<32x64xf32, strided<[64, 1]>>
      }
    }
    %0 = arith.addi %arg3, %c31_i32 : i32
    %1 = arith.divsi %0, %c32_i32 : i32
    %2 = arith.addi %arg4, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    %4 = arith.muli %3, %c8_i32 : i32
    %5 = arith.divsi %arg12, %4 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.subi %1, %6 : i32
    %8 = arith.minsi %7, %c8_i32 : i32
    %9 = arith.remsi %arg12, %8 : i32
    %10 = arith.addi %6, %9 : i32
    %11 = arith.remsi %arg12, %4 : i32
    %12 = arith.divsi %11, %8 : i32
    %13 = arith.muli %10, %c32_i32 : i32
    %14 = arith.index_cast %13 : i32 to index
    %15 = arith.muli %12, %c64_i32 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.index_cast %arg3 : i32 to index
    %18 = arith.index_cast %arg6 : i32 to index
    %19 = arith.muli %14, %18 : index
    %20 = arith.muli %17, %18 : index
    %21 = arith.index_cast %arg7 : i32 to index
    %22 = arith.index_cast %arg4 : i32 to index
    %23 = arith.addi %arg5, %c15_i32 : i32
    %24 = arith.divsi %23, %c16_i32 : i32
    %25 = arith.muli %arg7, %c16_i32 : i32
    %26 = arith.index_cast %25 : i32 to index
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
    memref.copy %alloc, %alloc_2 : memref<32x64xf32> to memref<32x64xf32>
    %27:3 = scf.for %arg15 = %c0_i32 to %24 step %c1_i32 iter_args(%arg16 = %19, %arg17 = %c0, %arg18 = %alloc_2) -> (index, index, memref<32x64xf32>)  : i32 {
      %41 = arith.addi %arg17, %16 : index
      %42 = arith.remsi %41, %22 : index
      %43 = arith.subi %41, %42 : index
      %44 = arith.addi %42, %c64 : index
      %45 = arith.minsi %44, %22 : index
      %46 = arith.subi %45, %42 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [%c16, %46], strides: [%21, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %47 = arith.subi %c64, %46 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%c16, %47], strides: [%21, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %48 = arith.remsi %arg16, %18 : index
      %49 = arith.addi %20, %48 : index
      %50 = arith.subi %49, %arg16 : index
      %51 = arith.divsi %50, %18 : index
      %52 = arith.minsi %51, %c32 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%arg16], sizes: [%52, %c16], strides: [%18, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %53 = arith.subi %c32, %52 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%48], sizes: [%53, %c16], strides: [%18, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %54 = arith.muli %arg15, %c16_i32 : i32
      %55 = arith.subi %arg5, %54 : i32
      %56 = arith.index_cast %55 : i32 to index
      %57 = arith.minsi %56, %c16 : index
      %58 = arith.maxsi %57, %c0 : index
      %alloc_8 = memref.alloc() : memref<32x16xf32>
      %59 = arith.cmpi slt, %58, %c16 : index
      scf.if %59 {
        %c0_23 = arith.constant 0 : index
        %c32_24 = arith.constant 32 : index
        affine.for %arg19 = #map(%c0_23) to #map(%c32_24) {
          %c4_25 = arith.constant 4 : index
          %66 = arith.subi %c16, %c4_25 : index
          %c1_26 = arith.constant 1 : index
          %67 = arith.addi %66, %c1_26 : index
          %c0_27 = arith.constant 0 : index
          affine.for %arg20 = #map(%c0_27) to #map(%67) step 4 {
            %transpose = memref.transpose %alloc_8 (d0, d1) -> (d0, d1) : memref<32x16xf32> to memref<32x16xf32, strided<[16, 1]>>
            %70 = vector.broadcast %cst : f32 to vector<4xf32>
            vector.store %70, %transpose[%arg19, %arg20] : memref<32x16xf32, strided<[16, 1]>>, vector<4xf32>
          }
          %68 = arith.remsi %c16, %c4_25 : index
          %69 = arith.subi %c16, %68 : index
          affine.for %arg20 = #map(%69) to #map(%c16) {
            %transpose = memref.transpose %alloc_8 (d0, d1) -> (d0, d1) : memref<32x16xf32> to memref<32x16xf32, strided<[16, 1]>>
            memref.store %cst, %transpose[%arg19, %arg20] : memref<32x16xf32, strided<[16, 1]>>
          }
        }
      }
      %subview_9 = memref.subview %reinterpret_cast_6[0, 0] [%52, %58] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_10 = memref.subview %reinterpret_cast_7[0, 0] [%53, %58] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_11 = memref.subview %alloc_8[0, 0] [%52, %58] [1, 1] : memref<32x16xf32> to memref<?x?xf32, strided<[16, 1]>>
      %subview_12 = memref.subview %alloc_8[%52, 0] [%53, %58] [1, 1] : memref<32x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
      memref.copy %subview_9, %subview_11 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
      memref.copy %subview_10, %subview_12 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[16, 1], offset: ?>>
      %alloc_13 = memref.alloc() : memref<16x64xf32>
      scf.if %59 {
        %c0_23 = arith.constant 0 : index
        %c16_24 = arith.constant 16 : index
        affine.for %arg19 = #map(%c0_23) to #map(%c16_24) {
          %c4_25 = arith.constant 4 : index
          %66 = arith.subi %c64, %c4_25 : index
          %c1_26 = arith.constant 1 : index
          %67 = arith.addi %66, %c1_26 : index
          %c0_27 = arith.constant 0 : index
          affine.for %arg20 = #map(%c0_27) to #map(%67) step 4 {
            %transpose = memref.transpose %alloc_13 (d0, d1) -> (d0, d1) : memref<16x64xf32> to memref<16x64xf32, strided<[64, 1]>>
            %70 = vector.broadcast %cst : f32 to vector<4xf32>
            vector.store %70, %transpose[%arg19, %arg20] : memref<16x64xf32, strided<[64, 1]>>, vector<4xf32>
          }
          %68 = arith.remsi %c64, %c4_25 : index
          %69 = arith.subi %c64, %68 : index
          affine.for %arg20 = #map(%69) to #map(%c64) {
            %transpose = memref.transpose %alloc_13 (d0, d1) -> (d0, d1) : memref<16x64xf32> to memref<16x64xf32, strided<[64, 1]>>
            memref.store %cst, %transpose[%arg19, %arg20] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
      }
      %subview_14 = memref.subview %reinterpret_cast_4[0, 0] [%58, %46] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_15 = memref.subview %reinterpret_cast_5[0, 0] [%58, %47] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_16 = memref.subview %alloc_13[0, 0] [%58, %46] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      %subview_17 = memref.subview %alloc_13[0, %46] [%58, %47] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %subview_14, %subview_16 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_15, %subview_17 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
      memref.copy %alloc, %alloc_18 : memref<32x64xf32> to memref<32x64xf32>
      %c4 = arith.constant 4 : index
      %60 = arith.subi %c64, %c4 : index
      %c1_19 = arith.constant 1 : index
      %61 = arith.addi %60, %c1_19 : index
      %c0_20 = arith.constant 0 : index
      affine.for %arg19 = #map(%c0_20) to #map(%61) step 4 {
        affine.for %arg20 = 0 to 32 {
          %66 = arith.addi %arg19, %c0 : index
          %67 = vector.load %alloc_18[%arg20, %66] : memref<32x64xf32>, vector<4xf32>
          %68 = affine.for %arg21 = 0 to 16 iter_args(%arg22 = %67) -> (vector<4xf32>) {
            %70 = memref.load %alloc_8[%arg20, %arg21] : memref<32x16xf32>
            %71 = vector.broadcast %70 : f32 to vector<4xf32>
            %72 = arith.addi %arg19, %c0 : index
            %73 = vector.load %alloc_13[%arg21, %72] : memref<16x64xf32>, vector<4xf32>
            %74 = vector.fma %71, %73, %arg22 : vector<4xf32>
            affine.yield %74 : vector<4xf32>
          }
          %69 = arith.addi %arg19, %c0 : index
          vector.store %68, %alloc_18[%arg20, %69] : memref<32x64xf32>, vector<4xf32>
        }
      }
      %62 = arith.remsi %c64, %c4 : index
      %63 = arith.subi %c64, %62 : index
      affine.for %arg19 = #map(%63) to #map(%c64) {
        affine.for %arg20 = 0 to 32 {
          %66 = arith.addi %arg19, %c0 : index
          %67 = memref.load %alloc_18[%arg20, %66] : memref<32x64xf32>
          %68 = affine.for %arg21 = 0 to 16 iter_args(%arg22 = %67) -> (f32) {
            %70 = memref.load %alloc_8[%arg20, %arg21] : memref<32x16xf32>
            %71 = arith.addi %arg19, %c0 : index
            %72 = memref.load %alloc_13[%arg21, %71] : memref<16x64xf32>
            %73 = arith.mulf %70, %72 : f32
            %74 = arith.addf %73, %arg22 : f32
            affine.yield %74 : f32
          }
          %69 = arith.addi %arg19, %c0 : index
          memref.store %68, %alloc_18[%arg20, %69] : memref<32x64xf32>
        }
      }
      %c0_21 = arith.constant 0 : index
      %c32_22 = arith.constant 32 : index
      affine.for %arg19 = #map(%c0_21) to #map(%c32_22) {
        %c4_23 = arith.constant 4 : index
        %66 = arith.subi %c64, %c4_23 : index
        %c1_24 = arith.constant 1 : index
        %67 = arith.addi %66, %c1_24 : index
        %c0_25 = arith.constant 0 : index
        affine.for %arg20 = #map(%c0_25) to #map(%67) step 4 {
          %transpose = memref.transpose %alloc_18 (d0, d1) -> (d0, d1) : memref<32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
          %transpose_26 = memref.transpose %arg18 (d0, d1) -> (d0, d1) : memref<32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
          %70 = vector.load %transpose_26[%arg19, %arg20] : memref<32x64xf32, strided<[64, 1]>>, vector<4xf32>
          %71 = vector.load %transpose[%arg19, %arg20] : memref<32x64xf32, strided<[64, 1]>>, vector<4xf32>
          %72 = arith.addf %70, %71 : vector<4xf32>
          vector.store %72, %transpose_26[%arg19, %arg20] : memref<32x64xf32, strided<[64, 1]>>, vector<4xf32>
        }
        %68 = arith.remsi %c64, %c4_23 : index
        %69 = arith.subi %c64, %68 : index
        affine.for %arg20 = #map(%69) to #map(%c64) {
          %transpose = memref.transpose %alloc_18 (d0, d1) -> (d0, d1) : memref<32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
          %transpose_26 = memref.transpose %arg18 (d0, d1) -> (d0, d1) : memref<32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
          %70 = memref.load %transpose_26[%arg19, %arg20] : memref<32x64xf32, strided<[64, 1]>>
          %71 = memref.load %transpose[%arg19, %arg20] : memref<32x64xf32, strided<[64, 1]>>
          %72 = arith.addf %70, %71 : f32
          memref.store %72, %transpose_26[%arg19, %arg20] : memref<32x64xf32, strided<[64, 1]>>
        }
      }
      %64 = arith.addi %arg16, %c16 : index
      %65 = arith.addi %arg17, %26 : index
      scf.yield %64, %65, %arg18 : index, index, memref<32x64xf32>
    }
    %28 = arith.index_cast %arg8 : i32 to index
    %29 = arith.muli %14, %28 : index
    %30 = arith.addi %29, %16 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%30], sizes: [32, 64], strides: [%28, 1] : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
    %31 = arith.addi %14, %c32 : index
    %32 = arith.minsi %31, %17 : index
    %33 = arith.maxsi %32, %14 : index
    %34 = arith.subi %33, %14 : index
    %35 = arith.addi %16, %c64 : index
    %36 = arith.minsi %35, %22 : index
    %37 = arith.maxsi %36, %16 : index
    %38 = arith.subi %37, %16 : index
    %39 = arith.minsi %34, %c32 : index
    %40 = arith.minsi %38, %c64 : index
    %subview = memref.subview %27#2[0, 0] [%39, %40] [1, 1] : memref<32x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    %subview_3 = memref.subview %reinterpret_cast[0, 0] [%39, %40] [1, 1] : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    memref.copy %subview, %subview_3 : memref<?x?xf32, strided<[64, 1]>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    return
  }
  func.func @main() -> i32 {
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<32x16xf32>
    %alloc_0 = memref.alloc() : memref<16x64xf32>
    %alloc_1 = memref.alloc() : memref<32x64xf32>
    %cast = memref.cast %alloc : memref<32x16xf32> to memref<*xf32>
    %cast_2 = memref.cast %alloc_0 : memref<16x64xf32> to memref<*xf32>
    %cast_3 = memref.cast %alloc_1 : memref<32x64xf32> to memref<*xf32>
    call @matmul_kernel(%cast, %cast_2, %cast_3, %c32_i32, %c64_i32, %c16_i32, %c16_i32, %c64_i32, %c64_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    memref.dealloc %alloc : memref<32x16xf32>
    memref.dealloc %alloc_0 : memref<16x64xf32>
    memref.dealloc %alloc_1 : memref<32x64xf32>
    return %c0_i32 : i32
  }
}

