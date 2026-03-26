#map = affine_map<(d0) -> (d0)>
module {
  func.func @_layer_norm_fwd_fused(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: memref<*xf32> {tt.divisibility = 16 : i32}, %arg5: memref<*xf32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c512_i32 = arith.constant 512 : i32
    %c512 = arith.constant 512 : index
    %cst_1 = arith.constant 0.000000e+00 : f16
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    %c4 = arith.constant 4 : index
    %0 = arith.subi %c512, %c4 : index
    %c1_3 = arith.constant 1 : index
    %1 = arith.addi %0, %c1_3 : index
    %c0_4 = arith.constant 0 : index
    affine.for %arg15 = #map(%c0_4) to #map(%1) step 4 {
      %transpose = memref.transpose %alloc_2 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
      %41 = vector.broadcast %cst : f32 to vector<4xf32>
      vector.store %41, %transpose[%arg15] : memref<512xf32, strided<[1]>>, vector<4xf32>
    }
    %2 = arith.remsi %c512, %c4 : index
    %3 = arith.subi %c512, %2 : index
    affine.for %arg15 = #map(%3) to #map(%c512) {
      %transpose = memref.transpose %alloc_2 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
      memref.store %cst, %transpose[%arg15] : memref<512xf32, strided<[1]>>
    }
    %4 = arith.muli %arg12, %arg6 : i32
    %5 = arith.index_cast %4 : i32 to index
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<512xi32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<512xi32>
    scf.for %arg15 = %c0 to %c512 step %c1 {
      %41 = arith.index_cast %arg15 : index to i32
      memref.store %41, %alloc_6[%arg15] : memref<512xi32>
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<512xi32>
    %c4_8 = arith.constant 4 : index
    %6 = arith.subi %c512, %c4_8 : index
    %c1_9 = arith.constant 1 : index
    %7 = arith.addi %6, %c1_9 : index
    %c0_10 = arith.constant 0 : index
    affine.for %arg15 = #map(%c0_10) to #map(%7) step 4 {
      %transpose = memref.transpose %alloc_7 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
      %41 = vector.broadcast %arg7 : i32 to vector<4xi32>
      vector.store %41, %transpose[%arg15] : memref<512xi32, strided<[1]>>, vector<4xi32>
    }
    %8 = arith.remsi %c512, %c4_8 : index
    %9 = arith.subi %c512, %8 : index
    affine.for %arg15 = #map(%9) to #map(%c512) {
      %transpose = memref.transpose %alloc_7 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
      memref.store %arg7, %transpose[%arg15] : memref<512xi32, strided<[1]>>
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    memref.copy %alloc_2, %alloc_11 : memref<512xf32> to memref<512xf32>
    %10 = scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32 iter_args(%arg16 = %alloc_11) -> (memref<512xf32>)  : i32 {
      %41 = arith.index_cast %arg15 : i32 to index
      %42 = arith.addi %5, %41 : index
      %reinterpret_cast_31 = memref.reinterpret_cast %arg0 to offset: [%42], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %43 = arith.addi %41, %c512 : index
      %44 = arith.index_cast %arg7 : i32 to index
      %45 = arith.minsi %43, %44 : index
      %46 = arith.maxsi %45, %41 : index
      %47 = arith.subi %46, %41 : index
      %alloc_32 = memref.alloc() : memref<512xf16>
      %48 = arith.cmpi slt, %47, %c512 : index
      scf.if %48 {
        %c4_40 = arith.constant 4 : index
        %57 = arith.subi %c512, %c4_40 : index
        %c1_41 = arith.constant 1 : index
        %58 = arith.addi %57, %c1_41 : index
        %c0_42 = arith.constant 0 : index
        affine.for %arg17 = #map(%c0_42) to #map(%58) step 4 {
          %transpose = memref.transpose %alloc_32 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
          %61 = vector.broadcast %cst_1 : f16 to vector<4xf16>
          vector.store %61, %transpose[%arg17] : memref<512xf16, strided<[1]>>, vector<4xf16>
        }
        %59 = arith.remsi %c512, %c4_40 : index
        %60 = arith.subi %c512, %59 : index
        affine.for %arg17 = #map(%60) to #map(%c512) {
          %transpose = memref.transpose %alloc_32 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
          memref.store %cst_1, %transpose[%arg17] : memref<512xf16, strided<[1]>>
        }
      }
      %subview = memref.subview %reinterpret_cast_31[0] [%47] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_33 = memref.subview %alloc_32[0] [%47] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_33 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %c4_34 = arith.constant 4 : index
      %49 = arith.subi %c512, %c4_34 : index
      %c1_35 = arith.constant 1 : index
      %50 = arith.addi %49, %c1_35 : index
      %c0_36 = arith.constant 0 : index
      affine.for %arg17 = #map(%c0_36) to #map(%50) step 4 {
        %transpose = memref.transpose %alloc_32 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_40 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %57 = vector.load %transpose[%arg17] : memref<512xf16, strided<[1]>>, vector<4xf16>
        %58 = arith.extf %57 : vector<4xf16> to vector<4xf32>
        vector.store %58, %transpose_40[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %51 = arith.remsi %c512, %c4_34 : index
      %52 = arith.subi %c512, %51 : index
      affine.for %arg17 = #map(%52) to #map(%c512) {
        %transpose = memref.transpose %alloc_32 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_40 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %57 = memref.load %transpose[%arg17] : memref<512xf16, strided<[1]>>
        %58 = arith.extf %57 : f16 to f32
        memref.store %58, %transpose_40[%arg17] : memref<512xf32, strided<[1]>>
      }
      %c4_37 = arith.constant 4 : index
      %53 = arith.subi %c512, %c4_37 : index
      %c1_38 = arith.constant 1 : index
      %54 = arith.addi %53, %c1_38 : index
      %c0_39 = arith.constant 0 : index
      affine.for %arg17 = #map(%c0_39) to #map(%54) step 4 {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_40 = memref.transpose %arg16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %57 = vector.load %transpose_40[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %58 = vector.load %transpose[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %59 = arith.addf %57, %58 : vector<4xf32>
        vector.store %59, %transpose_40[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %55 = arith.remsi %c512, %c4_37 : index
      %56 = arith.subi %c512, %55 : index
      affine.for %arg17 = #map(%56) to #map(%c512) {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_40 = memref.transpose %arg16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %57 = memref.load %transpose_40[%arg17] : memref<512xf32, strided<[1]>>
        %58 = memref.load %transpose[%arg17] : memref<512xf32, strided<[1]>>
        %59 = arith.addf %57, %58 : f32
        memref.store %59, %transpose_40[%arg17] : memref<512xf32, strided<[1]>>
      }
      scf.yield %arg16 : memref<512xf32>
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_12[] : memref<f32>
    %alloca = memref.alloca() : memref<f32>
    %11 = memref.load %alloc_12[] : memref<f32>
    memref.store %11, %alloca[] : memref<f32>
    %c4_13 = arith.constant 4 : index
    %12 = arith.subi %c512, %c4_13 : index
    %c1_14 = arith.constant 1 : index
    %13 = arith.addi %12, %c1_14 : index
    %c0_15 = arith.constant 0 : index
    affine.for %arg15 = #map(%c0_15) to #map(%13) step 4 {
      %41 = arith.addi %arg15, %c0 : index
      %42 = vector.load %10[%41] : memref<512xf32>, vector<4xf32>
      %43 = memref.load %alloca[] : memref<f32>
      %44 = vector.reduction <add>, %42, %43 : vector<4xf32> into f32
      memref.store %44, %alloca[] : memref<f32>
    }
    %14 = arith.remsi %c512, %c4_13 : index
    %15 = arith.subi %c512, %14 : index
    affine.for %arg15 = #map(%15) to #map(%c512) {
      %41 = arith.addi %arg15, %c0 : index
      %42 = memref.load %10[%41] : memref<512xf32>
      %43 = memref.load %alloca[] : memref<f32>
      %44 = arith.addf %42, %43 : f32
      memref.store %44, %alloca[] : memref<f32>
    }
    %16 = memref.load %alloca[] : memref<f32>
    memref.store %16, %alloc_12[] : memref<f32>
    %17 = memref.load %alloc_12[] : memref<f32>
    %18 = arith.sitofp %arg7 : i32 to f32
    %19 = arith.divf %17, %18 : f32
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    %c4_17 = arith.constant 4 : index
    %20 = arith.subi %c512, %c4_17 : index
    %c1_18 = arith.constant 1 : index
    %21 = arith.addi %20, %c1_18 : index
    %c0_19 = arith.constant 0 : index
    affine.for %arg15 = #map(%c0_19) to #map(%21) step 4 {
      %transpose = memref.transpose %alloc_16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
      %41 = vector.broadcast %19 : f32 to vector<4xf32>
      vector.store %41, %transpose[%arg15] : memref<512xf32, strided<[1]>>, vector<4xf32>
    }
    %22 = arith.remsi %c512, %c4_17 : index
    %23 = arith.subi %c512, %22 : index
    affine.for %arg15 = #map(%23) to #map(%c512) {
      %transpose = memref.transpose %alloc_16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
      memref.store %19, %transpose[%arg15] : memref<512xf32, strided<[1]>>
    }
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    memref.copy %alloc_2, %alloc_20 : memref<512xf32> to memref<512xf32>
    %24 = scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32 iter_args(%arg16 = %alloc_20) -> (memref<512xf32>)  : i32 {
      %c4_31 = arith.constant 4 : index
      %41 = arith.subi %c512, %c4_31 : index
      %c1_32 = arith.constant 1 : index
      %42 = arith.addi %41, %c1_32 : index
      %c0_33 = arith.constant 0 : index
      affine.for %arg17 = #map(%c0_33) to #map(%42) step 4 {
        %transpose = memref.transpose %alloc_5 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %81 = vector.broadcast %arg15 : i32 to vector<4xi32>
        vector.store %81, %transpose[%arg17] : memref<512xi32, strided<[1]>>, vector<4xi32>
      }
      %43 = arith.remsi %c512, %c4_31 : index
      %44 = arith.subi %c512, %43 : index
      affine.for %arg17 = #map(%44) to #map(%c512) {
        %transpose = memref.transpose %alloc_5 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        memref.store %arg15, %transpose[%arg17] : memref<512xi32, strided<[1]>>
      }
      %c4_34 = arith.constant 4 : index
      %45 = arith.subi %c512, %c4_34 : index
      %c1_35 = arith.constant 1 : index
      %46 = arith.addi %45, %c1_35 : index
      %c0_36 = arith.constant 0 : index
      affine.for %arg17 = #map(%c0_36) to #map(%46) step 4 {
        %transpose = memref.transpose %alloc_6 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %transpose_59 = memref.transpose %alloc_5 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %81 = vector.load %transpose_59[%arg17] : memref<512xi32, strided<[1]>>, vector<4xi32>
        %82 = vector.load %transpose[%arg17] : memref<512xi32, strided<[1]>>, vector<4xi32>
        %83 = arith.addi %81, %82 : vector<4xi32>
        vector.store %83, %transpose_59[%arg17] : memref<512xi32, strided<[1]>>, vector<4xi32>
      }
      %47 = arith.remsi %c512, %c4_34 : index
      %48 = arith.subi %c512, %47 : index
      affine.for %arg17 = #map(%48) to #map(%c512) {
        %transpose = memref.transpose %alloc_6 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %transpose_59 = memref.transpose %alloc_5 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %81 = memref.load %transpose_59[%arg17] : memref<512xi32, strided<[1]>>
        %82 = memref.load %transpose[%arg17] : memref<512xi32, strided<[1]>>
        %83 = arith.addi %81, %82 : i32
        memref.store %83, %transpose_59[%arg17] : memref<512xi32, strided<[1]>>
      }
      %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<512xi1>
      %c4_38 = arith.constant 4 : index
      %49 = arith.subi %c512, %c4_38 : index
      %c1_39 = arith.constant 1 : index
      %50 = arith.addi %49, %c1_39 : index
      %c0_40 = arith.constant 0 : index
      affine.for %arg17 = #map(%c0_40) to #map(%50) step 4 {
        %transpose = memref.transpose %alloc_5 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %transpose_59 = memref.transpose %alloc_7 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %transpose_60 = memref.transpose %alloc_37 (d0) -> (d0) : memref<512xi1> to memref<512xi1, strided<[1]>>
        %81 = vector.load %transpose[%arg17] : memref<512xi32, strided<[1]>>, vector<4xi32>
        %82 = vector.load %transpose_59[%arg17] : memref<512xi32, strided<[1]>>, vector<4xi32>
        %83 = arith.cmpi slt, %81, %82 : vector<4xi32>
        vector.store %83, %transpose_60[%arg17] : memref<512xi1, strided<[1]>>, vector<4xi1>
      }
      %51 = arith.remsi %c512, %c4_38 : index
      %52 = arith.subi %c512, %51 : index
      affine.for %arg17 = #map(%52) to #map(%c512) {
        %transpose = memref.transpose %alloc_5 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %transpose_59 = memref.transpose %alloc_7 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %transpose_60 = memref.transpose %alloc_37 (d0) -> (d0) : memref<512xi1> to memref<512xi1, strided<[1]>>
        %81 = memref.load %transpose[%arg17] : memref<512xi32, strided<[1]>>
        %82 = memref.load %transpose_59[%arg17] : memref<512xi32, strided<[1]>>
        %83 = arith.cmpi slt, %81, %82 : i32
        memref.store %83, %transpose_60[%arg17] : memref<512xi1, strided<[1]>>
      }
      %53 = arith.index_cast %arg15 : i32 to index
      %54 = arith.addi %5, %53 : index
      %reinterpret_cast_41 = memref.reinterpret_cast %arg0 to offset: [%54], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %55 = arith.addi %53, %c512 : index
      %56 = arith.index_cast %arg7 : i32 to index
      %57 = arith.minsi %55, %56 : index
      %58 = arith.maxsi %57, %53 : index
      %59 = arith.subi %58, %53 : index
      %alloc_42 = memref.alloc() : memref<512xf16>
      %60 = arith.cmpi slt, %59, %c512 : index
      scf.if %60 {
        %c4_59 = arith.constant 4 : index
        %81 = arith.subi %c512, %c4_59 : index
        %c1_60 = arith.constant 1 : index
        %82 = arith.addi %81, %c1_60 : index
        %c0_61 = arith.constant 0 : index
        affine.for %arg17 = #map(%c0_61) to #map(%82) step 4 {
          %transpose = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
          %85 = vector.broadcast %cst_1 : f16 to vector<4xf16>
          vector.store %85, %transpose[%arg17] : memref<512xf16, strided<[1]>>, vector<4xf16>
        }
        %83 = arith.remsi %c512, %c4_59 : index
        %84 = arith.subi %c512, %83 : index
        affine.for %arg17 = #map(%84) to #map(%c512) {
          %transpose = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
          memref.store %cst_1, %transpose[%arg17] : memref<512xf16, strided<[1]>>
        }
      }
      %subview = memref.subview %reinterpret_cast_41[0] [%59] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_43 = memref.subview %alloc_42[0] [%59] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_43 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %c4_44 = arith.constant 4 : index
      %61 = arith.subi %c512, %c4_44 : index
      %c1_45 = arith.constant 1 : index
      %62 = arith.addi %61, %c1_45 : index
      %c0_46 = arith.constant 0 : index
      affine.for %arg17 = #map(%c0_46) to #map(%62) step 4 {
        %transpose = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_59 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose[%arg17] : memref<512xf16, strided<[1]>>, vector<4xf16>
        %82 = arith.extf %81 : vector<4xf16> to vector<4xf32>
        vector.store %82, %transpose_59[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %63 = arith.remsi %c512, %c4_44 : index
      %64 = arith.subi %c512, %63 : index
      affine.for %arg17 = #map(%64) to #map(%c512) {
        %transpose = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_59 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose[%arg17] : memref<512xf16, strided<[1]>>
        %82 = arith.extf %81 : f16 to f32
        memref.store %82, %transpose_59[%arg17] : memref<512xf32, strided<[1]>>
      }
      %c4_47 = arith.constant 4 : index
      %65 = arith.subi %c512, %c4_47 : index
      %c1_48 = arith.constant 1 : index
      %66 = arith.addi %65, %c1_48 : index
      %c0_49 = arith.constant 0 : index
      affine.for %arg17 = #map(%c0_49) to #map(%66) step 4 {
        %transpose = memref.transpose %alloc_16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_59 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose_59[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %82 = vector.load %transpose[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %83 = arith.subf %81, %82 : vector<4xf32>
        vector.store %83, %transpose_59[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %67 = arith.remsi %c512, %c4_47 : index
      %68 = arith.subi %c512, %67 : index
      affine.for %arg17 = #map(%68) to #map(%c512) {
        %transpose = memref.transpose %alloc_16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_59 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose_59[%arg17] : memref<512xf32, strided<[1]>>
        %82 = memref.load %transpose[%arg17] : memref<512xf32, strided<[1]>>
        %83 = arith.subf %81, %82 : f32
        memref.store %83, %transpose_59[%arg17] : memref<512xf32, strided<[1]>>
      }
      %c4_50 = arith.constant 4 : index
      %69 = arith.subi %c512, %c4_50 : index
      %c1_51 = arith.constant 1 : index
      %70 = arith.addi %69, %c1_51 : index
      %c0_52 = arith.constant 0 : index
      affine.for %arg17 = #map(%c0_52) to #map(%70) step 4 {
        %transpose = memref.transpose %alloc_37 (d0) -> (d0) : memref<512xi1> to memref<512xi1, strided<[1]>>
        %transpose_59 = memref.transpose %alloc_2 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_60 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose[%arg17] : memref<512xi1, strided<[1]>>, vector<4xi1>
        %82 = vector.load %transpose_60[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %83 = vector.load %transpose_59[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %84 = llvm.select %81, %82, %83 : vector<4xi1>, vector<4xf32>
        vector.store %84, %transpose_60[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %71 = arith.remsi %c512, %c4_50 : index
      %72 = arith.subi %c512, %71 : index
      affine.for %arg17 = #map(%72) to #map(%c512) {
        %transpose = memref.transpose %alloc_37 (d0) -> (d0) : memref<512xi1> to memref<512xi1, strided<[1]>>
        %transpose_59 = memref.transpose %alloc_2 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_60 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose[%arg17] : memref<512xi1, strided<[1]>>
        %82 = memref.load %transpose_60[%arg17] : memref<512xf32, strided<[1]>>
        %83 = memref.load %transpose_59[%arg17] : memref<512xf32, strided<[1]>>
        %84 = arith.select %81, %82, %83 : f32
        memref.store %84, %transpose_60[%arg17] : memref<512xf32, strided<[1]>>
      }
      %c4_53 = arith.constant 4 : index
      %73 = arith.subi %c512, %c4_53 : index
      %c1_54 = arith.constant 1 : index
      %74 = arith.addi %73, %c1_54 : index
      %c0_55 = arith.constant 0 : index
      affine.for %arg17 = #map(%c0_55) to #map(%74) step 4 {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %82 = vector.load %transpose[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %83 = arith.mulf %81, %82 : vector<4xf32>
        vector.store %83, %transpose[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %75 = arith.remsi %c512, %c4_53 : index
      %76 = arith.subi %c512, %75 : index
      affine.for %arg17 = #map(%76) to #map(%c512) {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose[%arg17] : memref<512xf32, strided<[1]>>
        %82 = memref.load %transpose[%arg17] : memref<512xf32, strided<[1]>>
        %83 = arith.mulf %81, %82 : f32
        memref.store %83, %transpose[%arg17] : memref<512xf32, strided<[1]>>
      }
      %c4_56 = arith.constant 4 : index
      %77 = arith.subi %c512, %c4_56 : index
      %c1_57 = arith.constant 1 : index
      %78 = arith.addi %77, %c1_57 : index
      %c0_58 = arith.constant 0 : index
      affine.for %arg17 = #map(%c0_58) to #map(%78) step 4 {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_59 = memref.transpose %arg16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose_59[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %82 = vector.load %transpose[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %83 = arith.addf %81, %82 : vector<4xf32>
        vector.store %83, %transpose_59[%arg17] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %79 = arith.remsi %c512, %c4_56 : index
      %80 = arith.subi %c512, %79 : index
      affine.for %arg17 = #map(%80) to #map(%c512) {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_59 = memref.transpose %arg16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose_59[%arg17] : memref<512xf32, strided<[1]>>
        %82 = memref.load %transpose[%arg17] : memref<512xf32, strided<[1]>>
        %83 = arith.addf %81, %82 : f32
        memref.store %83, %transpose_59[%arg17] : memref<512xf32, strided<[1]>>
      }
      scf.yield %arg16 : memref<512xf32>
    }
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_21[] : memref<f32>
    %alloca_22 = memref.alloca() : memref<f32>
    %25 = memref.load %alloc_21[] : memref<f32>
    memref.store %25, %alloca_22[] : memref<f32>
    %c4_23 = arith.constant 4 : index
    %26 = arith.subi %c512, %c4_23 : index
    %c1_24 = arith.constant 1 : index
    %27 = arith.addi %26, %c1_24 : index
    %c0_25 = arith.constant 0 : index
    affine.for %arg15 = #map(%c0_25) to #map(%27) step 4 {
      %41 = arith.addi %arg15, %c0 : index
      %42 = vector.load %24[%41] : memref<512xf32>, vector<4xf32>
      %43 = memref.load %alloca_22[] : memref<f32>
      %44 = vector.reduction <add>, %42, %43 : vector<4xf32> into f32
      memref.store %44, %alloca_22[] : memref<f32>
    }
    %28 = arith.remsi %c512, %c4_23 : index
    %29 = arith.subi %c512, %28 : index
    affine.for %arg15 = #map(%29) to #map(%c512) {
      %41 = arith.addi %arg15, %c0 : index
      %42 = memref.load %24[%41] : memref<512xf32>
      %43 = memref.load %alloca_22[] : memref<f32>
      %44 = arith.addf %42, %43 : f32
      memref.store %44, %alloca_22[] : memref<f32>
    }
    %30 = memref.load %alloca_22[] : memref<f32>
    memref.store %30, %alloc_21[] : memref<f32>
    %31 = memref.load %alloc_21[] : memref<f32>
    %32 = arith.divf %31, %18 : f32
    %33 = arith.addf %32, %arg8 : f32
    %34 = math.sqrt %33 : f32
    %35 = arith.divf %cst_0, %34 : f32
    %36 = arith.index_cast %arg12 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%36], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %19, %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
    %reinterpret_cast_26 = memref.reinterpret_cast %arg5 to offset: [%36], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %35, %reinterpret_cast_26[0] : memref<1xf32, strided<[1], offset: ?>>
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    %c4_28 = arith.constant 4 : index
    %37 = arith.subi %c512, %c4_28 : index
    %c1_29 = arith.constant 1 : index
    %38 = arith.addi %37, %c1_29 : index
    %c0_30 = arith.constant 0 : index
    affine.for %arg15 = #map(%c0_30) to #map(%38) step 4 {
      %transpose = memref.transpose %alloc_27 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
      %41 = vector.broadcast %35 : f32 to vector<4xf32>
      vector.store %41, %transpose[%arg15] : memref<512xf32, strided<[1]>>, vector<4xf32>
    }
    %39 = arith.remsi %c512, %c4_28 : index
    %40 = arith.subi %c512, %39 : index
    affine.for %arg15 = #map(%40) to #map(%c512) {
      %transpose = memref.transpose %alloc_27 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
      memref.store %35, %transpose[%arg15] : memref<512xf32, strided<[1]>>
    }
    scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32  : i32 {
      %41 = arith.index_cast %arg15 : i32 to index
      %reinterpret_cast_31 = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %42 = arith.addi %41, %c512 : index
      %43 = arith.index_cast %arg7 : i32 to index
      %44 = arith.minsi %42, %43 : index
      %45 = arith.maxsi %44, %41 : index
      %46 = arith.subi %45, %41 : index
      %alloc_32 = memref.alloc() : memref<512xf16>
      %subview = memref.subview %reinterpret_cast_31[0] [%46] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_33 = memref.subview %alloc_32[0] [%46] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_33 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %reinterpret_cast_34 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %alloc_35 = memref.alloc() : memref<512xf16>
      %subview_36 = memref.subview %reinterpret_cast_34[0] [%46] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_37 = memref.subview %alloc_35[0] [%46] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview_36, %subview_37 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %47 = arith.addi %5, %41 : index
      %reinterpret_cast_38 = memref.reinterpret_cast %arg0 to offset: [%47], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %alloc_39 = memref.alloc() : memref<512xf16>
      %48 = arith.cmpi slt, %46, %c512 : index
      scf.if %48 {
        %c4_71 = arith.constant 4 : index
        %81 = arith.subi %c512, %c4_71 : index
        %c1_72 = arith.constant 1 : index
        %82 = arith.addi %81, %c1_72 : index
        %c0_73 = arith.constant 0 : index
        affine.for %arg16 = #map(%c0_73) to #map(%82) step 4 {
          %transpose = memref.transpose %alloc_39 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
          %85 = vector.broadcast %cst_1 : f16 to vector<4xf16>
          vector.store %85, %transpose[%arg16] : memref<512xf16, strided<[1]>>, vector<4xf16>
        }
        %83 = arith.remsi %c512, %c4_71 : index
        %84 = arith.subi %c512, %83 : index
        affine.for %arg16 = #map(%84) to #map(%c512) {
          %transpose = memref.transpose %alloc_39 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
          memref.store %cst_1, %transpose[%arg16] : memref<512xf16, strided<[1]>>
        }
      }
      %subview_40 = memref.subview %reinterpret_cast_38[0] [%46] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_41 = memref.subview %alloc_39[0] [%46] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview_40, %subview_41 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %alloc_42 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
      %c4_43 = arith.constant 4 : index
      %49 = arith.subi %c512, %c4_43 : index
      %c1_44 = arith.constant 1 : index
      %50 = arith.addi %49, %c1_44 : index
      %c0_45 = arith.constant 0 : index
      affine.for %arg16 = #map(%c0_45) to #map(%50) step 4 {
        %transpose = memref.transpose %alloc_39 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose[%arg16] : memref<512xf16, strided<[1]>>, vector<4xf16>
        %82 = arith.extf %81 : vector<4xf16> to vector<4xf32>
        vector.store %82, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %51 = arith.remsi %c512, %c4_43 : index
      %52 = arith.subi %c512, %51 : index
      affine.for %arg16 = #map(%52) to #map(%c512) {
        %transpose = memref.transpose %alloc_39 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose[%arg16] : memref<512xf16, strided<[1]>>
        %82 = arith.extf %81 : f16 to f32
        memref.store %82, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>
      }
      %c4_46 = arith.constant 4 : index
      %53 = arith.subi %c512, %c4_46 : index
      %c1_47 = arith.constant 1 : index
      %54 = arith.addi %53, %c1_47 : index
      %c0_48 = arith.constant 0 : index
      affine.for %arg16 = #map(%c0_48) to #map(%54) step 4 {
        %transpose = memref.transpose %alloc_16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose_71[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %82 = vector.load %transpose[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %83 = arith.subf %81, %82 : vector<4xf32>
        vector.store %83, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %55 = arith.remsi %c512, %c4_46 : index
      %56 = arith.subi %c512, %55 : index
      affine.for %arg16 = #map(%56) to #map(%c512) {
        %transpose = memref.transpose %alloc_16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose_71[%arg16] : memref<512xf32, strided<[1]>>
        %82 = memref.load %transpose[%arg16] : memref<512xf32, strided<[1]>>
        %83 = arith.subf %81, %82 : f32
        memref.store %83, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>
      }
      %c4_49 = arith.constant 4 : index
      %57 = arith.subi %c512, %c4_49 : index
      %c1_50 = arith.constant 1 : index
      %58 = arith.addi %57, %c1_50 : index
      %c0_51 = arith.constant 0 : index
      affine.for %arg16 = #map(%c0_51) to #map(%58) step 4 {
        %transpose = memref.transpose %alloc_27 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose_71[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %82 = vector.load %transpose[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %83 = arith.mulf %81, %82 : vector<4xf32>
        vector.store %83, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %59 = arith.remsi %c512, %c4_49 : index
      %60 = arith.subi %c512, %59 : index
      affine.for %arg16 = #map(%60) to #map(%c512) {
        %transpose = memref.transpose %alloc_27 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose_71[%arg16] : memref<512xf32, strided<[1]>>
        %82 = memref.load %transpose[%arg16] : memref<512xf32, strided<[1]>>
        %83 = arith.mulf %81, %82 : f32
        memref.store %83, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>
      }
      %c4_52 = arith.constant 4 : index
      %61 = arith.subi %c512, %c4_52 : index
      %c1_53 = arith.constant 1 : index
      %62 = arith.addi %61, %c1_53 : index
      %c0_54 = arith.constant 0 : index
      affine.for %arg16 = #map(%c0_54) to #map(%62) step 4 {
        %transpose = memref.transpose %alloc_32 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_71 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose[%arg16] : memref<512xf16, strided<[1]>>, vector<4xf16>
        %82 = arith.extf %81 : vector<4xf16> to vector<4xf32>
        vector.store %82, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %63 = arith.remsi %c512, %c4_52 : index
      %64 = arith.subi %c512, %63 : index
      affine.for %arg16 = #map(%64) to #map(%c512) {
        %transpose = memref.transpose %alloc_32 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_71 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose[%arg16] : memref<512xf16, strided<[1]>>
        %82 = arith.extf %81 : f16 to f32
        memref.store %82, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>
      }
      %c4_55 = arith.constant 4 : index
      %65 = arith.subi %c512, %c4_55 : index
      %c1_56 = arith.constant 1 : index
      %66 = arith.addi %65, %c1_56 : index
      %c0_57 = arith.constant 0 : index
      affine.for %arg16 = #map(%c0_57) to #map(%66) step 4 {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose_71[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %82 = vector.load %transpose[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %83 = arith.mulf %81, %82 : vector<4xf32>
        vector.store %83, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %67 = arith.remsi %c512, %c4_55 : index
      %68 = arith.subi %c512, %67 : index
      affine.for %arg16 = #map(%68) to #map(%c512) {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose_71[%arg16] : memref<512xf32, strided<[1]>>
        %82 = memref.load %transpose[%arg16] : memref<512xf32, strided<[1]>>
        %83 = arith.mulf %81, %82 : f32
        memref.store %83, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>
      }
      %c4_58 = arith.constant 4 : index
      %69 = arith.subi %c512, %c4_58 : index
      %c1_59 = arith.constant 1 : index
      %70 = arith.addi %69, %c1_59 : index
      %c0_60 = arith.constant 0 : index
      affine.for %arg16 = #map(%c0_60) to #map(%70) step 4 {
        %transpose = memref.transpose %alloc_35 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_71 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose[%arg16] : memref<512xf16, strided<[1]>>, vector<4xf16>
        %82 = arith.extf %81 : vector<4xf16> to vector<4xf32>
        vector.store %82, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %71 = arith.remsi %c512, %c4_58 : index
      %72 = arith.subi %c512, %71 : index
      affine.for %arg16 = #map(%72) to #map(%c512) {
        %transpose = memref.transpose %alloc_35 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_71 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose[%arg16] : memref<512xf16, strided<[1]>>
        %82 = arith.extf %81 : f16 to f32
        memref.store %82, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>
      }
      %c4_61 = arith.constant 4 : index
      %73 = arith.subi %c512, %c4_61 : index
      %c1_62 = arith.constant 1 : index
      %74 = arith.addi %73, %c1_62 : index
      %c0_63 = arith.constant 0 : index
      affine.for %arg16 = #map(%c0_63) to #map(%74) step 4 {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = vector.load %transpose_71[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %82 = vector.load %transpose[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %83 = arith.addf %81, %82 : vector<4xf32>
        vector.store %83, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
      }
      %75 = arith.remsi %c512, %c4_61 : index
      %76 = arith.subi %c512, %75 : index
      affine.for %arg16 = #map(%76) to #map(%c512) {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %81 = memref.load %transpose_71[%arg16] : memref<512xf32, strided<[1]>>
        %82 = memref.load %transpose[%arg16] : memref<512xf32, strided<[1]>>
        %83 = arith.addf %81, %82 : f32
        memref.store %83, %transpose_71[%arg16] : memref<512xf32, strided<[1]>>
      }
      %reinterpret_cast_64 = memref.reinterpret_cast %arg1 to offset: [%47], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %alloc_65 = memref.alloc() {alignment = 64 : i64} : memref<512xf16>
      %c4_66 = arith.constant 4 : index
      %77 = arith.subi %c512, %c4_66 : index
      %c1_67 = arith.constant 1 : index
      %78 = arith.addi %77, %c1_67 : index
      %c0_68 = arith.constant 0 : index
      affine.for %arg16 = #map(%c0_68) to #map(%78) step 4 {
        %transpose = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_65 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %81 = vector.load %transpose[%arg16] : memref<512xf32, strided<[1]>>, vector<4xf32>
        %82 = arith.truncf %81 : vector<4xf32> to vector<4xf16>
        vector.store %82, %transpose_71[%arg16] : memref<512xf16, strided<[1]>>, vector<4xf16>
      }
      %79 = arith.remsi %c512, %c4_66 : index
      %80 = arith.subi %c512, %79 : index
      affine.for %arg16 = #map(%80) to #map(%c512) {
        %transpose = memref.transpose %alloc_42 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_71 = memref.transpose %alloc_65 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %81 = memref.load %transpose[%arg16] : memref<512xf32, strided<[1]>>
        %82 = arith.truncf %81 : f32 to f16
        memref.store %82, %transpose_71[%arg16] : memref<512xf16, strided<[1]>>
      }
      %subview_69 = memref.subview %alloc_65[0] [%46] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      %subview_70 = memref.subview %reinterpret_cast_64[0] [%46] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      memref.copy %subview_69, %subview_70 : memref<?xf16, strided<[1]>> to memref<?xf16, strided<[1], offset: ?>>
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

