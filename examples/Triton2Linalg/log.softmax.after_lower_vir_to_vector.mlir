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
      %c4_28 = arith.constant 4 : index
      %42 = arith.subi %c256, %c4_28 : index
      %c1_29 = arith.constant 1 : index
      %43 = arith.addi %42, %c1_29 : index
      %c0_30 = arith.constant 0 : index
      affine.for %arg11 = #map(%c0_30) to #map(%43) step 4 {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
        %46 = vector.broadcast %cst : f32 to vector<4xf32>
        vector.store %46, %transpose[%arg11] : memref<256xf32, strided<[1]>>, vector<4xf32>
      }
      %44 = arith.remsi %c256, %c4_28 : index
      %45 = arith.subi %c256, %44 : index
      affine.for %arg11 = #map(%45) to #map(%c256) {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
        memref.store %cst, %transpose[%arg11] : memref<256xf32, strided<[1]>>
      }
    }
    %subview = memref.subview %reinterpret_cast[0] [%4] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_1 = memref.subview %alloc[0] [%4] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_2[] : memref<f32>
    %alloca = memref.alloca() : memref<f32>
    %6 = memref.load %alloc_2[] : memref<f32>
    memref.store %6, %alloca[] : memref<f32>
    %c4 = arith.constant 4 : index
    %7 = arith.subi %c256, %c4 : index
    %c1 = arith.constant 1 : index
    %8 = arith.addi %7, %c1 : index
    %c0_3 = arith.constant 0 : index
    affine.for %arg11 = #map(%c0_3) to #map(%8) step 4 {
      %42 = arith.addi %arg11, %c0 : index
      %43 = vector.load %alloc[%42] : memref<256xf32>, vector<4xf32>
      %44 = memref.load %alloca[] : memref<f32>
      %45 = vector.reduction <maxnumf>, %43, %44 : vector<4xf32> into f32
      memref.store %45, %alloca[] : memref<f32>
    }
    %9 = arith.remsi %c256, %c4 : index
    %10 = arith.subi %c256, %9 : index
    affine.for %arg11 = #map(%10) to #map(%c256) {
      %42 = arith.addi %arg11, %c0 : index
      %43 = memref.load %alloc[%42] : memref<256xf32>
      %44 = memref.load %alloca[] : memref<f32>
      %45 = arith.maxnumf %43, %44 : f32
      memref.store %45, %alloca[] : memref<f32>
    }
    %11 = memref.load %alloca[] : memref<f32>
    memref.store %11, %alloc_2[] : memref<f32>
    %12 = memref.load %alloc_2[] : memref<f32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
    %c4_5 = arith.constant 4 : index
    %13 = arith.subi %c256, %c4_5 : index
    %c1_6 = arith.constant 1 : index
    %14 = arith.addi %13, %c1_6 : index
    %c0_7 = arith.constant 0 : index
    affine.for %arg11 = #map(%c0_7) to #map(%14) step 4 {
      %transpose = memref.transpose %alloc_4 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %42 = vector.broadcast %12 : f32 to vector<4xf32>
      vector.store %42, %transpose[%arg11] : memref<256xf32, strided<[1]>>, vector<4xf32>
    }
    %15 = arith.remsi %c256, %c4_5 : index
    %16 = arith.subi %c256, %15 : index
    affine.for %arg11 = #map(%16) to #map(%c256) {
      %transpose = memref.transpose %alloc_4 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      memref.store %12, %transpose[%arg11] : memref<256xf32, strided<[1]>>
    }
    %c4_8 = arith.constant 4 : index
    %17 = arith.subi %c256, %c4_8 : index
    %c1_9 = arith.constant 1 : index
    %18 = arith.addi %17, %c1_9 : index
    %c0_10 = arith.constant 0 : index
    affine.for %arg11 = #map(%c0_10) to #map(%18) step 4 {
      %transpose = memref.transpose %alloc_4 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %transpose_28 = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %42 = vector.load %transpose_28[%arg11] : memref<256xf32, strided<[1]>>, vector<4xf32>
      %43 = vector.load %transpose[%arg11] : memref<256xf32, strided<[1]>>, vector<4xf32>
      %44 = arith.subf %42, %43 : vector<4xf32>
      vector.store %44, %transpose_28[%arg11] : memref<256xf32, strided<[1]>>, vector<4xf32>
    }
    %19 = arith.remsi %c256, %c4_8 : index
    %20 = arith.subi %c256, %19 : index
    affine.for %arg11 = #map(%20) to #map(%c256) {
      %transpose = memref.transpose %alloc_4 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %transpose_28 = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %42 = memref.load %transpose_28[%arg11] : memref<256xf32, strided<[1]>>
      %43 = memref.load %transpose[%arg11] : memref<256xf32, strided<[1]>>
      %44 = arith.subf %42, %43 : f32
      memref.store %44, %transpose_28[%arg11] : memref<256xf32, strided<[1]>>
    }
    %c4_11 = arith.constant 4 : index
    %21 = arith.subi %c256, %c4_11 : index
    %c1_12 = arith.constant 1 : index
    %22 = arith.addi %21, %c1_12 : index
    %c0_13 = arith.constant 0 : index
    affine.for %arg11 = #map(%c0_13) to #map(%22) step 4 {
      %transpose = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %42 = vector.load %transpose[%arg11] : memref<256xf32, strided<[1]>>, vector<4xf32>
      %43 = math.exp %42 : vector<4xf32>
      vector.store %43, %transpose[%arg11] : memref<256xf32, strided<[1]>>, vector<4xf32>
    }
    %23 = arith.remsi %c256, %c4_11 : index
    %24 = arith.subi %c256, %23 : index
    affine.for %arg11 = #map(%24) to #map(%c256) {
      %transpose = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %42 = memref.load %transpose[%arg11] : memref<256xf32, strided<[1]>>
      %43 = math.exp %42 : f32
      memref.store %43, %transpose[%arg11] : memref<256xf32, strided<[1]>>
    }
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_14[] : memref<f32>
    %alloca_15 = memref.alloca() : memref<f32>
    %25 = memref.load %alloc_14[] : memref<f32>
    memref.store %25, %alloca_15[] : memref<f32>
    %c4_16 = arith.constant 4 : index
    %26 = arith.subi %c256, %c4_16 : index
    %c1_17 = arith.constant 1 : index
    %27 = arith.addi %26, %c1_17 : index
    %c0_18 = arith.constant 0 : index
    affine.for %arg11 = #map(%c0_18) to #map(%27) step 4 {
      %42 = arith.addi %arg11, %c0 : index
      %43 = vector.load %alloc[%42] : memref<256xf32>, vector<4xf32>
      %44 = memref.load %alloca_15[] : memref<f32>
      %45 = vector.reduction <add>, %43, %44 : vector<4xf32> into f32
      memref.store %45, %alloca_15[] : memref<f32>
    }
    %28 = arith.remsi %c256, %c4_16 : index
    %29 = arith.subi %c256, %28 : index
    affine.for %arg11 = #map(%29) to #map(%c256) {
      %42 = arith.addi %arg11, %c0 : index
      %43 = memref.load %alloc[%42] : memref<256xf32>
      %44 = memref.load %alloca_15[] : memref<f32>
      %45 = arith.addf %43, %44 : f32
      memref.store %45, %alloca_15[] : memref<f32>
    }
    %30 = memref.load %alloca_15[] : memref<f32>
    memref.store %30, %alloc_14[] : memref<f32>
    %31 = memref.load %alloc_14[] : memref<f32>
    %c4_19 = arith.constant 4 : index
    %32 = arith.subi %c256, %c4_19 : index
    %c1_20 = arith.constant 1 : index
    %33 = arith.addi %32, %c1_20 : index
    %c0_21 = arith.constant 0 : index
    affine.for %arg11 = #map(%c0_21) to #map(%33) step 4 {
      %transpose = memref.transpose %alloc_4 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %42 = vector.broadcast %31 : f32 to vector<4xf32>
      vector.store %42, %transpose[%arg11] : memref<256xf32, strided<[1]>>, vector<4xf32>
    }
    %34 = arith.remsi %c256, %c4_19 : index
    %35 = arith.subi %c256, %34 : index
    affine.for %arg11 = #map(%35) to #map(%c256) {
      %transpose = memref.transpose %alloc_4 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      memref.store %31, %transpose[%arg11] : memref<256xf32, strided<[1]>>
    }
    %c4_22 = arith.constant 4 : index
    %36 = arith.subi %c256, %c4_22 : index
    %c1_23 = arith.constant 1 : index
    %37 = arith.addi %36, %c1_23 : index
    %c0_24 = arith.constant 0 : index
    affine.for %arg11 = #map(%c0_24) to #map(%37) step 4 {
      %transpose = memref.transpose %alloc_4 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %transpose_28 = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %42 = vector.load %transpose_28[%arg11] : memref<256xf32, strided<[1]>>, vector<4xf32>
      %43 = vector.load %transpose[%arg11] : memref<256xf32, strided<[1]>>, vector<4xf32>
      %44 = arith.divf %42, %43 : vector<4xf32>
      vector.store %44, %transpose_28[%arg11] : memref<256xf32, strided<[1]>>, vector<4xf32>
    }
    %38 = arith.remsi %c256, %c4_22 : index
    %39 = arith.subi %c256, %38 : index
    affine.for %arg11 = #map(%39) to #map(%c256) {
      %transpose = memref.transpose %alloc_4 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %transpose_28 = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %42 = memref.load %transpose_28[%arg11] : memref<256xf32, strided<[1]>>
      %43 = memref.load %transpose[%arg11] : memref<256xf32, strided<[1]>>
      %44 = arith.divf %42, %43 : f32
      memref.store %44, %transpose_28[%arg11] : memref<256xf32, strided<[1]>>
    }
    %40 = arith.muli %arg8, %arg3 : i32
    %41 = arith.index_cast %40 : i32 to index
    %reinterpret_cast_25 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
    %subview_26 = memref.subview %alloc[0] [%4] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    %subview_27 = memref.subview %reinterpret_cast_25[0] [%4] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    memref.copy %subview_26, %subview_27 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1], offset: ?>>
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

