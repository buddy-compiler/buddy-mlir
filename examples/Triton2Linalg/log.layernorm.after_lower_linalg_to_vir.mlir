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
    vir.set_vl %c512 : index {
      %transpose = memref.transpose %alloc_2 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
      %17 = vir.broadcast %cst : f32 -> !vir.vec<512xf32>
      vir.store %17, %transpose[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
      vector.yield
    }
    %0 = arith.muli %arg12, %arg6 : i32
    %1 = arith.index_cast %0 : i32 to index
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<512xi32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<512xi32>
    scf.for %arg15 = %c0 to %c512 step %c1 {
      %17 = arith.index_cast %arg15 : index to i32
      memref.store %17, %alloc_4[%arg15] : memref<512xi32>
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<512xi32>
    vir.set_vl %c512 : index {
      %transpose = memref.transpose %alloc_5 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
      %17 = vir.broadcast %arg7 : i32 -> !vir.vec<512xi32>
      vir.store %17, %transpose[] : !vir.vec<512xi32> -> memref<512xi32, strided<[1]>>
      vector.yield
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    memref.copy %alloc_2, %alloc_6 : memref<512xf32> to memref<512xf32>
    %2 = scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32 iter_args(%arg16 = %alloc_6) -> (memref<512xf32>)  : i32 {
      %17 = arith.index_cast %arg15 : i32 to index
      %18 = arith.addi %1, %17 : index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %19 = arith.addi %17, %c512 : index
      %20 = arith.index_cast %arg7 : i32 to index
      %21 = arith.minsi %19, %20 : index
      %22 = arith.maxsi %21, %17 : index
      %23 = arith.subi %22, %17 : index
      %alloc_15 = memref.alloc() : memref<512xf16>
      %24 = arith.cmpi slt, %23, %c512 : index
      scf.if %24 {
        vir.set_vl %c512 : index {
          %transpose = memref.transpose %alloc_15 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
          %25 = vir.broadcast %cst_1 : f16 -> !vir.vec<512xf16>
          vir.store %25, %transpose[] : !vir.vec<512xf16> -> memref<512xf16, strided<[1]>>
          vector.yield
        }
      }
      %subview = memref.subview %reinterpret_cast_14[0] [%23] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_16 = memref.subview %alloc_15[0] [%23] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_16 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_15 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_17 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose[] : memref<512xf16, strided<[1]>> -> !vir.vec<512xf16>
        %26 = vir.extf %25 : !vir.vec<512xf16> -> !vir.vec<512xf32>
        vir.store %26, %transpose_17[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_17 = memref.transpose %arg16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose_17[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %26 = vir.load %transpose[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %27 = arith.addf %25, %26 : !vir.vec<512xf32>
        vir.store %27, %transpose_17[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      scf.yield %arg16 : memref<512xf32>
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_7[] : memref<f32>
    %alloca = memref.alloca() : memref<f32>
    %3 = memref.load %alloc_7[] : memref<f32>
    memref.store %3, %alloca[] : memref<f32>
    vir.set_vl %c512 : index {
      %17 = vir.load %2[%c0] : memref<512xf32> -> !vir.vec<?xf32>
      %18 = memref.load %alloca[] : memref<f32>
      %19 = vir.reduce %17, %18 {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
      memref.store %19, %alloca[] : memref<f32>
      vector.yield
    }
    %4 = memref.load %alloca[] : memref<f32>
    memref.store %4, %alloc_7[] : memref<f32>
    %5 = memref.load %alloc_7[] : memref<f32>
    %6 = arith.sitofp %arg7 : i32 to f32
    %7 = arith.divf %5, %6 : f32
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    vir.set_vl %c512 : index {
      %transpose = memref.transpose %alloc_8 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
      %17 = vir.broadcast %7 : f32 -> !vir.vec<512xf32>
      vir.store %17, %transpose[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
      vector.yield
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    memref.copy %alloc_2, %alloc_9 : memref<512xf32> to memref<512xf32>
    %8 = scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32 iter_args(%arg16 = %alloc_9) -> (memref<512xf32>)  : i32 {
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_3 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %25 = vir.broadcast %arg15 : i32 -> !vir.vec<512xi32>
        vir.store %25, %transpose[] : !vir.vec<512xi32> -> memref<512xi32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_4 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %transpose_18 = memref.transpose %alloc_3 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %25 = vir.load %transpose_18[] : memref<512xi32, strided<[1]>> -> !vir.vec<512xi32>
        %26 = vir.load %transpose[] : memref<512xi32, strided<[1]>> -> !vir.vec<512xi32>
        %27 = arith.addi %25, %26 : !vir.vec<512xi32>
        vir.store %27, %transpose_18[] : !vir.vec<512xi32> -> memref<512xi32, strided<[1]>>
        vector.yield
      }
      %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<512xi1>
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_3 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %transpose_18 = memref.transpose %alloc_5 (d0) -> (d0) : memref<512xi32> to memref<512xi32, strided<[1]>>
        %transpose_19 = memref.transpose %alloc_14 (d0) -> (d0) : memref<512xi1> to memref<512xi1, strided<[1]>>
        %25 = vir.load %transpose[] : memref<512xi32, strided<[1]>> -> !vir.vec<512xi32>
        %26 = vir.load %transpose_18[] : memref<512xi32, strided<[1]>> -> !vir.vec<512xi32>
        %27 = arith.cmpi slt, %25, %26 : !vir.vec<512xi32>
        vir.store %27, %transpose_19[] : !vir.vec<512xi1> -> memref<512xi1, strided<[1]>>
        vector.yield
      }
      %17 = arith.index_cast %arg15 : i32 to index
      %18 = arith.addi %1, %17 : index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %19 = arith.addi %17, %c512 : index
      %20 = arith.index_cast %arg7 : i32 to index
      %21 = arith.minsi %19, %20 : index
      %22 = arith.maxsi %21, %17 : index
      %23 = arith.subi %22, %17 : index
      %alloc_16 = memref.alloc() : memref<512xf16>
      %24 = arith.cmpi slt, %23, %c512 : index
      scf.if %24 {
        vir.set_vl %c512 : index {
          %transpose = memref.transpose %alloc_16 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
          %25 = vir.broadcast %cst_1 : f16 -> !vir.vec<512xf16>
          vir.store %25, %transpose[] : !vir.vec<512xf16> -> memref<512xf16, strided<[1]>>
          vector.yield
        }
      }
      %subview = memref.subview %reinterpret_cast_15[0] [%23] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_17 = memref.subview %alloc_16[0] [%23] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_17 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_16 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_18 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose[] : memref<512xf16, strided<[1]>> -> !vir.vec<512xf16>
        %26 = vir.extf %25 : !vir.vec<512xf16> -> !vir.vec<512xf32>
        vir.store %26, %transpose_18[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_8 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_18 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose_18[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %26 = vir.load %transpose[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %27 = arith.subf %25, %26 : !vir.vec<512xf32>
        vir.store %27, %transpose_18[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_14 (d0) -> (d0) : memref<512xi1> to memref<512xi1, strided<[1]>>
        %transpose_18 = memref.transpose %alloc_2 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_19 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose[] : memref<512xi1, strided<[1]>> -> !vir.vec<512xi1>
        %26 = vir.load %transpose_19[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %27 = vir.load %transpose_18[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %28 = vir.select %25, %26, %27 : !vir.vec<512xi1>, !vir.vec<512xf32>
        vir.store %28, %transpose_19[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %26 = vir.load %transpose[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %27 = arith.mulf %25, %26 : !vir.vec<512xf32>
        vir.store %27, %transpose[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_18 = memref.transpose %arg16 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose_18[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %26 = vir.load %transpose[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %27 = arith.addf %25, %26 : !vir.vec<512xf32>
        vir.store %27, %transpose_18[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      scf.yield %arg16 : memref<512xf32>
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_10[] : memref<f32>
    %alloca_11 = memref.alloca() : memref<f32>
    %9 = memref.load %alloc_10[] : memref<f32>
    memref.store %9, %alloca_11[] : memref<f32>
    vir.set_vl %c512 : index {
      %17 = vir.load %8[%c0] : memref<512xf32> -> !vir.vec<?xf32>
      %18 = memref.load %alloca_11[] : memref<f32>
      %19 = vir.reduce %17, %18 {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
      memref.store %19, %alloca_11[] : memref<f32>
      vector.yield
    }
    %10 = memref.load %alloca_11[] : memref<f32>
    memref.store %10, %alloc_10[] : memref<f32>
    %11 = memref.load %alloc_10[] : memref<f32>
    %12 = arith.divf %11, %6 : f32
    %13 = arith.addf %12, %arg8 : f32
    %14 = math.sqrt %13 : f32
    %15 = arith.divf %cst_0, %14 : f32
    %16 = arith.index_cast %arg12 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%16], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %7, %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg5 to offset: [%16], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %15, %reinterpret_cast_12[0] : memref<1xf32, strided<[1], offset: ?>>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
    vir.set_vl %c512 : index {
      %transpose = memref.transpose %alloc_13 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
      %17 = vir.broadcast %15 : f32 -> !vir.vec<512xf32>
      vir.store %17, %transpose[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
      vector.yield
    }
    scf.for %arg15 = %c0_i32 to %arg7 step %c512_i32  : i32 {
      %17 = arith.index_cast %arg15 : i32 to index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg2 to offset: [%17], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %18 = arith.addi %17, %c512 : index
      %19 = arith.index_cast %arg7 : i32 to index
      %20 = arith.minsi %18, %19 : index
      %21 = arith.maxsi %20, %17 : index
      %22 = arith.subi %21, %17 : index
      %alloc_15 = memref.alloc() : memref<512xf16>
      %subview = memref.subview %reinterpret_cast_14[0] [%22] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_16 = memref.subview %alloc_15[0] [%22] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_16 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %reinterpret_cast_17 = memref.reinterpret_cast %arg3 to offset: [%17], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %alloc_18 = memref.alloc() : memref<512xf16>
      %subview_19 = memref.subview %reinterpret_cast_17[0] [%22] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_20 = memref.subview %alloc_18[0] [%22] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview_19, %subview_20 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %23 = arith.addi %1, %17 : index
      %reinterpret_cast_21 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %alloc_22 = memref.alloc() : memref<512xf16>
      %24 = arith.cmpi slt, %22, %c512 : index
      scf.if %24 {
        vir.set_vl %c512 : index {
          %transpose = memref.transpose %alloc_22 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
          %25 = vir.broadcast %cst_1 : f16 -> !vir.vec<512xf16>
          vir.store %25, %transpose[] : !vir.vec<512xf16> -> memref<512xf16, strided<[1]>>
          vector.yield
        }
      }
      %subview_23 = memref.subview %reinterpret_cast_21[0] [%22] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_24 = memref.subview %alloc_22[0] [%22] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview_23, %subview_24 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_22 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_30 = memref.transpose %alloc_25 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose[] : memref<512xf16, strided<[1]>> -> !vir.vec<512xf16>
        %26 = vir.extf %25 : !vir.vec<512xf16> -> !vir.vec<512xf32>
        vir.store %26, %transpose_30[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_8 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_30 = memref.transpose %alloc_25 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose_30[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %26 = vir.load %transpose[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %27 = arith.subf %25, %26 : !vir.vec<512xf32>
        vir.store %27, %transpose_30[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_13 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_30 = memref.transpose %alloc_25 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose_30[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %26 = vir.load %transpose[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %27 = arith.mulf %25, %26 : !vir.vec<512xf32>
        vir.store %27, %transpose_30[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_15 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_30 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose[] : memref<512xf16, strided<[1]>> -> !vir.vec<512xf16>
        %26 = vir.extf %25 : !vir.vec<512xf16> -> !vir.vec<512xf32>
        vir.store %26, %transpose_30[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_30 = memref.transpose %alloc_25 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose_30[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %26 = vir.load %transpose[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %27 = arith.mulf %25, %26 : !vir.vec<512xf32>
        vir.store %27, %transpose_30[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_18 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %transpose_30 = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose[] : memref<512xf16, strided<[1]>> -> !vir.vec<512xf16>
        %26 = vir.extf %25 : !vir.vec<512xf16> -> !vir.vec<512xf32>
        vir.store %26, %transpose_30[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_30 = memref.transpose %alloc_25 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %25 = vir.load %transpose_30[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %26 = vir.load %transpose[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %27 = arith.addf %25, %26 : !vir.vec<512xf32>
        vir.store %27, %transpose_30[] : !vir.vec<512xf32> -> memref<512xf32, strided<[1]>>
        vector.yield
      }
      %reinterpret_cast_26 = memref.reinterpret_cast %arg1 to offset: [%23], sizes: [512], strides: [1] : memref<*xf16> to memref<512xf16, strided<[1], offset: ?>>
      %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<512xf16>
      vir.set_vl %c512 : index {
        %transpose = memref.transpose %alloc_25 (d0) -> (d0) : memref<512xf32> to memref<512xf32, strided<[1]>>
        %transpose_30 = memref.transpose %alloc_27 (d0) -> (d0) : memref<512xf16> to memref<512xf16, strided<[1]>>
        %25 = vir.load %transpose[] : memref<512xf32, strided<[1]>> -> !vir.vec<512xf32>
        %26 = vir.truncf %25 : !vir.vec<512xf32> -> !vir.vec<512xf16>
        vir.store %26, %transpose_30[] : !vir.vec<512xf16> -> memref<512xf16, strided<[1]>>
        vector.yield
      }
      %subview_28 = memref.subview %alloc_27[0] [%22] [1] : memref<512xf16> to memref<?xf16, strided<[1]>>
      %subview_29 = memref.subview %reinterpret_cast_26[0] [%22] [1] : memref<512xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      memref.copy %subview_28, %subview_29 : memref<?xf16, strided<[1]>> to memref<?xf16, strided<[1], offset: ?>>
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

