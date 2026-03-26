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
      vir.set_vl %c256 : index {
        %transpose = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
        %14 = vir.broadcast %cst : f32 -> !vir.vec<256xf32>
        vir.store %14, %transpose[] : !vir.vec<256xf32> -> memref<256xf32, strided<[1]>>
        vector.yield
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
    vir.set_vl %c256 : index {
      %14 = vir.load %alloc[%c0] : memref<256xf32> -> !vir.vec<?xf32>
      %15 = memref.load %alloca[] : memref<f32>
      %16 = vir.reduce %14, %15 {kind = "maxnum"} : !vir.vec<?xf32>, f32 -> f32
      memref.store %16, %alloca[] : memref<f32>
      vector.yield
    }
    %7 = memref.load %alloca[] : memref<f32>
    memref.store %7, %alloc_2[] : memref<f32>
    %8 = memref.load %alloc_2[] : memref<f32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
    vir.set_vl %c256 : index {
      %transpose = memref.transpose %alloc_3 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %14 = vir.broadcast %8 : f32 -> !vir.vec<256xf32>
      vir.store %14, %transpose[] : !vir.vec<256xf32> -> memref<256xf32, strided<[1]>>
      vector.yield
    }
    vir.set_vl %c256 : index {
      %transpose = memref.transpose %alloc_3 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %transpose_9 = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %14 = vir.load %transpose_9[] : memref<256xf32, strided<[1]>> -> !vir.vec<256xf32>
      %15 = vir.load %transpose[] : memref<256xf32, strided<[1]>> -> !vir.vec<256xf32>
      %16 = arith.subf %14, %15 : !vir.vec<256xf32>
      vir.store %16, %transpose_9[] : !vir.vec<256xf32> -> memref<256xf32, strided<[1]>>
      vector.yield
    }
    vir.set_vl %c256 : index {
      %transpose = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %14 = vir.load %transpose[] : memref<256xf32, strided<[1]>> -> !vir.vec<256xf32>
      %15 = math.exp %14 : !vir.vec<256xf32>
      vir.store %15, %transpose[] : !vir.vec<256xf32> -> memref<256xf32, strided<[1]>>
      vector.yield
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    %9 = memref.load %alloc_4[] : memref<f32>
    memref.store %9, %alloca_5[] : memref<f32>
    vir.set_vl %c256 : index {
      %14 = vir.load %alloc[%c0] : memref<256xf32> -> !vir.vec<?xf32>
      %15 = memref.load %alloca_5[] : memref<f32>
      %16 = vir.reduce %14, %15 {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
      memref.store %16, %alloca_5[] : memref<f32>
      vector.yield
    }
    %10 = memref.load %alloca_5[] : memref<f32>
    memref.store %10, %alloc_4[] : memref<f32>
    %11 = memref.load %alloc_4[] : memref<f32>
    vir.set_vl %c256 : index {
      %transpose = memref.transpose %alloc_3 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %14 = vir.broadcast %11 : f32 -> !vir.vec<256xf32>
      vir.store %14, %transpose[] : !vir.vec<256xf32> -> memref<256xf32, strided<[1]>>
      vector.yield
    }
    vir.set_vl %c256 : index {
      %transpose = memref.transpose %alloc_3 (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %transpose_9 = memref.transpose %alloc (d0) -> (d0) : memref<256xf32> to memref<256xf32, strided<[1]>>
      %14 = vir.load %transpose_9[] : memref<256xf32, strided<[1]>> -> !vir.vec<256xf32>
      %15 = vir.load %transpose[] : memref<256xf32, strided<[1]>> -> !vir.vec<256xf32>
      %16 = arith.divf %14, %15 : !vir.vec<256xf32>
      vir.store %16, %transpose_9[] : !vir.vec<256xf32> -> memref<256xf32, strided<[1]>>
      vector.yield
    }
    %12 = arith.muli %arg8, %arg3 : i32
    %13 = arith.index_cast %12 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%13], sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
    %subview_7 = memref.subview %alloc[0] [%4] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    %subview_8 = memref.subview %reinterpret_cast_6[0] [%4] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    memref.copy %subview_7, %subview_8 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1], offset: ?>>
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

