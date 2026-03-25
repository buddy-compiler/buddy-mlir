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
    vir.set_vl %c64 : index {
      %transpose = memref.transpose %alloc (d0, d1) -> (d0, d1) : memref<32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      %41 = vir.broadcast %cst : f32 -> !vir.vec<32x64xf32>
      vir.store %41, %transpose[] : !vir.vec<32x64xf32> -> memref<32x64xf32, strided<[64, 1]>>
      vector.yield
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
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
    memref.copy %alloc, %alloc_0 : memref<32x64xf32> to memref<32x64xf32>
    %27:3 = scf.for %arg15 = %c0_i32 to %24 step %c1_i32 iter_args(%arg16 = %19, %arg17 = %c0, %arg18 = %alloc_0) -> (index, index, memref<32x64xf32>)  : i32 {
      %41 = arith.addi %arg17, %16 : index
      %42 = arith.remsi %41, %22 : index
      %43 = arith.subi %41, %42 : index
      %44 = arith.addi %42, %c64 : index
      %45 = arith.minsi %44, %22 : index
      %46 = arith.subi %45, %42 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [%c16, %46], strides: [%21, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %47 = arith.subi %c64, %46 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%c16, %47], strides: [%21, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %48 = arith.remsi %arg16, %18 : index
      %49 = arith.addi %20, %48 : index
      %50 = arith.subi %49, %arg16 : index
      %51 = arith.divsi %50, %18 : index
      %52 = arith.minsi %51, %c32 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%arg16], sizes: [%52, %c16], strides: [%18, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %53 = arith.subi %c32, %52 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%48], sizes: [%53, %c16], strides: [%18, %c1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %54 = arith.muli %arg15, %c16_i32 : i32
      %55 = arith.subi %arg5, %54 : i32
      %56 = arith.index_cast %55 : i32 to index
      %57 = arith.minsi %56, %c16 : index
      %58 = arith.maxsi %57, %c0 : index
      %alloc_6 = memref.alloc() : memref<32x16xf32>
      %59 = arith.cmpi slt, %58, %c16 : index
      scf.if %59 {
        vir.set_vl %c16 : index {
          %transpose = memref.transpose %alloc_6 (d0, d1) -> (d0, d1) : memref<32x16xf32> to memref<32x16xf32, strided<[16, 1]>>
          %62 = vir.broadcast %cst : f32 -> !vir.vec<32x16xf32>
          vir.store %62, %transpose[] : !vir.vec<32x16xf32> -> memref<32x16xf32, strided<[16, 1]>>
          vector.yield
        }
      }
      %subview_7 = memref.subview %reinterpret_cast_4[0, 0] [%52, %58] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_8 = memref.subview %reinterpret_cast_5[0, 0] [%53, %58] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_9 = memref.subview %alloc_6[0, 0] [%52, %58] [1, 1] : memref<32x16xf32> to memref<?x?xf32, strided<[16, 1]>>
      %subview_10 = memref.subview %alloc_6[%52, 0] [%53, %58] [1, 1] : memref<32x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
      memref.copy %subview_7, %subview_9 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
      memref.copy %subview_8, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[16, 1], offset: ?>>
      %alloc_11 = memref.alloc() : memref<16x64xf32>
      scf.if %59 {
        vir.set_vl %c64 : index {
          %transpose = memref.transpose %alloc_11 (d0, d1) -> (d0, d1) : memref<16x64xf32> to memref<16x64xf32, strided<[64, 1]>>
          %62 = vir.broadcast %cst : f32 -> !vir.vec<16x64xf32>
          vir.store %62, %transpose[] : !vir.vec<16x64xf32> -> memref<16x64xf32, strided<[64, 1]>>
          vector.yield
        }
      }
      %subview_12 = memref.subview %reinterpret_cast_2[0, 0] [%58, %46] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_13 = memref.subview %reinterpret_cast_3[0, 0] [%58, %47] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_14 = memref.subview %alloc_11[0, 0] [%58, %46] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      %subview_15 = memref.subview %alloc_11[0, %46] [%58, %47] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %subview_12, %subview_14 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_13, %subview_15 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
      memref.copy %alloc, %alloc_16 : memref<32x64xf32> to memref<32x64xf32>
      vir.set_vl %c64 : index {
        affine.for %arg19 = 0 to 32 {
          %62 = vir.load %alloc_16[%arg19, %c0] : memref<32x64xf32> -> !vir.vec<?xf32>
          %63 = affine.for %arg20 = 0 to 16 iter_args(%arg21 = %62) -> (!vir.vec<?xf32>) {
            %64 = memref.load %alloc_6[%arg19, %arg20] : memref<32x16xf32>
            %65 = vir.broadcast %64 : f32 -> !vir.vec<?xf32>
            %66 = vir.load %alloc_11[%arg20, %c0] : memref<16x64xf32> -> !vir.vec<?xf32>
            %67 = vir.fma %65, %66, %arg21 : !vir.vec<?xf32>
            affine.yield %67 : !vir.vec<?xf32>
          }
          vir.store %63, %alloc_16[%arg19, %c0] : !vir.vec<?xf32> -> memref<32x64xf32>
        }
        vector.yield
      }
      vir.set_vl %c64 : index {
        %transpose = memref.transpose %alloc_16 (d0, d1) -> (d0, d1) : memref<32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
        %transpose_17 = memref.transpose %arg18 (d0, d1) -> (d0, d1) : memref<32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
        %62 = vir.load %transpose_17[] : memref<32x64xf32, strided<[64, 1]>> -> !vir.vec<32x64xf32>
        %63 = vir.load %transpose[] : memref<32x64xf32, strided<[64, 1]>> -> !vir.vec<32x64xf32>
        %64 = arith.addf %62, %63 : !vir.vec<32x64xf32>
        vir.store %64, %transpose_17[] : !vir.vec<32x64xf32> -> memref<32x64xf32, strided<[64, 1]>>
        vector.yield
      }
      %60 = arith.addi %arg16, %c16 : index
      %61 = arith.addi %arg17, %26 : index
      scf.yield %60, %61, %arg18 : index, index, memref<32x64xf32>
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
    %subview_1 = memref.subview %reinterpret_cast[0, 0] [%39, %40] [1, 1] : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    memref.copy %subview, %subview_1 : memref<?x?xf32, strided<[64, 1]>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
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

