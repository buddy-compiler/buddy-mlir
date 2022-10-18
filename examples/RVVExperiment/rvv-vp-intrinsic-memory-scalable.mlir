module {
  // Define the 4x4 global memory.
  llvm.mlir.global private @gv(dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.200000e+01], [1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]]> : tensor<4x4xf32>) {addr_space = 0 : i32} : !llvm.array<4 x array<4 x f32>>

  func.func @alloc_mem() -> memref<4xf32> {
    %f0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mem = memref.alloc() : memref<4xf32>
    %dim = memref.dim %mem, %c0 : memref<4xf32>
    scf.for %idx = %c0 to %dim step %c1 {
      memref.store %f0, %mem[%idx] : memref<4xf32>
    }
    return %mem : memref<4xf32>
  }

  func.func @alloc_mask() -> memref<8xi1> {
    %f0 = arith.constant 0 : i1
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mem = memref.alloc() : memref<8xi1>
    %dim = memref.dim %mem, %c0 : memref<8xi1>
    scf.for %idx = %c0 to %dim step %c1 {
      memref.store %f0, %mem[%idx] : memref<8xi1>
    }
    return %mem : memref<8xi1>
  }
  
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() -> i32 {
    // Prepare the memory pointer.
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %c0 : index to i64
    %1 = llvm.mlir.constant(4 : index) : i64
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(16 : index) : i64
    %5 = llvm.mlir.null : !llvm.ptr<f32>
    %6 = llvm.getelementptr %5[%4] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %7 = llvm.ptrtoint %6 : !llvm.ptr<f32> to i64
    %8 = llvm.mlir.addressof @gv : !llvm.ptr<array<4 x array<4 x f32>>>
    %9 = llvm.getelementptr %8[0, 0, 0] : (!llvm.ptr<array<4 x array<4 x f32>>>) -> !llvm.ptr<f32>
    %10 = llvm.mlir.constant(3735928559 : index) : i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr<f32>
    %12 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %11, %12[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %9, %13[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.insertvalue %15, %14[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %1, %16[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %2, %17[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %2, %18[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %3, %19[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.mlir.constant(4 : index) : i64
    %23 = llvm.mul %0, %22  : i64
    %24 = llvm.add %23, %0  : i64
    %iptr = llvm.getelementptr %21[%24] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>

    // Configure the register.
    // SEW = 32
    %sew = arith.constant 2 : index
    // LMUL = 2
    %lmul = arith.constant 1 : index
    // AVL = 4
    %avl = arith.constant 4 : index
    %vl = rvv.setvl %avl, %sew, %lmul : index

    // Create the mask.
    %c4 = arith.constant 4 : index
    %mask_scalable = vector.create_mask %c4 : vector<[4]xi1>

    // Set the EVL.
    %evl3 = arith.constant 3 : i32
    // VP intrinsic "load" operation + scalable vector type.
    %vec_scalable = "llvm.intr.vp.load" (%iptr, %mask_scalable, %evl3) :
         (!llvm.ptr<f32>, vector<[4]xi1>, i32) -> vector<[4]xf32>
    // Print the load result.
    %res = call @alloc_mem() : () -> memref<4xf32>
    rvv.store %vec_scalable, %res[%c0], %vl : vector<[4]xf32>, memref<4xf32>, index
    %print_res = memref.cast %res : memref<4xf32> to memref<*xf32>
    call @printMemrefF32(%print_res) : (memref<*xf32>) -> ()


    // Define the output index.
    %c1 = arith.constant 1 : index
    %c1_i64 = builtin.unrealized_conversion_cast %c1 : index to i64
    %c3 = arith.constant 3 : index
    %c3_i64 = builtin.unrealized_conversion_cast %c3 : index to i64
    // Prepare the output pointer
    %29 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.mlir.constant(4 : index) : i64
    %31 = llvm.mul %c3_i64, %30  : i64
    %32 = llvm.add %31, %c1_i64  : i64
    %33 = llvm.getelementptr %29[%32] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    // VP intrinsic "store" operation + scalable vector type.
    "llvm.intr.vp.store" (%vec_scalable, %33, %mask_scalable, %evl3) :
         (vector<[4]xf32>, !llvm.ptr<f32>, vector<[4]xi1>, i32) -> ()
    %34 = llvm.mlir.constant(1 : index) : i64
    %35 = llvm.alloca %34 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    llvm.store %20, %35 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    %36 = llvm.bitcast %35 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>> to !llvm.ptr<i8>
    %37 = llvm.mlir.constant(2 : index) : i64
    %38 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %39 = llvm.insertvalue %37, %38[0] : !llvm.struct<(i64, ptr<i8>)> 
    %40 = llvm.insertvalue %36, %39[1] : !llvm.struct<(i64, ptr<i8>)> 
    %41 = builtin.unrealized_conversion_cast %40 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
    call @printMemrefF32(%41) : (memref<*xf32>) -> ()

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
