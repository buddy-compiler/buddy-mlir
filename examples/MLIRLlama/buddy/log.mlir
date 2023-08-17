module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @forward() -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(8 : index) : i64
    %1 = llvm.mlir.constant(16 : index) : i64
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(64 : index) : i64
    %5 = llvm.mlir.constant(512 : index) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.getelementptr %6[512] : (!llvm.ptr) -> !llvm.ptr, f32
    %8 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %9 = llvm.mlir.constant(64 : index) : i64
    %10 = llvm.add %8, %9  : i64
    %11 = llvm.call @malloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.sub %9, %13  : i64
    %15 = llvm.add %12, %14  : i64
    %16 = llvm.urem %15, %9  : i64
    %17 = llvm.sub %15, %16  : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %19 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %20 = llvm.insertvalue %11, %19[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.mlir.constant(0 : index) : i64
    %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %0, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.insertvalue %1, %24[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %2, %25[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.insertvalue %4, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %28 = llvm.insertvalue %2, %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %29 = llvm.insertvalue %3, %28[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %31 = llvm.insertvalue %11, %30[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %32 = llvm.insertvalue %18, %31[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %33 = llvm.mlir.constant(0 : index) : i64
    %34 = llvm.insertvalue %33, %32[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.insertvalue %35, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %37 = llvm.mlir.constant(64 : index) : i64
    %38 = llvm.insertvalue %37, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.mlir.constant(16 : index) : i64
    %40 = llvm.insertvalue %39, %38[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %41 = llvm.mlir.constant(4 : index) : i64
    %42 = llvm.insertvalue %41, %40[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %43 = llvm.mlir.constant(4 : index) : i64
    %44 = llvm.insertvalue %43, %42[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %45 = llvm.mlir.constant(1 : index) : i64
    %46 = llvm.insertvalue %45, %44[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.return %46 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
  llvm.func @_mlir_ciface_forward(%arg0: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @forward() : () -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.store %0, %arg0 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    llvm.return
  }
}

