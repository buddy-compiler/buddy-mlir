module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.mlir.constant(1024 : index) : i64
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.mlir.constant(1024 : index) : i64
    %20 = llvm.mlir.constant(1024 : index) : i64
    %21 = llvm.mlir.constant(1 : index) : i64
    %22 = llvm.mlir.constant(1048576 : index) : i64
    %23 = llvm.mlir.zero : !llvm.ptr
    %24 = llvm.getelementptr %23[1048576] : (!llvm.ptr) -> !llvm.ptr, f32
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.mlir.constant(64 : index) : i64
    %27 = llvm.add %25, %26  : i64
    %28 = llvm.call @malloc(%27) : (i64) -> !llvm.ptr
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.sub %26, %30  : i64
    %32 = llvm.add %29, %31  : i64
    %33 = llvm.urem %32, %26  : i64
    %34 = llvm.sub %32, %33  : i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.insertvalue %28, %36[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.insertvalue %39, %38[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %19, %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.insertvalue %20, %41[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.insertvalue %20, %42[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %21, %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%18 : i64)
  ^bb1(%45: i64):  // 2 preds: ^bb0, ^bb5
    %46 = llvm.icmp "slt" %45, %17 : i64
    llvm.cond_br %46, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%18 : i64)
  ^bb3(%47: i64):  // 2 preds: ^bb2, ^bb4
    %48 = llvm.icmp "slt" %47, %17 : i64
    llvm.cond_br %48, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %49 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.mlir.constant(1024 : index) : i64
    %51 = llvm.mul %45, %50  : i64
    %52 = llvm.add %51, %47  : i64
    %53 = llvm.getelementptr %49[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %54 = llvm.load %53 : !llvm.ptr -> f32
    %55 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.mlir.constant(1024 : index) : i64
    %57 = llvm.mul %45, %56  : i64
    %58 = llvm.add %57, %47  : i64
    %59 = llvm.getelementptr %55[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %60 = llvm.load %59 : !llvm.ptr -> f32
    %61 = llvm.fadd %54, %60  : f32
    %62 = llvm.mlir.constant(1024 : index) : i64
    %63 = llvm.mul %45, %62  : i64
    %64 = llvm.add %63, %47  : i64
    %65 = llvm.getelementptr %35[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %61, %65 : f32, !llvm.ptr
    %66 = llvm.add %47, %16  : i64
    llvm.br ^bb3(%66 : i64)
  ^bb5:  // pred: ^bb3
    %67 = llvm.add %45, %16  : i64
    llvm.br ^bb1(%67 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return %44 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @_mlir_ciface_forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.call @forward(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %16, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
}
