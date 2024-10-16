module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @timingStart() attributes {sym_visibility = "private"}
  llvm.func @timingEnd() attributes {sym_visibility = "private"}
  llvm.func @rtclock() -> f64 attributes {sym_visibility = "private"}
  llvm.func @printF64(f64) attributes {sym_visibility = "private"}
  llvm.func @printNewline() attributes {sym_visibility = "private"}
  llvm.func @alloc_2d_filled_f32(%arg0: i64, %arg1: i64, %arg2: f32) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = builtin.unrealized_conversion_cast %arg1 : i64 to index
    %1 = builtin.unrealized_conversion_cast %arg0 : i64 to index
    %2 = builtin.unrealized_conversion_cast %1 : index to i64
    %3 = builtin.unrealized_conversion_cast %0 : index to i64
    %4 = builtin.unrealized_conversion_cast %1 : index to i64
    %5 = builtin.unrealized_conversion_cast %0 : index to i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = builtin.unrealized_conversion_cast %6 : i64 to index
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mul %5, %4  : i64
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.getelementptr %11[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.call @malloc(%13) : (i64) -> !llvm.ptr
    %15 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %4, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %5, %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %5, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %9, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = builtin.unrealized_conversion_cast %23 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32>
    llvm.br ^bb1(%6 : i64)
  ^bb1(%25: i64):  // 2 preds: ^bb0, ^bb5
    %26 = builtin.unrealized_conversion_cast %25 : i64 to index
    %27 = builtin.unrealized_conversion_cast %26 : index to i64
    %28 = builtin.unrealized_conversion_cast %26 : index to i64
    %29 = llvm.icmp "slt" %27, %2 : i64
    llvm.cond_br %29, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%6 : i64)
  ^bb3(%30: i64):  // 2 preds: ^bb2, ^bb4
    %31 = builtin.unrealized_conversion_cast %30 : i64 to index
    %32 = builtin.unrealized_conversion_cast %31 : index to i64
    %33 = builtin.unrealized_conversion_cast %31 : index to i64
    %34 = llvm.icmp "slt" %32, %3 : i64
    llvm.cond_br %34, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %35 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.mul %28, %36  : i64
    %38 = llvm.add %37, %33  : i64
    %39 = llvm.getelementptr %35[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %arg2, %39 : f32, !llvm.ptr
    %40 = llvm.add %32, %8  : i64
    %41 = builtin.unrealized_conversion_cast %40 : i64 to index
    llvm.br ^bb3(%40 : i64)
  ^bb5:  // pred: ^bb3
    %42 = llvm.add %27, %8  : i64
    %43 = builtin.unrealized_conversion_cast %42 : i64 to index
    llvm.br ^bb1(%42 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return %23 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = builtin.unrealized_conversion_cast %1 : index to i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = builtin.unrealized_conversion_cast %3 : i64 to index
    %5 = builtin.unrealized_conversion_cast %4 : index to i64
    %6 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %7 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(3 : index) : i64
    %9 = builtin.unrealized_conversion_cast %8 : i64 to index
    %10 = llvm.mlir.constant(8 : index) : i64
    %11 = builtin.unrealized_conversion_cast %10 : i64 to index
    %12 = llvm.mlir.constant(10 : index) : i64
    %13 = builtin.unrealized_conversion_cast %12 : i64 to index
    %14 = llvm.call @alloc_2d_filled_f32(%8, %8, %6) : (i64, i64, f32) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = builtin.unrealized_conversion_cast %14 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32>
    %16 = builtin.unrealized_conversion_cast %15 : memref<?x?xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.call @alloc_2d_filled_f32(%12, %12, %6) : (i64, i64, f32) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %18 = builtin.unrealized_conversion_cast %17 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32>
    %19 = builtin.unrealized_conversion_cast %18 : memref<?x?xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.call @alloc_2d_filled_f32(%10, %10, %7) : (i64, i64, f32) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %21 = builtin.unrealized_conversion_cast %20 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32>
    %22 = builtin.unrealized_conversion_cast %21 : memref<?x?xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @timingStart() : () -> ()
    %23 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = builtin.unrealized_conversion_cast %23 : i64 to index
    %25 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = builtin.unrealized_conversion_cast %25 : i64 to index
    %27 = llvm.extractvalue %22[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = builtin.unrealized_conversion_cast %27 : i64 to index
    %29 = llvm.extractvalue %22[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = builtin.unrealized_conversion_cast %29 : i64 to index
    llvm.br ^bb1(%0 : i64)
  ^bb1(%31: i64):  // 2 preds: ^bb0, ^bb11
    %32 = builtin.unrealized_conversion_cast %31 : i64 to index
    %33 = builtin.unrealized_conversion_cast %32 : index to i64
    %34 = builtin.unrealized_conversion_cast %32 : index to i64
    %35 = llvm.icmp "slt" %33, %27 : i64
    llvm.cond_br %35, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%0 : i64)
  ^bb3(%36: i64):  // 2 preds: ^bb2, ^bb10
    %37 = builtin.unrealized_conversion_cast %36 : i64 to index
    %38 = builtin.unrealized_conversion_cast %37 : index to i64
    %39 = builtin.unrealized_conversion_cast %37 : index to i64
    %40 = llvm.icmp "slt" %38, %29 : i64
    llvm.cond_br %40, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%0 : i64)
  ^bb5(%41: i64):  // 2 preds: ^bb4, ^bb9
    %42 = builtin.unrealized_conversion_cast %41 : i64 to index
    %43 = builtin.unrealized_conversion_cast %42 : index to i64
    %44 = builtin.unrealized_conversion_cast %42 : index to i64
    %45 = llvm.icmp "slt" %43, %23 : i64
    llvm.cond_br %45, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%0 : i64)
  ^bb7(%46: i64):  // 2 preds: ^bb6, ^bb8
    %47 = builtin.unrealized_conversion_cast %46 : i64 to index
    %48 = builtin.unrealized_conversion_cast %47 : index to i64
    %49 = builtin.unrealized_conversion_cast %47 : index to i64
    %50 = llvm.icmp "slt" %48, %25 : i64
    llvm.cond_br %50, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %51 = llvm.add %33, %43  : i64
    %52 = builtin.unrealized_conversion_cast %51 : i64 to index
    %53 = builtin.unrealized_conversion_cast %52 : index to i64
    %54 = llvm.add %38, %48  : i64
    %55 = builtin.unrealized_conversion_cast %54 : i64 to index
    %56 = builtin.unrealized_conversion_cast %55 : index to i64
    %57 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.mul %53, %58  : i64
    %60 = llvm.add %59, %56  : i64
    %61 = llvm.getelementptr %57[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %62 = llvm.load %61 : !llvm.ptr -> f32
    %63 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.mul %44, %64  : i64
    %66 = llvm.add %65, %49  : i64
    %67 = llvm.getelementptr %63[%66] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %68 = llvm.load %67 : !llvm.ptr -> f32
    %69 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.extractvalue %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.mul %34, %70  : i64
    %72 = llvm.add %71, %39  : i64
    %73 = llvm.getelementptr %69[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %74 = llvm.load %73 : !llvm.ptr -> f32
    %75 = llvm.fmul %62, %68  : f32
    %76 = llvm.fadd %74, %75  : f32
    %77 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.extractvalue %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mul %34, %78  : i64
    %80 = llvm.add %79, %39  : i64
    %81 = llvm.getelementptr %77[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %76, %81 : f32, !llvm.ptr
    %82 = llvm.add %48, %3  : i64
    %83 = builtin.unrealized_conversion_cast %82 : i64 to index
    llvm.br ^bb7(%82 : i64)
  ^bb9:  // pred: ^bb7
    %84 = llvm.add %43, %3  : i64
    %85 = builtin.unrealized_conversion_cast %84 : i64 to index
    llvm.br ^bb5(%84 : i64)
  ^bb10:  // pred: ^bb5
    %86 = llvm.add %38, %3  : i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    llvm.br ^bb3(%86 : i64)
  ^bb11:  // pred: ^bb3
    %88 = llvm.add %33, %3  : i64
    %89 = builtin.unrealized_conversion_cast %88 : i64 to index
    llvm.br ^bb1(%88 : i64)
  ^bb12:  // pred: ^bb1
    llvm.call @timingEnd() : () -> ()
    llvm.return
  }
}

