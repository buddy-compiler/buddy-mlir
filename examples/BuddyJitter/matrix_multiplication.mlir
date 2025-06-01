module {
  llvm.func @malloc(i64) -> !llvm.ptr
  # 对应memref %cst中存储的数据
  llvm.mlir.global private constant @__constant_2x2xf32(dense<0.000000e+00> : tensor<2x2xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2 x array<2 x f32>>

  llvm.func @forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    # 以下将参数提取到ranked_memref结构体中
    # struct memref(allocated, aligned, offset, shape, strides)
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    # %7 是memref %arg1
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    # %15 是memref $arg2

    # 循环的起始结束和步长
    %16 = llvm.mlir.constant(3 : index) : i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mlir.constant(2 : index) : i64
    %19 = llvm.mlir.constant(0 : index) : i64

    %20 = llvm.mlir.constant(2 : index) : i64
    %21 = llvm.mlir.constant(2 : index) : i64
    %22 = llvm.mlir.constant(1 : index) : i64
    # Useless
    %23 = llvm.mlir.constant(4 : index) : i64

    # Useless
    %24 = llvm.mlir.zero : !llvm.ptr
    %25 = llvm.getelementptr %24[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64

    # 这里应该是在初始化memref %cst
    %27 = llvm.mlir.addressof @__constant_2x2xf32 : !llvm.ptr
    %28 = llvm.getelementptr %27[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<2 x f32>>

    %29 = llvm.mlir.constant(3735928559 : index) : i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr

    %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>


    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

    # allocate 字段
    # 存储是%cst的地址
    %33 = llvm.insertvalue %28, %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

    # offset始终为0
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

    # shape字段 (2, 2)
    %36 = llvm.insertvalue %20, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.insertvalue %21, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    # strides字段 (2, 1)
    %38 = llvm.insertvalue %21, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.insertvalue %22, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

    # 分配结果memref
    %40 = llvm.mlir.constant(2 : index) : i64
    %41 = llvm.mlir.constant(2 : index) : i64
    %42 = llvm.mlir.constant(1 : index) : i64
    %43 = llvm.mlir.constant(4 : index) : i64

    %44 = llvm.mlir.zero : !llvm.ptr
    %45 = llvm.getelementptr %44[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.mlir.constant(64 : index) : i64
    %48 = llvm.add %46, %47  : i64
    %49 = llvm.call @malloc(%48) : (i64) -> !llvm.ptr
    %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %51 = llvm.mlir.constant(1 : index) : i64
    %52 = llvm.sub %47, %51  : i64
    %53 = llvm.add %50, %52  : i64
    %54 = llvm.urem %53, %47  : i64
    %55 = llvm.sub %53, %54  : i64
    %56 = llvm.inttoptr %55 : i64 to !llvm.ptr

    %57 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %58 = llvm.insertvalue %49, %57[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %59 = llvm.insertvalue %56, %58[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

    %60 = llvm.mlir.constant(0 : index) : i64
    %61 = llvm.insertvalue %60, %59[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

    %62 = llvm.insertvalue %40, %61[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.insertvalue %41, %62[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.insertvalue %41, %63[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %65 = llvm.insertvalue %42, %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

    %66 = llvm.mlir.constant(1 : index) : i64
    # 2 * 1 = 2
    %67 = llvm.mul %20, %66  : i64
    # 2 * 2 = 4
    %68 = llvm.mul %67, %21  : i64
    # 0
    %69 = llvm.mlir.zero : !llvm.ptr
    # 0 + 4 * 1 = 4
    %70 = llvm.getelementptr %69[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %71 = llvm.ptrtoint %70 : !llvm.ptr to i64
    # 4 * 4 = 16
    %72 = llvm.mul %68, %71  : i64
    # __constant_2x2xf32复制到结果中
    "llvm.intr.memcpy"(%56, %28, %72) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    omp.parallel {
      # i = 0 -> 2
      omp.wsloop for  (%arg14) : i64 = (%19) to (%18) step (%17) {
        %73 = llvm.intr.stacksave : !llvm.ptr
        llvm.br ^bb1
      ^bb1:  // pred: ^bb0
        omp.parallel {
          # j = 0 -> 2
          omp.wsloop for  (%arg15) : i64 = (%19) to (%18) step (%17) {
            %74 = llvm.intr.stacksave : !llvm.ptr
            llvm.br ^bb1
          ^bb1:  // pred: ^bb0
            llvm.br ^bb2(%19 : i64)
          # basic block 传参
          # 所以%75就是 k
          # k 0 -> 3
          ^bb2(%75: i64):  // 2 preds: ^bb1, ^bb3
            %76 = llvm.icmp "slt" %75, %16 : i64
            llvm.cond_br %76, ^bb3, ^bb4
          ^bb3:  // pred: ^bb2
            %77 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %78 = llvm.mlir.constant(3 : index) : i64
            # i * 3
            %79 = llvm.mul %arg14, %78  : i64
            # i * 3 + k
            %80 = llvm.add %79, %75  : i64
            %81 = llvm.getelementptr %77[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            # 按照memref arg1的规格加载memref arg2?
            %82 = llvm.load %81 : !llvm.ptr -> f32


            # 加载memref arg2的规格加载memref arg1?
            %83 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
            %84 = llvm.mlir.constant(2 : index) : i64
            # k * 2
            %85 = llvm.mul %75, %84  : i64
            # j + k * 2
            %86 = llvm.add %85, %arg15  : i64
            %87 = llvm.getelementptr %83[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %88 = llvm.load %87 : !llvm.ptr -> f32

            # 加载结果位置
            %89 = llvm.mlir.constant(2 : index) : i64
            # i * 2
            %90 = llvm.mul %arg14, %89  : i64
            # i * 2 + j
            %91 = llvm.add %90, %arg15  : i64
            %92 = llvm.getelementptr %56[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %93 = llvm.load %92 : !llvm.ptr -> f32

            %94 = llvm.fmul %82, %88  : f32
            %95 = llvm.fadd %93, %94  : f32

            # 再次加载结果位置
            %96 = llvm.mlir.constant(2 : index) : i64
            %97 = llvm.mul %arg14, %96  : i64
            %98 = llvm.add %97, %arg15  : i64
            %99 = llvm.getelementptr %56[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            llvm.store %95, %99 : f32, !llvm.ptr
            # k + 1
            %100 = llvm.add %75, %17  : i64
            llvm.br ^bb2(%100 : i64)
          ^bb4:  // pred: ^bb2
            llvm.intr.stackrestore %74 : !llvm.ptr
            llvm.br ^bb5
          ^bb5:  // pred: ^bb4
            omp.yield
          }
          omp.terminator
        }
        llvm.intr.stackrestore %73 : !llvm.ptr
        llvm.br ^bb2
      ^bb2:  // pred: ^bb1
        omp.yield
      }
      omp.terminator
    }
    llvm.return %65 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
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