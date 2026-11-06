//===- kernel.contract.mlir ---------------------------------------------===//
//
// 196x768x768 matmul 的跨项目契约 IR（design reference，buddy-opt 可解析）。
//
// 图模型（v1 决策）：compute graph 以 region/scope 为单位组织：
//   - scope 0 = function 根 scope；1/2 = if 的两个 arm；3/4 = 两个 K 循环体。
//   - 每个 scope 各自声明完备性；函数级 complete 是聚合（所有 scope complete）。
//   - 循环用 uniform-loop 摘要（trip_count/uniform/serial + carried slots），
//     不要求枚举动态实例。
//   - 逻辑值引用仍用整数 ID；契约 verifier 做双向/闭包检查。

module attributes {
  contract.version = 1 : i32,
  contract.producer = "triton-riscv",
  contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
} {
  func.func @triton_matmul_i8_196x768x768(
      %arg0: memref<*xi8> {contract.value_id = 0 : i64},
      %arg1: memref<*xi8> {contract.value_id = 1 : i64},
      %arg2: memref<*xi32> {contract.value_id = 2 : i64},
      %arg3: i32 {contract.program = {role = "num_programs", axis = 0 : i32}},
      %arg4: i32 {contract.program = {role = "num_programs", axis = 1 : i32}},
      %arg5: i32 {contract.program = {role = "num_programs", axis = 2 : i32}},
      %arg6: i32 {contract.program = {role = "program_id", axis = 0 : i32}},
      %arg7: i32 {contract.program = {role = "program_id", axis = 1 : i32}},
      %arg8: i32 {contract.program = {role = "program_id", axis = 2 : i32}})
      attributes {
        contract.grid = array<i32: 7, 6, 1>,
        contract.compute_graph = {version = 1 : i32, root_scope = 0 : i64,
                                  complete = true, scope_count = 5 : i64,
                                  source_lineage_complete = true},
        llvm.emit_c_interface, noinline = false} {

    %c0_i32 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %c768 = arith.constant 768 : index
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c6_i32 = arith.constant 6 : i32
    %0 = arith.cmpi slt, %arg6, %c6_i32 : i32
    scf.if %0 {

      // ===== then arm：满块（scope 1）=====
      %1 = arith.muli %arg6, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.muli %arg7, %c128_i32 : i32
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.muli %2, %c768 : index
      %6 = arith.addi %5, %4 : index
      %c_view = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [32, 128],
          strides: [768, 1] : memref<*xi32> to memref<32x128xi32, strided<[768, 1], offset: ?>>
      %c_buf = memref.alloc() alignment = 64
          {contract.buffer = {role = "tile", kind = "accumulator",
                              logical_shape = array<i64: 32, 128>}}
          : memref<32x128xi32>
      memref.copy %c_view, %c_buf
          {contract.task = {id = 0 : i64, kind = "memory-load", scope = 1 : i64,
                            stage = 0 : i64, predecessors = array<i64>,
                            input_values = array<i64: 2>, output_values = array<i64: 3>,
                            barrier_after = true, rematerializable = false,
                            origin_id = 0 : i64},
           contract.access = {kind = "load", boundary = "external_ddr",
                              origin_id = 0 : i64, mask = {kind = "none"},
                              other = 0 : i32},
           contract.program_axes = {varying = array<i64: 0, 1>, invariant = array<i64: 2>},
           contract.provenance = {source_load_origin_ids = array<i64: 0>}}
          : memref<32x128xi32, strided<[768, 1], offset: ?>> to memref<32x128xi32>
      %7 = arith.muli %4, %c768 : index

      scf.for %k = %c0_i32 to %c6_i32 step %c1_i32 : i32 {

        // ===== then 循环体（scope 3）=====
        %8 = arith.muli %k, %c128_i32 : i32
        %9 = arith.index_cast %8 : i32 to index
        %10 = arith.addi %5, %9 : index
        %a_view = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 128],
            strides: [768, 1] : memref<*xi8> to memref<32x128xi8, strided<[768, 1], offset: ?>>
        %a_buf = memref.alloc() alignment = 64
            {contract.buffer = {role = "tile", kind = "input-a",
                                logical_shape = array<i64: 32, 128>}}
            : memref<32x128xi8>
        memref.copy %a_view, %a_buf
            {contract.task = {id = 1 : i64, kind = "memory-load", scope = 3 : i64,
                              stage = 0 : i64, predecessors = array<i64>,
                              input_values = array<i64: 0>, output_values = array<i64: 4>,
                              barrier_after = true, rematerializable = false,
                              origin_id = 1 : i64},
             contract.access = {kind = "load", boundary = "external_ddr",
                                origin_id = 1 : i64, mask = {kind = "none"},
                                other = 0 : i8},
             contract.program_axes = {varying = array<i64: 0>, invariant = array<i64: 1, 2>},
             contract.provenance = {source_load_origin_ids = array<i64: 1>}}
            : memref<32x128xi8, strided<[768, 1], offset: ?>> to memref<32x128xi8>
        %11 = arith.addi %9, %7 : index
        %b_view = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [128, 128],
            strides: [1, 768] : memref<*xi8> to memref<128x128xi8, strided<[1, 768], offset: ?>>
        %b_flat = memref.alloc() alignment = 64
            {contract.buffer = {role = "tile", kind = "input-b",
                                logical_shape = array<i64: 128, 128>}}
            : memref<16384xi8>
        %b_2d = memref.reinterpret_cast %b_flat to offset: [0], sizes: [128, 128],
            strides: [1, 128] : memref<16384xi8> to memref<128x128xi8, strided<[1, 128]>>
        memref.copy %b_view, %b_2d
            {contract.task = {id = 2 : i64, kind = "memory-load", scope = 3 : i64,
                              stage = 0 : i64, predecessors = array<i64>,
                              input_values = array<i64: 1>, output_values = array<i64: 5>,
                              barrier_after = true, rematerializable = false,
                              origin_id = 2 : i64},
             contract.access = {kind = "load", boundary = "external_ddr",
                                origin_id = 2 : i64, mask = {kind = "none"},
                                other = 0 : i8},
             contract.program_axes = {varying = array<i64: 1>, invariant = array<i64: 0, 2>},
             contract.provenance = {source_load_origin_ids = array<i64: 2>}}
            : memref<128x128xi8, strided<[1, 768], offset: ?>> to memref<128x128xi8, strided<[1, 128]>>
        linalg.matmul
            {cast = #linalg.type_fn<cast_signed>,
             contract.task = {id = 3 : i64, kind = "dot", scope = 3 : i64,
                              stage = 1 : i64, predecessors = array<i64: 0, 1, 2>,
                              input_values = array<i64: 4, 5, 7>,
                              output_values = array<i64: 6>,
                              barrier_after = true, rematerializable = false,
                              origin_id = 3 : i64},
             contract.compute = {kind = "dot", signed = true, lhs = "i8",
                                 rhs = "i8", acc = "i32"},
             contract.program_axes = {varying = array<i64: 0, 1>, invariant = array<i64: 2>},
             contract.provenance = {origin_id = 3 : i64,
                                    source_load_origin_ids = array<i64: 0, 1, 2>,
                                    a_origin_id = 1 : i64, b_origin_id = 2 : i64,
                                    a_source_load_origin_ids = array<i64: 1>,
                                    b_source_load_origin_ids = array<i64: 2>,
                                    accumulator_source_load_origin_ids = array<i64: 0>,
                                    a_source_lineage_complete = true,
                                    b_source_lineage_complete = true,
                                    accumulator_source_lineage_complete = true}}
            ins(%a_buf, %b_2d : memref<32x128xi8>, memref<128x128xi8, strided<[1, 128]>>)
            outs(%c_buf : memref<32x128xi32>)
      } {contract.cf = {kind = "uniform_for", trip_count = 6 : i64,
                        uniform = true, serial = true},
         contract.scope = {id = 3 : i64, kind = "for-body", parent = 1 : i64,
                           complete = true},
         contract.value_ids = array<i64: 7, 8>,
         contract.loop_carried_slots = [{slot = 0 : i64,
             init_value_id = 3 : i64, iter_arg_value_id = 7 : i64,
             yielded_value_id = 6 : i64, result_value_id = 8 : i64,
             init_producer_compute_ids = array<i64: 0>,
             iter_arg_consumer_compute_ids = array<i64: 3>,
             backedge_producer_compute_ids = array<i64: 3>,
             exit_consumer_compute_ids = array<i64: 4>}]}

      memref.copy %c_buf, %c_view
          {contract.task = {id = 4 : i64, kind = "memory-store", scope = 1 : i64,
                            stage = 2 : i64, predecessors = array<i64: 0, 3>,
                            input_values = array<i64: 8>, output_values = array<i64>,
                            barrier_after = true, rematerializable = false,
                            origin_id = 4 : i64},
           contract.access = {kind = "store", boundary = "external_ddr",
                              origin_id = 4 : i64, mask = {kind = "none"}},
           contract.program_axes = {varying = array<i64: 0, 1>, invariant = array<i64: 2>},
           contract.provenance = {source_load_origin_ids = array<i64: 0, 1, 2>}}
          : memref<32x128xi32> to memref<32x128xi32, strided<[768, 1], offset: ?>>

    } else {

      // ===== else arm：tail 4 行（scope 2）=====
      %12 = arith.muli %arg6, %c32_i32 : i32
      %13 = arith.index_cast %12 : i32 to index
      %14 = arith.muli %arg7, %c128_i32 : i32
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.muli %13, %c768 : index
      %17 = arith.addi %16, %15 : index
      %c_view_tail = memref.reinterpret_cast %arg2 to offset: [%17], sizes: [32, 128],
          strides: [768, 1] : memref<*xi32> to memref<32x128xi32, strided<[768, 1], offset: ?>>
      %c_buf_tail = memref.alloc() alignment = 64
          {contract.buffer = {role = "tile", kind = "accumulator",
                              logical_shape = array<i64: 32, 128>}}
          : memref<32x128xi32>
      linalg.fill ins(%c0_i32 : i32) outs(%c_buf_tail : memref<32x128xi32>)
          {contract.task = {id = 5 : i64, kind = "fill", scope = 2 : i64,
                            stage = 0 : i64, predecessors = array<i64>,
                            input_values = array<i64>, output_values = array<i64: 9>,
                            barrier_after = false, rematerializable = false},
           contract.fill = {padding = true}}
      %c_src4 = memref.subview %c_view_tail[0, 0] [4, 128] [1, 1]
          : memref<32x128xi32, strided<[768, 1], offset: ?>> to memref<4x128xi32, strided<[768, 1], offset: ?>>
      %c_dst4 = memref.subview %c_buf_tail[0, 0] [4, 128] [1, 1]
          : memref<32x128xi32> to memref<4x128xi32, strided<[128, 1]>>
      memref.copy %c_src4, %c_dst4
          {contract.task = {id = 6 : i64, kind = "memory-load", scope = 2 : i64,
                            stage = 0 : i64, predecessors = array<i64: 5>,
                            input_values = array<i64: 2, 9>, output_values = array<i64: 10>,
                            barrier_after = true, rematerializable = false,
                            origin_id = 0 : i64},
           contract.access = {kind = "load", boundary = "external_ddr",
                              origin_id = 0 : i64,
                              mask = {kind = "rectangular",
                                      valid_shape = array<i64: 4, 128>},
                              other = 0 : i32},
           contract.program_axes = {varying = array<i64: 0, 1>, invariant = array<i64: 2>},
           contract.provenance = {source_load_origin_ids = array<i64: 0>}}
          : memref<4x128xi32, strided<[768, 1], offset: ?>> to memref<4x128xi32, strided<[128, 1]>>
      %18 = arith.muli %15, %c768 : index

      scf.for %k2 = %c0_i32 to %c6_i32 step %c1_i32 : i32 {

        // ===== else 循环体（scope 4）=====
        %19 = arith.muli %k2, %c128_i32 : i32
        %20 = arith.index_cast %19 : i32 to index
        %21 = arith.addi %16, %20 : index
        %a_view_tail = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [32, 128],
            strides: [768, 1] : memref<*xi8> to memref<32x128xi8, strided<[768, 1], offset: ?>>
        %a_buf_tail = memref.alloc() alignment = 64
            {contract.buffer = {role = "tile", kind = "input-a",
                                logical_shape = array<i64: 32, 128>}}
            : memref<32x128xi8>
        linalg.fill ins(%c0_i8 : i8) outs(%a_buf_tail : memref<32x128xi8>)
            {contract.task = {id = 7 : i64, kind = "fill", scope = 4 : i64,
                              stage = 0 : i64, predecessors = array<i64>,
                              input_values = array<i64>, output_values = array<i64: 11>,
                              barrier_after = false, rematerializable = false},
             contract.fill = {padding = true}}
        %a_src4 = memref.subview %a_view_tail[0, 0] [4, 128] [1, 1]
            : memref<32x128xi8, strided<[768, 1], offset: ?>> to memref<4x128xi8, strided<[768, 1], offset: ?>>
        %a_dst4 = memref.subview %a_buf_tail[0, 0] [4, 128] [1, 1]
            : memref<32x128xi8> to memref<4x128xi8, strided<[128, 1]>>
        memref.copy %a_src4, %a_dst4
            {contract.task = {id = 8 : i64, kind = "memory-load", scope = 4 : i64,
                              stage = 0 : i64, predecessors = array<i64: 7>,
                              input_values = array<i64: 0, 11>, output_values = array<i64: 12>,
                              barrier_after = true, rematerializable = false,
                              origin_id = 1 : i64},
             contract.access = {kind = "load", boundary = "external_ddr",
                                origin_id = 1 : i64,
                                mask = {kind = "rectangular",
                                        valid_shape = array<i64: 4, 128>},
                                other = 0 : i8},
             contract.program_axes = {varying = array<i64: 0>, invariant = array<i64: 1, 2>},
             contract.provenance = {source_load_origin_ids = array<i64: 1>}}
            : memref<4x128xi8, strided<[768, 1], offset: ?>> to memref<4x128xi8, strided<[128, 1]>>
        %22 = arith.addi %20, %18 : index
        %b_view_tail = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [128, 128],
            strides: [1, 768] : memref<*xi8> to memref<128x128xi8, strided<[1, 768], offset: ?>>
        %b_flat_tail = memref.alloc() alignment = 64
            {contract.buffer = {role = "tile", kind = "input-b",
                                logical_shape = array<i64: 128, 128>}}
            : memref<16384xi8>
        %b_2d_tail = memref.reinterpret_cast %b_flat_tail to offset: [0], sizes: [128, 128],
            strides: [1, 128] : memref<16384xi8> to memref<128x128xi8, strided<[1, 128]>>
        memref.copy %b_view_tail, %b_2d_tail
            {contract.task = {id = 9 : i64, kind = "memory-load", scope = 4 : i64,
                              stage = 0 : i64, predecessors = array<i64>,
                              input_values = array<i64: 1>, output_values = array<i64: 13>,
                              barrier_after = true, rematerializable = false,
                              origin_id = 2 : i64},
             contract.access = {kind = "load", boundary = "external_ddr",
                                origin_id = 2 : i64, mask = {kind = "none"},
                                other = 0 : i8},
             contract.program_axes = {varying = array<i64: 1>, invariant = array<i64: 0, 2>},
             contract.provenance = {source_load_origin_ids = array<i64: 2>}}
            : memref<128x128xi8, strided<[1, 768], offset: ?>> to memref<128x128xi8, strided<[1, 128]>>
        linalg.matmul
            {cast = #linalg.type_fn<cast_signed>,
             contract.task = {id = 10 : i64, kind = "dot", scope = 4 : i64,
                              stage = 1 : i64, predecessors = array<i64: 6, 7, 8, 9>,
                              input_values = array<i64: 12, 13, 15>,
                              output_values = array<i64: 14>,
                              barrier_after = true, rematerializable = false,
                              origin_id = 3 : i64},
             contract.compute = {kind = "dot", signed = true, lhs = "i8",
                                 rhs = "i8", acc = "i32"},
             contract.program_axes = {varying = array<i64: 0, 1>, invariant = array<i64: 2>},
             contract.provenance = {origin_id = 3 : i64,
                                    source_load_origin_ids = array<i64: 0, 1, 2>,
                                    a_origin_id = 1 : i64, b_origin_id = 2 : i64,
                                    a_source_load_origin_ids = array<i64: 1>,
                                    b_source_load_origin_ids = array<i64: 2>,
                                    accumulator_source_load_origin_ids = array<i64: 0>,
                                    a_source_lineage_complete = true,
                                    b_source_lineage_complete = true,
                                    accumulator_source_lineage_complete = true}}
            ins(%a_buf_tail, %b_2d_tail : memref<32x128xi8>, memref<128x128xi8, strided<[1, 128]>>)
            outs(%c_buf_tail : memref<32x128xi32>)
      } {contract.cf = {kind = "uniform_for", trip_count = 6 : i64,
                        uniform = true, serial = true},
         contract.scope = {id = 4 : i64, kind = "for-body", parent = 2 : i64,
                           complete = true},
         contract.value_ids = array<i64: 15, 16>,
         contract.loop_carried_slots = [{slot = 0 : i64,
             init_value_id = 10 : i64, iter_arg_value_id = 15 : i64,
             yielded_value_id = 14 : i64, result_value_id = 16 : i64,
             init_producer_compute_ids = array<i64: 6>,
             iter_arg_consumer_compute_ids = array<i64: 10>,
             backedge_producer_compute_ids = array<i64: 10>,
             exit_consumer_compute_ids = array<i64: 11>}]}

      %res4 = memref.subview %c_buf_tail[0, 0] [4, 128] [1, 1]
          : memref<32x128xi32> to memref<4x128xi32, strided<[128, 1]>>
      memref.copy %res4, %c_src4
          {contract.task = {id = 11 : i64, kind = "memory-store", scope = 2 : i64,
                            stage = 2 : i64, predecessors = array<i64: 6, 10>,
                            input_values = array<i64: 16>, output_values = array<i64>,
                            barrier_after = true, rematerializable = false,
                            origin_id = 4 : i64},
           contract.access = {kind = "store", boundary = "external_ddr",
                              origin_id = 4 : i64,
                              mask = {kind = "rectangular",
                                      valid_shape = array<i64: 4, 128>}},
           contract.program_axes = {varying = array<i64: 0, 1>, invariant = array<i64: 2>},
           contract.provenance = {source_load_origin_ids = array<i64: 0, 1, 2>}}
          : memref<4x128xi32, strided<[128, 1]>> to memref<4x128xi32, strided<[768, 1], offset: ?>>

    } {contract.cf = {kind = "uniform_if", condition_axes = array<i64: 0>,
                      arms_exclusive = true, arm_logical_tile = array<i64: 32, 128>},
       contract.scopes = [{id = 1 : i64, kind = "if-arm", arm = "then",
                           parent = 0 : i64, arms_exclusive = true,
                           condition_axes = array<i64: 0>, complete = true},
                          {id = 2 : i64, kind = "if-arm", arm = "else",
                           parent = 0 : i64, arms_exclusive = true,
                           condition_axes = array<i64: 0>, complete = true}]}
    return
  }
}
