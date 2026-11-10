// LayerNorm 197x768: manually authored design fixture, not compiler output.
// One program handles one row. Logical width=1024, valid prefix=768, EPS=1e-12.
// 21 tasks, 22 content versions; squares has its own root and no initializer.
#identity = affine_map<(d0) -> (d0)>

module attributes {
  contract.version = 1 : i32, contract.producer = "triton-riscv",
  contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
} {
  func.func @layernorm_197x768(
      %external_x: memref<*xf32> {contract.value_id = 0 : i64},
      %external_weight: memref<*xf32> {contract.value_id = 1 : i64},
      %external_bias: memref<*xf32> {contract.value_id = 2 : i64},
      %external_out: memref<*xf32> {contract.value_id = 3 : i64},
      %pid0: i32 {contract.program = {role = "program_id", axis = 0 : i32}})
      attributes {
        contract.grid = array<i32: 197, 1, 1>,
        contract.compute_graph = {version = 1 : i32, root_scope = 0 : i64,
            complete = true, scope_count = 1 : i64, source_lineage_complete = true}
      } {
    %zero = arith.constant 0.000000e+00 : f32
    %one = arith.constant 1.000000e+00 : f32
    %width_f32 = arith.constant 7.680000e+02 : f32
    %epsilon = arith.constant 1.000000e-12 : f32
    %width = arith.constant 768 : index
    %row = arith.index_cast %pid0 : i32 to index
    %row_offset = arith.muli %row, %width : index

    %x_view = memref.reinterpret_cast %external_x to offset: [%row_offset], sizes: [768], strides: [1]
        : memref<*xf32> to memref<768xf32, strided<[1], offset: ?>>
    %weight_view = memref.reinterpret_cast %external_weight to offset: [0], sizes: [768], strides: [1]
        : memref<*xf32> to memref<768xf32, strided<[1]>>
    %bias_view = memref.reinterpret_cast %external_bias to offset: [0], sizes: [768], strides: [1]
        : memref<*xf32> to memref<768xf32, strided<[1]>>
    %out_view = memref.reinterpret_cast %external_out to offset: [%row_offset], sizes: [768], strides: [1]
        : memref<*xf32> to memref<768xf32, strided<[1], offset: ?>>

    %x = memref.alloc() alignment = 64 {contract.buffer = {role = "tile", kind = "temporary", logical_shape = array<i64: 1024>}} : memref<1024xf32>
    %weight = memref.alloc() alignment = 64 {contract.buffer = {role = "tile", kind = "temporary", logical_shape = array<i64: 1024>}} : memref<1024xf32>
    %bias = memref.alloc() alignment = 64 {contract.buffer = {role = "tile", kind = "temporary", logical_shape = array<i64: 1024>}} : memref<1024xf32>
    %mean_vec = memref.alloc() alignment = 64 {contract.buffer = {role = "tile", kind = "temporary", logical_shape = array<i64: 1024>}} : memref<1024xf32>
    %squares = memref.alloc() alignment = 64 {contract.buffer = {role = "tile", kind = "temporary", logical_shape = array<i64: 1024>}} : memref<1024xf32>
    %invstd_vec = memref.alloc() alignment = 64 {contract.buffer = {role = "tile", kind = "temporary", logical_shape = array<i64: 1024>}} : memref<1024xf32>
    %mean_sum = memref.alloc() alignment = 64 {contract.buffer = {role = "scratch", kind = "temporary", logical_shape = array<i64>}} : memref<f32>
    %variance_sum = memref.alloc() alignment = 64 {contract.buffer = {role = "scratch", kind = "temporary", logical_shape = array<i64>}} : memref<f32>

    linalg.fill {contract.task = {id = 0 : i64, kind = "fill", scope = 0 : i64, stage = 0 : i64,
        predecessors = array<i64>, input_values = array<i64>, output_values = array<i64: 4>,
        barrier_after = false, rematerializable = false},
        contract.fill = {padding = true}}
        ins(%zero : f32) outs(%x : memref<1024xf32>)

    %x_prefix = memref.subview %x[0] [768] [1]
        : memref<1024xf32> to memref<768xf32, strided<[1]>>
    memref.copy %x_view, %x_prefix {contract.task = {id = 1 : i64, kind = "memory-load", scope = 0 : i64, stage = 0 : i64,
        predecessors = array<i64: 0>, input_values = array<i64: 0, 4>, output_values = array<i64: 5>,
        barrier_after = true, rematerializable = false},
        contract.access = {kind = "load", boundary = "external_ddr", origin_id = 0 : i64,
            mask = {kind = "rectangular", valid_shape = array<i64: 768>}, other = 0.000000e+00 : f32},
        contract.provenance = {source_load_origin_ids = array<i64: 0>}}
        : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1]>>

    linalg.fill {contract.task = {id = 2 : i64, kind = "fill", scope = 0 : i64, stage = 0 : i64,
        predecessors = array<i64>, input_values = array<i64>, output_values = array<i64: 6>,
        barrier_after = false, rematerializable = false},
        contract.fill = {padding = true}}
        ins(%one : f32) outs(%weight : memref<1024xf32>)

    %weight_prefix = memref.subview %weight[0] [768] [1]
        : memref<1024xf32> to memref<768xf32, strided<[1]>>
    memref.copy %weight_view, %weight_prefix {contract.task = {id = 3 : i64, kind = "memory-load", scope = 0 : i64, stage = 0 : i64,
        predecessors = array<i64: 2>, input_values = array<i64: 1, 6>, output_values = array<i64: 7>,
        barrier_after = true, rematerializable = false},
        contract.access = {kind = "load", boundary = "external_ddr", origin_id = 1 : i64,
            mask = {kind = "rectangular", valid_shape = array<i64: 768>}, other = 1.000000e+00 : f32},
        contract.provenance = {source_load_origin_ids = array<i64: 1>}}
        : memref<768xf32, strided<[1]>> to memref<768xf32, strided<[1]>>

    linalg.fill {contract.task = {id = 4 : i64, kind = "fill", scope = 0 : i64, stage = 0 : i64,
        predecessors = array<i64>, input_values = array<i64>, output_values = array<i64: 8>,
        barrier_after = false, rematerializable = false},
        contract.fill = {padding = true}}
        ins(%zero : f32) outs(%bias : memref<1024xf32>)

    %bias_prefix = memref.subview %bias[0] [768] [1]
        : memref<1024xf32> to memref<768xf32, strided<[1]>>
    memref.copy %bias_view, %bias_prefix {contract.task = {id = 5 : i64, kind = "memory-load", scope = 0 : i64, stage = 0 : i64,
        predecessors = array<i64: 4>, input_values = array<i64: 2, 8>, output_values = array<i64: 9>,
        barrier_after = true, rematerializable = false},
        contract.access = {kind = "load", boundary = "external_ddr", origin_id = 2 : i64,
            mask = {kind = "rectangular", valid_shape = array<i64: 768>}, other = 0.000000e+00 : f32},
        contract.provenance = {source_load_origin_ids = array<i64: 2>}}
        : memref<768xf32, strided<[1]>> to memref<768xf32, strided<[1]>>

    linalg.fill {contract.task = {id = 6 : i64, kind = "fill", scope = 0 : i64, stage = 0 : i64,
        predecessors = array<i64>, input_values = array<i64>, output_values = array<i64: 10>,
        barrier_after = false, rematerializable = false}}
        ins(%zero : f32) outs(%mean_sum : memref<f32>)

    linalg.reduce ins(%x : memref<1024xf32>) outs(%mean_sum : memref<f32>)
        dimensions = [0] {contract.task = {id = 7 : i64, kind = "reduce", scope = 0 : i64, stage = 1 : i64,
        predecessors = array<i64: 1, 6>, input_values = array<i64: 5, 10>, output_values = array<i64: 11>,
        barrier_after = true, rematerializable = false},
        contract.compute = {kind = "reduce", dimensions = array<i64: 0>}}
        (%value: f32, %acc: f32) {
      %next = arith.addf %value, %acc : f32
      linalg.yield %next : f32
    }

    %sum_for_mean = memref.load %mean_sum[] {contract.task = {id = 8 : i64, kind = "scalar-load", scope = 0 : i64, stage = 2 : i64,
        predecessors = array<i64: 7>, input_values = array<i64: 11>, output_values = array<i64>,
        barrier_after = false, rematerializable = false}} : memref<f32>

    %mean = arith.divf %sum_for_mean, %width_f32 : f32

    linalg.fill {contract.task = {id = 9 : i64, kind = "fill", scope = 0 : i64, stage = 2 : i64,
        predecessors = array<i64: 8>, input_values = array<i64>, output_values = array<i64: 12>,
        barrier_after = false, rematerializable = false}}
        ins(%mean : f32) outs(%mean_vec : memref<1024xf32>)

    linalg.generic {indexing_maps = [#identity, #identity, #identity], iterator_types = ["parallel"],
        contract.task = {id = 10 : i64, kind = "elementwise", scope = 0 : i64, stage = 2 : i64,
        predecessors = array<i64: 1, 7, 9>, input_values = array<i64: 5, 12>, output_values = array<i64: 13>,
        barrier_after = false, rematerializable = false},
        contract.compute = {kind = "elementwise"}}
        ins(%x, %mean_vec : memref<1024xf32>, memref<1024xf32>)
        outs(%x : memref<1024xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %unused_out: f32):
      %result = arith.subf %lhs, %rhs : f32
      linalg.yield %result : f32
    }

    // Restore the masked-off centered lanes before squaring or reducing.
    %centered_tail = memref.subview %x[768] [256] [1]
        : memref<1024xf32> to memref<256xf32, strided<[1], offset: 768>>
    linalg.fill {contract.task = {id = 11 : i64, kind = "fill", scope = 0 : i64, stage = 2 : i64,
        predecessors = array<i64: 10>, input_values = array<i64: 13>, output_values = array<i64: 14>,
        barrier_after = false, rematerializable = false}}
        ins(%zero : f32) outs(%centered_tail : memref<256xf32, strided<[1], offset: 768>>)

    // Fresh squares: unused output argument, no initialization or old-output read.
    linalg.generic {indexing_maps = [#identity, #identity, #identity], iterator_types = ["parallel"],
        contract.task = {id = 12 : i64, kind = "elementwise", scope = 0 : i64, stage = 2 : i64,
        predecessors = array<i64: 11>, input_values = array<i64: 14>, output_values = array<i64: 15>,
        barrier_after = false, rematerializable = false},
        contract.compute = {kind = "elementwise"}}
        ins(%x, %x : memref<1024xf32>, memref<1024xf32>)
        outs(%squares : memref<1024xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %unused_out: f32):
      %result = arith.mulf %lhs, %rhs : f32
      linalg.yield %result : f32
    }

    linalg.fill {contract.task = {id = 13 : i64, kind = "fill", scope = 0 : i64, stage = 0 : i64,
        predecessors = array<i64>, input_values = array<i64>, output_values = array<i64: 16>,
        barrier_after = false, rematerializable = false}}
        ins(%zero : f32) outs(%variance_sum : memref<f32>)

    linalg.reduce ins(%squares : memref<1024xf32>) outs(%variance_sum : memref<f32>)
        dimensions = [0] {contract.task = {id = 14 : i64, kind = "reduce", scope = 0 : i64, stage = 2 : i64,
        predecessors = array<i64: 12, 13>, input_values = array<i64: 15, 16>, output_values = array<i64: 17>,
        barrier_after = true, rematerializable = false},
        contract.compute = {kind = "reduce", dimensions = array<i64: 0>}}
        (%value: f32, %acc: f32) {
      %next = arith.addf %value, %acc : f32
      linalg.yield %next : f32
    }

    %sum_for_variance = memref.load %variance_sum[] {contract.task = {id = 15 : i64, kind = "scalar-load", scope = 0 : i64, stage = 3 : i64,
        predecessors = array<i64: 14>, input_values = array<i64: 17>, output_values = array<i64>,
        barrier_after = false, rematerializable = false}} : memref<f32>

    %variance = arith.divf %sum_for_variance, %width_f32 : f32
    %variance_eps = arith.addf %variance, %epsilon : f32
    %invstd = math.rsqrt %variance_eps : f32

    linalg.fill {contract.task = {id = 16 : i64, kind = "fill", scope = 0 : i64, stage = 3 : i64,
        predecessors = array<i64: 15>, input_values = array<i64>, output_values = array<i64: 18>,
        barrier_after = false, rematerializable = false}}
        ins(%invstd : f32) outs(%invstd_vec : memref<1024xf32>)

    // Squares consumed centered first; normalization is its last memory reader.
    linalg.generic {indexing_maps = [#identity, #identity, #identity], iterator_types = ["parallel"],
        contract.task = {id = 17 : i64, kind = "elementwise", scope = 0 : i64, stage = 3 : i64,
        predecessors = array<i64: 11, 12, 16>, input_values = array<i64: 14, 18>, output_values = array<i64: 19>,
        barrier_after = false, rematerializable = false},
        contract.compute = {kind = "elementwise"}}
        ins(%x, %invstd_vec : memref<1024xf32>, memref<1024xf32>)
        outs(%x : memref<1024xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %unused_out: f32):
      %result = arith.mulf %lhs, %rhs : f32
      linalg.yield %result : f32
    }

    linalg.generic {indexing_maps = [#identity, #identity, #identity], iterator_types = ["parallel"],
        contract.task = {id = 18 : i64, kind = "elementwise", scope = 0 : i64, stage = 3 : i64,
        predecessors = array<i64: 3, 17>, input_values = array<i64: 19, 7>, output_values = array<i64: 20>,
        barrier_after = false, rematerializable = false},
        contract.compute = {kind = "elementwise"}}
        ins(%x, %weight : memref<1024xf32>, memref<1024xf32>)
        outs(%x : memref<1024xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %unused_out: f32):
      %result = arith.mulf %lhs, %rhs : f32
      linalg.yield %result : f32
    }

    linalg.generic {indexing_maps = [#identity, #identity, #identity], iterator_types = ["parallel"],
        contract.task = {id = 19 : i64, kind = "elementwise", scope = 0 : i64, stage = 3 : i64,
        predecessors = array<i64: 5, 18>, input_values = array<i64: 20, 9>, output_values = array<i64: 21>,
        barrier_after = false, rematerializable = false},
        contract.compute = {kind = "elementwise"}}
        ins(%x, %bias : memref<1024xf32>, memref<1024xf32>)
        outs(%x : memref<1024xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %unused_out: f32):
      %result = arith.addf %lhs, %rhs : f32
      linalg.yield %result : f32
    }

    // The existing x_prefix view now observes x version 21, not the loaded version 5.
    memref.copy %x_prefix, %out_view {contract.task = {id = 20 : i64, kind = "memory-store", scope = 0 : i64, stage = 3 : i64,
        predecessors = array<i64: 19>, input_values = array<i64: 21>, output_values = array<i64>,
        barrier_after = true, rematerializable = false},
        contract.access = {kind = "store", boundary = "external_ddr", origin_id = 3 : i64,
            mask = {kind = "rectangular", valid_shape = array<i64: 768>}},
        contract.provenance = {source_load_origin_ids = array<i64: 0, 1, 2>}}
        : memref<768xf32, strided<[1]>> to memref<768xf32, strided<[1], offset: ?>>
    return
  }
}
