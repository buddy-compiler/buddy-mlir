// RUN: split-file %s %t
// RUN: not buddy-opt %t/unresolved.mlir --kernel-contract-verify 2>&1 | FileCheck %s --check-prefix=UNRESOLVED
// RUN: not buddy-opt %t/duplicate.mlir --kernel-contract-verify 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not buddy-opt %t/stale-read.mlir --kernel-contract-verify 2>&1 | FileCheck %s --check-prefix=STALE
// RUN: not buddy-opt %t/stale-backedge.mlir --kernel-contract-verify 2>&1 | FileCheck %s --check-prefix=BACKEDGE
// RUN: not buddy-opt %t/negative-origin.mlir --kernel-contract-annotate 2>&1 | FileCheck %s --check-prefix=ORIGIN
// RUN: not buddy-opt %t/empty-lineage.mlir --kernel-contract-annotate 2>&1 | FileCheck %s --check-prefix=LINEAGE

// Incomplete suppresses unproven graph claims, never schema/ID validation.
// UNRESOLVED: kernel contract V7/V15: unresolved content version, including in an incomplete graph
// DUPLICATE: kernel contract V7/V21: content version must have one nonnegative definition
// STALE: kernel contract V22/V23: input_values must match actual reads, latest versions
// BACKEDGE: kernel contract V11/V20/V26: loop slot must match actual init, backedge and exit consumers
// ORIGIN: kernel contract V12: source origin IDs must be nonnegative i64 arrays
// LINEAGE: kernel contract V12: complete provenance requires a nonempty source origin list

//--- unresolved.mlir
module attributes {contract.version = 1 : i32, contract.producer = "test"} {
  func.func @unresolved() attributes {
    contract.compute_graph = {version = 1 : i32, root_scope = 0 : i64,
      complete = false, scope_count = 1 : i64, limitation = "unmodeled boundary"}} {
    %zero = arith.constant 0 : i32
    %c = memref.alloc() : memref<4xi32>
    linalg.fill {contract.task = {id = 0 : i64, kind = "fill", scope = 0 : i64,
      stage = 0 : i64, predecessors = array<i64>, input_values = array<i64: 99>,
      output_values = array<i64: 0>, barrier_after = false, rematerializable = false}}
      ins(%zero : i32) outs(%c : memref<4xi32>)
    return
  }
}

//--- duplicate.mlir
module attributes {contract.version = 1 : i32, contract.producer = "test"} {
  func.func @duplicate() attributes {
    contract.compute_graph = {version = 1 : i32, root_scope = 0 : i64,
      complete = false, scope_count = 1 : i64, limitation = "unmodeled boundary"}} {
    %zero = arith.constant 0 : i32
    %c = memref.alloc() : memref<4xi32>
    linalg.fill {contract.task = {id = 0 : i64, kind = "fill", scope = 0 : i64,
      stage = 0 : i64, predecessors = array<i64>, input_values = array<i64>,
      output_values = array<i64: 0>, barrier_after = false, rematerializable = false}}
      ins(%zero : i32) outs(%c : memref<4xi32>)
    linalg.fill {contract.task = {id = 1 : i64, kind = "fill", scope = 0 : i64,
      stage = 0 : i64, predecessors = array<i64: 0>, input_values = array<i64>,
      output_values = array<i64: 0>, barrier_after = false, rematerializable = false}}
      ins(%zero : i32) outs(%c : memref<4xi32>)
    return
  }
}

//--- stale-read.mlir
module attributes {contract.version = 1 : i32, contract.producer = "test"} {
  func.func @stale_read() attributes {
    contract.compute_graph = {version = 1 : i32, root_scope = 0 : i64,
      complete = true, scope_count = 1 : i64}} {
    %zero = arith.constant 0 : i32
    %idx = arith.constant 0 : index
    %c = memref.alloc() : memref<4xi32>
    linalg.fill {contract.task = {id = 0 : i64, kind = "fill", scope = 0 : i64,
      stage = 0 : i64, predecessors = array<i64>, input_values = array<i64>,
      output_values = array<i64: 0>, barrier_after = false, rematerializable = false}}
      ins(%zero : i32) outs(%c : memref<4xi32>)
    linalg.fill {contract.task = {id = 1 : i64, kind = "fill", scope = 0 : i64,
      stage = 0 : i64, predecessors = array<i64: 0>, input_values = array<i64>,
      output_values = array<i64: 1>, barrier_after = false, rematerializable = false}}
      ins(%zero : i32) outs(%c : memref<4xi32>)
    %old = memref.load %c[%idx] {contract.task = {id = 2 : i64,
      kind = "scalar-load", scope = 0 : i64, stage = 0 : i64,
      predecessors = array<i64: 1>, input_values = array<i64: 0>,
      output_values = array<i64>, barrier_after = false, rematerializable = false}}
      : memref<4xi32>
    return
  }
}

//--- stale-backedge.mlir
module attributes {contract.version = 1 : i32, contract.producer = "test"} {
  func.func @stale_backedge() attributes {
    contract.compute_graph = {version = 1 : i32, root_scope = 0 : i64,
      complete = true, scope_count = 2 : i64}} {
    %zero = arith.constant 0 : i32
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %step = arith.constant 1 : index
    %c = memref.alloc() : memref<4xi32>
    linalg.fill {contract.task = {id = 0 : i64, kind = "fill", scope = 0 : i64,
      stage = 0 : i64, predecessors = array<i64>, input_values = array<i64>,
      output_values = array<i64: 0>, barrier_after = false, rematerializable = false}}
      ins(%zero : i32) outs(%c : memref<4xi32>)
    scf.for %i = %lb to %ub step %step {
      linalg.fill {contract.task = {id = 1 : i64, kind = "fill", scope = 1 : i64,
        stage = 0 : i64, predecessors = array<i64: 0>, input_values = array<i64>,
        output_values = array<i64: 3>, barrier_after = false, rematerializable = false}}
        ins(%zero : i32) outs(%c : memref<4xi32>)
      linalg.fill {contract.task = {id = 2 : i64, kind = "fill", scope = 1 : i64,
        stage = 0 : i64, predecessors = array<i64: 1>, input_values = array<i64>,
        output_values = array<i64: 4>, barrier_after = false, rematerializable = false}}
        ins(%zero : i32) outs(%c : memref<4xi32>)
    } {contract.scope = {id = 1 : i64, kind = "for-body", parent = 0 : i64, complete = true},
      contract.cf = {kind = "uniform_for", trip_count = 3 : i64, uniform = true, serial = true},
      contract.value_ids = array<i64: 1, 2>,
      contract.loop_carried_slots = [{slot = 0 : i64, init_value_id = 0 : i64,
        iter_arg_value_id = 1 : i64, yielded_value_id = 3 : i64, result_value_id = 2 : i64,
        init_producer_compute_ids = array<i64: 0>, iter_arg_consumer_compute_ids = array<i64: 1>,
        backedge_producer_compute_ids = array<i64: 1>, exit_consumer_compute_ids = array<i64>}]}
    return
  }
}

//--- negative-origin.mlir
module {
  func.func @negative_origin() attributes {
    contract.provenance = {source_store_origin_ids = array<i64: -1>}} {
    return
  }
}

//--- empty-lineage.mlir
module {
  func.func @empty_lineage() attributes {
    contract.provenance = {source_store_origin_ids = array<i64>,
      source_store_lineage_complete = true}} {
    return
  }
}
