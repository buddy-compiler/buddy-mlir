// RUN: split-file %s %t
// RUN: buddy-opt %t/loop.mlir --kernel-contract-annotate --kernel-contract-verify | FileCheck %s --check-prefix=LOOP
// RUN: buddy-opt %t/views.mlir --kernel-contract-annotate --kernel-contract-verify | FileCheck %s --check-prefix=VIEWS
// RUN: buddy-opt %t/staging.mlir --kernel-contract-annotate --kernel-contract-verify | FileCheck %s --check-prefix=STAGING
// RUN: buddy-opt %t/zero-trip.mlir --kernel-contract-annotate --kernel-contract-verify | FileCheck %s --check-prefix=INCOMPLETE
// RUN: buddy-opt %t/private-merge.mlir --kernel-contract-annotate --kernel-contract-verify | FileCheck %s --check-prefix=MERGE
// RUN: not buddy-opt %t/overlap.mlir --kernel-contract-annotate 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not buddy-opt %t/shifted-init.mlir --kernel-contract-annotate 2>&1 | FileCheck %s --check-prefix=SHIFT
// RUN: not buddy-opt %t/hidden-target.mlir --kernel-contract-annotate 2>&1 | FileCheck %s --check-prefix=HIDDEN
// RUN: not buddy-opt %t/hidden-tensor.mlir --kernel-contract-annotate 2>&1 | FileCheck %s --check-prefix=TENSOR
// RUN: buddy-opt %S/Inputs/kernel.contract.mlir --kernel-contract-verify -o /dev/null
// RUN: buddy-opt %S/Inputs/matmul128.contract.mlir --kernel-contract-verify -o /dev/null
// RUN: buddy-opt %S/Inputs/layernorm.contract.mlir --kernel-contract-verify -o /dev/null
// RUN: buddy-opt %S/Inputs/kernel.contract.mlir --kernel-contract-annotate --kernel-contract-verify -o /dev/null
// RUN: buddy-opt %S/Inputs/matmul128.contract.mlir --kernel-contract-annotate --kernel-contract-verify -o /dev/null
// RUN: buddy-opt %S/Inputs/layernorm.contract.mlir --kernel-contract-annotate --kernel-contract-verify -o /dev/null
//
// Annotate rebuilds contract.* metadata; verify checks the rebuilt graph.
// Cases cover loops, views, planner staging, zero-trip incompleteness, private
// merge, and hard errors (overlap, shifted init, hidden target/tensor).
// Frozen Inputs/*.contract.mlir must remain annotate+verify clean.
//

// Full kills need WAW ordering across both loop boundaries, but do not read
// the old version. Static versions describe distinct dynamic loop instances.
// LOOP: contract.compute_graph = {{.*}}complete = true
// LOOP: linalg.fill {{.*}}input_values = array<i64>, {{.*}}predecessors = array<i64>
// LOOP: linalg.fill {{.*}}input_values = array<i64>, {{.*}}predecessors = array<i64: 0>
// LOOP: backedge_producer_compute_ids = array<i64: 1>
// LOOP-SAME: exit_consumer_compute_ids = array<i64: 2>
// LOOP-SAME: iter_arg_consumer_compute_ids = array<i64: 1>
// LOOP: linalg.fill {{.*}}input_values = array<i64>, {{.*}}predecessors = array<i64: 1>

// VIEWS: contract.compute_graph = {{.*}}complete = true
// VIEWS: linalg.generic
// STAGING: source_lineage_complete = false
// STAGING: contract.provenance = {generated_by = "planner-staging", source_lineage_complete = false}
// STAGING-NOT: origin_id
// INCOMPLETE: contract.compute_graph = {{.*}}complete = false
// INCOMPLETE-NOT: contract.loop_carried_slots
// INCOMPLETE-NOT: contract.value_ids
// INCOMPLETE: contract.scope = {{.*}}complete = false
// MERGE: contract.compute_graph = {{.*}}complete = false
// MERGE: parent private content crossing if is outside v1
// OVERLAP: kernel contract V25: overlapping views must have identical element addresses
// SHIFT: kernel contract V22/V23: read or partial inheritance has no initialized content version
// HIDDEN: kernel contract V14: target attributes are forbidden
// TENSOR: kernel contract V3: contract must be buffer form, without tensor types

//--- loop.mlir
module {
  func.func @kill_boundaries() {
    %zero = arith.constant 0 : i32
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %step = arith.constant 1 : index
    %c = memref.alloc() : memref<4xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    scf.for %i = %lb to %ub step %step {
      linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    }
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    return
  }
}

//--- views.mlir
module {
  func.func @disjoint_byte_views() {
    %zero = arith.constant 0 : i8
    %b0 = arith.constant 0 : index
    %b32 = arith.constant 32 : index
    %bytes = memref.alloc() : memref<64xi8>
    linalg.fill ins(%zero : i8) outs(%bytes : memref<64xi8>)
    %a = memref.view %bytes[%b0][] : memref<64xi8> to memref<4xf32>
    %b = memref.view %bytes[%b32][] : memref<64xi8> to memref<4xf32>
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : memref<4xf32>, memref<4xf32>) outs(%a : memref<4xf32>) {
    ^bb0(%x: f32, %y: f32, %old: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }
    return
  }
}

//--- staging.mlir
module {
  func.func @generated_access(%source: memref<4xi32>) {
    %snapshot = memref.alloc() : memref<4xi32>
    memref.copy %source, %snapshot {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}},
      contract.provenance = {generated_by = "planner-staging"}}
      : memref<4xi32> to memref<4xi32>
    return
  }
}

//--- zero-trip.mlir
module {
  func.func @unmodeled_zero_trip() {
    %zero = arith.constant 0 : i32
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    %c = memref.alloc() : memref<4xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    scf.for %i = %lb to %lb step %step {
      linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    }
    return
  }
}

//--- private-merge.mlir
module {
  func.func @unmodeled_private_merge(
      %pid: i32 {contract.program = {role = "program_id", axis = 0 : i32}}) {
    %zero = arith.constant 0 : i32
    %c = memref.alloc() : memref<4xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    %condition = arith.cmpi eq, %pid, %zero : i32
    scf.if %condition {
      linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    } else {
      linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    }
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    return
  }
}

//--- overlap.mlir
module {
  func.func @overlapping_byte_views() {
    %zero = arith.constant 0 : i8
    %b0 = arith.constant 0 : index
    %b4 = arith.constant 4 : index
    %bytes = memref.alloc() : memref<64xi8>
    linalg.fill ins(%zero : i8) outs(%bytes : memref<64xi8>)
    %a = memref.view %bytes[%b0][] : memref<64xi8> to memref<4xf32>
    %b = memref.view %bytes[%b4][] : memref<64xi8> to memref<4xf32>
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : memref<4xf32>, memref<4xf32>) outs(%a : memref<4xf32>) {
    ^bb0(%x: f32, %y: f32, %old: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }
    return
  }
}

//--- shifted-init.mlir
module {
  func.func @shift_does_not_initialize_root() {
    %zero = arith.constant 0 : i8
    %shift = arith.constant 1 : index
    %bytes = memref.alloc() : memref<64xi8>
    %shifted = memref.view %bytes[%shift][] : memref<64xi8> to memref<64xi8>
    linalg.fill ins(%zero : i8) outs(%shifted : memref<64xi8>)
    return
  }
}

//--- hidden-target.mlir
module attributes {test.hidden = {nested = [{bosc_ame.store_policy = "direct"}]}} {
  func.func @hidden(%x: memref<4xi32>) { return }
}

//--- hidden-tensor.mlir
module {
  func.func @hidden(%x: memref<4xi32> {test.hidden = {payload = tensor<4xf32>}}) {
    return
  }
}
