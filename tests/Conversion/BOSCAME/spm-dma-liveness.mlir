// RUN: split-file %s %t
// RUN: buddy-opt --allow-unregistered-dialect -verify-diagnostics %t/unknown-user.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt -verify-diagnostics %t/start-between-acc.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt -verify-diagnostics %t/wait-between-acc.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt -verify-diagnostics %t/store-before-reuse.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt -verify-diagnostics %t/loop-init-reuse.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt -verify-diagnostics %t/loop-capture-reuse.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt -verify-diagnostics %t/loop-duplicate-init.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt %t/stored-branches.mlir --bosc-ame-spm-dma-check -o /dev/null
// RUN: not buddy-opt %t/call-helper.mlir --bosc-ame-spm-dma-check 2>&1 | FileCheck %s --check-prefix=CALL
// RUN: not buddy-opt %t/call-helper.mlir --linalg-to-boscame 2>&1 | FileCheck %s --check-prefix=ADAPTER
// RUN: buddy-opt -verify-diagnostics %t/call-external.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt -verify-diagnostics %t/call-indirect.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt -verify-diagnostics %t/call-llvm.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt -verify-diagnostics %t/call-llvm-indirect.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt -verify-diagnostics %t/call-intrinsic.mlir --bosc-ame-spm-dma-check
// RUN: buddy-opt %t/calls-after-store.mlir --bosc-ame-spm-dma-check -o /dev/null
//
// SpmDma liveness: reject DMA/wait/reuse while an ACC is live, illegal loop
// captures, and calls during ACC lifetime; stored-then-call patterns stay legal.

// CALL: cannot call code while an AME accumulator is live
// The compatibility adapter accepts storage inputs before target computation
// emission. The dedicated checker above still verifies live-accumulator calls.
// ADAPTER: target compatibility accepts SPM windows and synchronous DMA only; already emitted instructions cannot be replanned

//--- unknown-user.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @unknown_user(%src: memref<64xi8>) {
    %bytes = arith.constant 64 : index
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %spm, %bytes {channel = 0 : i32}
        : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    // expected-error@+1 {{has unknown memory effects while a DMA request is in flight}}
    "test.hidden_memory_user"() : () -> ()
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- start-between-acc.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @start_between_acc(%src: memref<64xi8>) {
    %c0 = arith.constant 0 : index
    %bytes = arith.constant 64 : index
    %acc_bytes = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<1>>
    %acc = memref.view %acc_bytes[%c0][]
        : memref<64xi8, #bosc_ame.spm<0>>
          to memref<1x1xi32, #bosc_ame.spm<0>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    // expected-error@+1 {{cannot start a DMA request while an AME accumulator is live}}
    %t = bosc_ame.dma.start %src, %dma, %bytes {channel = 0 : i32}
        : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    bosc_ame.tile.store_acc %tc, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

//--- store-before-reuse.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @store_before_reuse(%src: memref<64xi8>) {
    %bytes = arith.constant 64 : index
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %a = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64}
        : memref<1x1xi8, #bosc_ame.spm<0>>
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<1>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    %ta = bosc_ame.tile.load_a %a
        : memref<1x1xi8, #bosc_ame.spm<0>> -> vector<1x1xi8>
    bosc_ame.tile.store_acc %tc, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    // The earlier store preserves %tc for its later MMA user.
    // expected-error@+1 {{cannot start a DMA request while an AME accumulator is live}}
    %t = bosc_ame.dma.start %src, %dma, %bytes {channel = 0 : i32}
        : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    %r = bosc_ame.tile.mma %tc, %ta, %ta
        : vector<1x1xi32>, vector<1x1xi8>, vector<1x1xi8>
          -> vector<1x1xi32>
    bosc_ame.tile.store_acc %r, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

//--- loop-init-reuse.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @loop_init_reuse(%src: memref<64xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %bytes = arith.constant 64 : index
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %a = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64}
        : memref<1x1xi8, #bosc_ame.spm<0>>
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<1>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    %ta = bosc_ame.tile.load_a %a
        : memref<1x1xi8, #bosc_ame.spm<0>> -> vector<1x1xi8>
    %out = scf.for %i = %c0 to %c2 step %c1
        iter_args(%iter = %tc) -> (vector<1x1xi32>) {
      %next = bosc_ame.tile.mma %iter, %ta, %ta
          : vector<1x1xi32>, vector<1x1xi8>, vector<1x1xi8>
            -> vector<1x1xi32>
      scf.yield %next : vector<1x1xi32>
    }
    bosc_ame.tile.store_acc %out, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    // The loop result is stored, but %tc still feeds another MMA.
    // expected-error@+1 {{cannot start a DMA request while an AME accumulator is live}}
    %t = bosc_ame.dma.start %src, %dma, %bytes {channel = 0 : i32}
        : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    %r = bosc_ame.tile.mma %tc, %ta, %ta
        : vector<1x1xi32>, vector<1x1xi8>, vector<1x1xi8>
          -> vector<1x1xi32>
    bosc_ame.tile.store_acc %r, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

//--- loop-capture-reuse.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @loop_capture_reuse(%src: memref<64xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %bytes = arith.constant 64 : index
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %a = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64}
        : memref<1x1xi8, #bosc_ame.spm<0>>
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<1>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    %ta = bosc_ame.tile.load_a %a
        : memref<1x1xi8, #bosc_ame.spm<0>> -> vector<1x1xi8>
    %out = scf.for %i = %c0 to %c2 step %c1
        iter_args(%iter = %tc) -> (vector<1x1xi32>) {
      bosc_ame.tile.store_acc %iter, %acc
          : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
      // The iter_arg is stored, but the loop captures %tc as well.
      // expected-error@+1 {{cannot start a DMA request while an AME accumulator is live}}
      %t = bosc_ame.dma.start %src, %dma, %bytes {channel = 0 : i32}
          : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index)
            -> !bosc_ame.dma_token<0>
      bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
      %next = bosc_ame.tile.mma %tc, %ta, %ta
          : vector<1x1xi32>, vector<1x1xi8>, vector<1x1xi8>
            -> vector<1x1xi32>
      scf.yield %next : vector<1x1xi32>
    }
    bosc_ame.tile.store_acc %out, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

//--- loop-duplicate-init.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @loop_duplicate_init(%src: memref<64xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %bytes = arith.constant 64 : index
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %a = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64}
        : memref<1x1xi8, #bosc_ame.spm<0>>
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<1>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    %ta = bosc_ame.tile.load_a %a
        : memref<1x1xi8, #bosc_ame.spm<0>> -> vector<1x1xi8>
    %out:2 = scf.for %i = %c0 to %c2 step %c1
        iter_args(%first = %tc, %second = %tc)
        -> (vector<1x1xi32>, vector<1x1xi32>) {
      bosc_ame.tile.store_acc %first, %acc
          : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
      // Storing the first iter_arg leaves the second one live.
      // expected-error@+1 {{cannot start a DMA request while an AME accumulator is live}}
      %t = bosc_ame.dma.start %src, %dma, %bytes {channel = 0 : i32}
          : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index)
            -> !bosc_ame.dma_token<0>
      bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
      %next = bosc_ame.tile.mma %second, %ta, %ta
          : vector<1x1xi32>, vector<1x1xi8>, vector<1x1xi8>
            -> vector<1x1xi32>
      scf.yield %next, %next : vector<1x1xi32>, vector<1x1xi32>
    }
    bosc_ame.tile.store_acc %out#0, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    bosc_ame.tile.store_acc %out#1, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

//--- stored-branches.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @stored_branches(%src: memref<64xi8>) {
    %bytes = arith.constant 64 : index
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %a = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64}
        : memref<1x1xi8, #bosc_ame.spm<0>>
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<1>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    %ta = bosc_ame.tile.load_a %a
        : memref<1x1xi8, #bosc_ame.spm<0>> -> vector<1x1xi8>
    %r0 = bosc_ame.tile.mma %tc, %ta, %ta
        : vector<1x1xi32>, vector<1x1xi8>, vector<1x1xi8>
          -> vector<1x1xi32>
    %r1 = bosc_ame.tile.mma %tc, %ta, %ta
        : vector<1x1xi32>, vector<1x1xi8>, vector<1x1xi8>
          -> vector<1x1xi32>
    bosc_ame.tile.store_acc %r0, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    bosc_ame.tile.store_acc %r1, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    // Both branches have ended, so DMA may run again.
    %t = bosc_ame.dma.start %src, %dma, %bytes {channel = 0 : i32}
        : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- wait-between-acc.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @wait_between_acc(%src: memref<64xi8>) {
    %c0 = arith.constant 0 : index
    %bytes = arith.constant 64 : index
    %acc_bytes = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<1>>
    %acc = memref.view %acc_bytes[%c0][]
        : memref<64xi8, #bosc_ame.spm<0>>
          to memref<1x1xi32, #bosc_ame.spm<0>>
    %a = memref.view %acc_bytes[%c0][]
        : memref<64xi8, #bosc_ame.spm<0>>
          to memref<1x1xi8, #bosc_ame.spm<0>>
    %b = memref.view %acc_bytes[%c0][]
        : memref<64xi8, #bosc_ame.spm<0>>
          to memref<1x1xi8, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dma, %bytes {channel = 0 : i32}
        : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index)
          -> !bosc_ame.dma_token<0>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    // expected-error@+1 {{cannot wait for a DMA request while an AME accumulator is live}}
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    %ta = bosc_ame.tile.load_a %a
        : memref<1x1xi8, #bosc_ame.spm<0>> -> vector<1x1xi8>
    %tb = bosc_ame.tile.load_b %b
        : memref<1x1xi8, #bosc_ame.spm<0>> -> vector<1x1xi8>
    %r = bosc_ame.tile.mma %tc, %ta, %tb
        : vector<1x1xi32>, vector<1x1xi8>, vector<1x1xi8>
          -> vector<1x1xi32>
    bosc_ame.tile.store_acc %r, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

//--- call-helper.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @caller(%src: memref<64xi8>) {
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    // The caller has no in-flight DMA, but its callee issues one.
    func.call @dma_helper(%src) : (memref<64xi8>) -> ()
    bosc_ame.tile.store_acc %tc, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
  func.func private @dma_helper(%src: memref<64xi8>) {
    %bytes = arith.constant 64 : index
    %dst = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<1>>
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- call-external.mlir
module {
  func.func private @external()
  func.func @caller() {
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    // expected-error@+1 {{cannot call code while an AME accumulator is live}}
    func.call @external() : () -> ()
    bosc_ame.tile.store_acc %tc, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

//--- call-indirect.mlir
module {
  func.func @caller(%callee: () -> ()) {
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    // expected-error@+1 {{cannot call code while an AME accumulator is live}}
    func.call_indirect %callee() : () -> ()
    bosc_ame.tile.store_acc %tc, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

//--- call-llvm.mlir
module {
  llvm.func @external()
  func.func @caller() {
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    // expected-error@+1 {{cannot call code while an AME accumulator is live}}
    llvm.call @external() : () -> ()
    bosc_ame.tile.store_acc %tc, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

//--- call-llvm-indirect.mlir
module {
  func.func @caller(%callee: !llvm.ptr) {
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    // expected-error@+1 {{cannot call code while an AME accumulator is live}}
    llvm.call %callee() : !llvm.ptr, () -> ()
    bosc_ame.tile.store_acc %tc, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

//--- call-intrinsic.mlir
module {
  func.func @caller(%value: f32) {
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    // expected-error@+1 {{cannot call code while an AME accumulator is live}}
    %root = llvm.call_intrinsic "llvm.sqrt"(%value) : (f32) -> f32
    bosc_ame.tile.store_acc %tc, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

//--- calls-after-store.mlir
module {
  func.func private @external()
  llvm.func @llvm_external()
  func.func @stored(%callee: () -> (), %llvm_callee: !llvm.ptr,
                    %value: f32) {
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<1x1xi32, #bosc_ame.spm<0>>
    %tc = bosc_ame.tile.load_acc %acc
        : memref<1x1xi32, #bosc_ame.spm<0>> -> vector<1x1xi32>
    bosc_ame.tile.store_acc %tc, %acc
        : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    // There is no later accumulator use; calls are safe again.
    func.call @external() : () -> ()
    func.call_indirect %callee() : () -> ()
    llvm.call @llvm_external() : () -> ()
    llvm.call %llvm_callee() : !llvm.ptr, () -> ()
    %root = llvm.call_intrinsic "llvm.sqrt"(%value) : (f32) -> f32
    return
  }
}
