// Explicit optional-kernel boundary in a Buddy computation. The compiler can
// insert this call at an eligible operator; automatic rewriting is future work.
module {
  func.func private @buddy_softmax_dispatch(memref<2x4xf32>, memref<2x4xf32>, i64, !llvm.ptr) -> i32 attributes {llvm.emit_c_interface}
  func.func @buddy_computation(%input: memref<2x4xf32>, %output: memref<2x4xf32>, %axis: i64, %session: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface} {
    %status = func.call @buddy_softmax_dispatch(%input, %output, %axis, %session) : (memref<2x4xf32>, memref<2x4xf32>, i64, !llvm.ptr) -> i32
    return %status : i32
  }
}
