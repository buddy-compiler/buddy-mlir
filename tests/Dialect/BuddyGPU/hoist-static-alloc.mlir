// RUN: buddy-opt --split-input-file --transform-interpreter %s | FileCheck %s

func.func @non_entry_bb_allocs() {
  cf.br ^bb1
 ^bb1() :
  %0 = memref.alloc() : memref<16xi32>
  memref.dealloc %0 : memref<16xi32>
  return
}
// CHECK-LABEL: func @non_entry_bb_allocs()
//  CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc() : memref<16xi32>
//  CHECK-NEXT:   memref.dealloc %[[ALLOC]] : memref<16xi32>
//  CHECK-NEXT:   cf.br ^bb1
//  CHECK-NEXT:   ^bb1:
//  CHECK-NEXT:   return

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.op<"func.func">
    transform.buddy.hoist_static_alloc %func : (!transform.op<"func.func">) -> ()
    transform.yield
  } // @__transform_main
} // module
