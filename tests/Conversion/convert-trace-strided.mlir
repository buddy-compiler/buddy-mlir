// RUN: buddy-opt %s -convert-trace-to-llvm | FileCheck %s
module {
  func.func @strided_slice(%arg: memref<2x3xf32, strided<[5, 1], offset: ?>>) -> memref<2x3xf32, strided<[5, 1], offset: ?>> {
    %0 = buddy_trace.end %arg {cycle = false, tensor = true, id = 0 : i64, id_path = [0], tag = "slice"} : memref<2x3xf32, strided<[5, 1], offset: ?>>
    return %0 : memref<2x3xf32, strided<[5, 1], offset: ?>>
  }
}
// CHECK-LABEL: func.func @strided_slice
// CHECK: %[[PACK:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK: memref.copy %arg0, %[[PACK]]
// CHECK: memref.collapse_shape %[[PACK]]
// CHECK: call @buddyTraceTensorF32Path
// CHECK: memref.dealloc %[[PACK]]
// CHECK: return %arg0
