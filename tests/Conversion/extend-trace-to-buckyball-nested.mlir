// RUN: buddy-opt %s -extend-trace-to-buckyball | FileCheck %s

module {
  func.func @trace_buckyball_nested(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>) -> memref<16x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    buddy_trace.start {id = 2 : i64, id_path = [2, 0], tag = "fc.linalg.matmul.0", trace_buckyball_include = ["matmul"], trace_extend_buckyball = true, trace_type = "linalg"}
    scf.for %i = %c0 to %c1 step %c1 {
      buckyball.matmul %arg0 %arg1 %arg2 : memref<16x16xf32> memref<16x16xf32> memref<16x16xf32>
    }
    %0 = buddy_trace.end %arg2 {id = 2 : i64, id_path = [2, 0], tag = "fc.linalg.matmul.0", trace_buckyball_include = ["matmul"], trace_extend_buckyball = true, trace_type = "linalg"} : memref<16x16xf32>
    return %0 : memref<16x16xf32>
  }
}

// CHECK-LABEL: func.func @trace_buckyball_nested
// CHECK:       buddy_trace.start
// CHECK-SAME:  id = 2
// CHECK-SAME:  trace_extended_buckyball = true
// CHECK:       scf.for
// CHECK:       buddy_trace.start
// CHECK-SAME:  generated = true
// CHECK-SAME:  id = 3
// CHECK-SAME:  id_path = [2, 0, 0]
// CHECK-SAME:  level = 2
// CHECK-SAME:  parent = 2
// CHECK-SAME:  trace_type = "buckyball"
// CHECK:       buckyball.matmul
// CHECK:       buddy_trace.end
// CHECK-SAME:  generated = true
// CHECK-SAME:  id = 3
// CHECK-SAME:  id_path = [2, 0, 0]
