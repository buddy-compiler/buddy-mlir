// RUN: buddy-opt %s -extend-trace-to-linalg | FileCheck %s
// RUN: buddy-opt %s -extend-trace-to-linalg \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" | \
// RUN:   FileCheck %s --check-prefix=CHECK-BUFFERIZE

module {
  func.func @trace_linalg(%arg0: tensor<1x4xf32>, %arg1: tensor<4x8xf32>) -> tensor<1x8xf32> {
    buddy_trace.start {id = 0 : i64, id_path = [0], tag = "fc", trace_buckyball_include = ["matmul"], trace_extend_buckyball = true, trace_extend_linalg = true, trace_linalg_include = ["matmul"]}
    %empty = tensor.empty() : tensor<1x8xf32>
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<1x4xf32>, tensor<4x8xf32>) outs(%empty : tensor<1x8xf32>) -> tensor<1x8xf32>
    %1 = buddy_trace.end %0 {id = 0 : i64, id_path = [0], tag = "fc", trace_buckyball_include = ["matmul"], trace_extend_buckyball = true, trace_extend_linalg = true, trace_linalg_include = ["matmul"]} : tensor<1x8xf32>
    return %1 : tensor<1x8xf32>
  }
}

// CHECK-LABEL: func.func @trace_linalg
// CHECK:       buddy_trace.start
// CHECK-SAME:  id = 0
// CHECK-SAME:  trace_extended_linalg = true
// CHECK-NOT:   trace_extend_buckyball
// CHECK:       buddy_trace.start
// CHECK-SAME:  generated = true
// CHECK-SAME:  id = 1
// CHECK-SAME:  id_path = [0, 0]

// CHECK-BUFFERIZE-LABEL: func.func @trace_linalg
// CHECK-BUFFERIZE:       buddy_trace.end
// CHECK-BUFFERIZE-SAME:  generated = true
// CHECK-BUFFERIZE-SAME:  id = 1
// CHECK-BUFFERIZE-SAME:  id_path = [0, 0]
// CHECK-BUFFERIZE-SAME:  level = 1
// CHECK-BUFFERIZE-SAME:  parent = 0
// CHECK-BUFFERIZE-SAME:  trace_type = "linalg"
// CHECK-SAME:  level = 1
// CHECK-SAME:  parent = 0
// CHECK-SAME:  trace_extend_buckyball = true
// CHECK-SAME:  trace_type = "linalg"
// CHECK:       linalg.matmul
// CHECK:       buddy_trace.end
// CHECK-SAME:  generated = true
// CHECK-SAME:  id = 1
// CHECK-SAME:  id_path = [0, 0]
