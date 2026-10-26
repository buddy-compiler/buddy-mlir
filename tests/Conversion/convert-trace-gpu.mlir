// RUN: buddy-opt %s -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -ownership-based-buffer-deallocation -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-trace-to-llvm="gpu" -gpu-async-region | FileCheck %s

module {
  func.func @trace_gpu(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    buddy_trace.start {cycle = true, tensor = true, id = 0 : i64, tag = "gpu"}
    %0 = buddy_trace.end %arg0 {cycle = true, tensor = true, id = 0 : i64, tag = "gpu"} : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @trace_gpu
// CHECK: call @buddyTraceCycleStartPath
// CHECK: call @buddyTraceCycleEndPath
// CHECK: %[[HOST:.*]] = memref.alloc() : memref<4xf32>
// CHECK: gpu.memcpy async
// CHECK: gpu.wait [
// CHECK: call @buddyTraceTensorF32Path
// CHECK: memref.dealloc %[[HOST]]
// CHECK-NOT: buddy_trace.
