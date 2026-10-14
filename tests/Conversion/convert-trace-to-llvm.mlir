// RUN: buddy-opt %s -convert-trace-to-llvm | FileCheck %s

module {
  func.func @trace_f32(%arg0: memref<2x2xf32>) -> memref<2x2xf32> {
    %0 = buddy_trace.end %arg0 {cycle = true, tensor = true, id = 3 : i64, id_path = [3, 0], tag = "f32"} : memref<2x2xf32>
    return %0 : memref<2x2xf32>
  }

  func.func @trace_bf16(%arg0: memref<2x2xbf16>) -> memref<2x2xbf16> {
    %0 = buddy_trace.end %arg0 {cycle = true, tensor = true, id = 7 : i64, id_path = [7, 0], tag = "bf16"} : memref<2x2xbf16>
    return %0 : memref<2x2xbf16>
  }

  func.func @trace_i8(%arg0: memref<2x2xi8>) -> memref<2x2xi8> {
    %0 = buddy_trace.end %arg0 {cycle = true, tensor = true, id = 11 : i64, id_path = [11, 0], tag = "i8"} : memref<2x2xi8>
    return %0 : memref<2x2xi8>
  }

  func.func @trace_i32(%arg0: memref<2x2xi32>) -> memref<2x2xi32> {
    %0 = buddy_trace.end %arg0 {cycle = true, tensor = true, id = 13 : i64, id_path = [13, 0], tag = "i32"} : memref<2x2xi32>
    return %0 : memref<2x2xi32>
  }
  func.func @cycle_only(%arg0: memref<2xf32>) -> memref<2xf32> {
    buddy_trace.start {id = 20 : i64, cycle = true, tensor = false}
    %0 = buddy_trace.end %arg0 {id = 20 : i64, cycle = true, tensor = false} : memref<2xf32>
    return %0 : memref<2xf32>
  }
  func.func @tensor_only(%arg0: memref<2xf32>) -> memref<2xf32> {
    buddy_trace.start {id = 21 : i64, cycle = false, tensor = true}
    %0 = buddy_trace.end %arg0 {id = 21 : i64, cycle = false, tensor = true} : memref<2xf32>
    return %0 : memref<2xf32>
  }
  func.func @disabled(%arg0: index) -> index {
    buddy_trace.start {id = 22 : i64, cycle = false, tensor = false}
    %0 = buddy_trace.end %arg0 {id = 22 : i64, cycle = false, tensor = false} : index
    return %0 : index
  }

}

// CHECK-DAG:  func.func private @buddyTraceCycleEndPath(i64, i64, i64, i64, i64, i64) attributes {llvm.emit_c_interface}
// CHECK-DAG:  func.func private @buddyTraceCycleStartPath(i64, i64, i64, i64, i64, i64) attributes {llvm.emit_c_interface}
// CHECK-DAG:  func.func private @buddyTraceTensorBF16Path(i64, i64, i64, i64, i64, i64, memref<?xbf16>) attributes {llvm.emit_c_interface}
// CHECK-DAG:  func.func private @buddyTraceTensorF32Path(i64, i64, i64, i64, i64, i64, memref<?xf32>) attributes {llvm.emit_c_interface}
// CHECK-DAG:  func.func private @buddyTraceTensorI8Path(i64, i64, i64, i64, i64, i64, memref<?xi8>) attributes {llvm.emit_c_interface}
// CHECK-DAG:  func.func private @buddyTraceTensorI32Path(i64, i64, i64, i64, i64, i64, memref<?xi32>) attributes {llvm.emit_c_interface}

// CHECK-LABEL: func.func @trace_f32
// CHECK:       call @buddyTraceCycleEndPath
// CHECK:       call @buddyTraceTensorF32Path

// CHECK-LABEL: func.func @trace_bf16
// CHECK:       call @buddyTraceCycleEndPath
// CHECK:       call @buddyTraceTensorBF16Path

// CHECK-LABEL: func.func @trace_i8
// CHECK:       call @buddyTraceCycleEndPath
// CHECK:       call @buddyTraceTensorI8Path

// CHECK-LABEL: func.func @trace_i32
// CHECK:       call @buddyTraceCycleEndPath
// CHECK:       call @buddyTraceTensorI32Path

// CHECK-LABEL: func.func @cycle_only
// CHECK: call @buddyTraceCycleStartPath
// CHECK: call @buddyTraceCycleEndPath
// CHECK-NOT: call @buddyTraceTensor
// CHECK: return
// CHECK-LABEL: func.func @tensor_only
// CHECK-NOT: call @buddyTraceCycle
// CHECK: call @buddyTraceTensorF32Path
// CHECK-NOT: call @buddyTraceCycle
// CHECK: return
// CHECK-LABEL: func.func @disabled
// CHECK-NOT: call
// CHECK-NOT: buddy_trace.
// CHECK: return %arg0
