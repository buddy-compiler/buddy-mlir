// RUN: buddy-opt %s '-convert-trace-to-llvm="tensor-trace,cycle-trace"' | FileCheck %s

module {
  func.func @trace_f32(%arg0: memref<2x2xf32>) -> memref<2x2xf32> {
    %0 = buddy_trace.end %arg0 {id = 3 : i64, id_path = [3, 0], tag = "f32"} : memref<2x2xf32>
    return %0 : memref<2x2xf32>
  }

  func.func @trace_bf16(%arg0: memref<2x2xbf16>) -> memref<2x2xbf16> {
    %0 = buddy_trace.end %arg0 {id = 7 : i64, id_path = [7, 0], tag = "bf16"} : memref<2x2xbf16>
    return %0 : memref<2x2xbf16>
  }
}

// CHECK-DAG:  func.func private @buddyTraceCycleEndPath(i64, i64, i64, i64, i64, i64) attributes {llvm.emit_c_interface}
// CHECK-DAG:  func.func private @buddyTraceCycleStartPath(i64, i64, i64, i64, i64, i64) attributes {llvm.emit_c_interface}
// CHECK-DAG:  func.func private @buddyTraceTensorBF16Path(i64, i64, i64, i64, i64, i64, memref<?xbf16>) attributes {llvm.emit_c_interface}
// CHECK-DAG:  func.func private @buddyTraceTensorF32Path(i64, i64, i64, i64, i64, i64, memref<?xf32>) attributes {llvm.emit_c_interface}

// CHECK-LABEL: func.func @trace_f32
// CHECK:       call @buddyTraceTensorF32Path
// CHECK:       call @buddyTraceCycleEndPath

// CHECK-LABEL: func.func @trace_bf16
// CHECK:       call @buddyTraceTensorBF16Path
// CHECK:       call @buddyTraceCycleEndPath
