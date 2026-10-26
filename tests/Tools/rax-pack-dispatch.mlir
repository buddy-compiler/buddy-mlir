// RUN: rax-pack %s -o %t.rax
// RUN: rax-inspect %t.rax | FileCheck %s
// RUN: sed 's/"params"/"missing"/g' %s > %t.invalid.mlir
// RUN: not rax-pack %t.invalid.mlir -o %t.invalid.rax 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: sed 's/id = 7 : i32/id = 0 : i32/' %s > %t.zero.mlir
// RUN: not rax-pack %t.zero.mlir -o %t.zero.rax 2>&1 | FileCheck %s --check-prefix=ZERO

// CHECK: buffers: 2
// CHECK: constants: 1
// CHECK: [7] @params
// CHECK: @kernels{{.*}}entry=_mlir_ciface_forward
// CHECK: @runtime{{.*}}uri=file:runtime.so{{$}}
// CHECK: functions: 1
// CHECK: @forward
// CHECK-NEXT: dispatch code_object=1 args=[buffer:2, constant:7, buffer:1]
// UNKNOWN: unknown buffer or constant 'missing' in args
// ZERO: constant 'params' uses reserved id 0

rhal.module @model attributes {version = "0.1.0", model_name = "test"} {
  // Bind by symbol rather than declaration order or a shared ID namespace.
  rhal.func @forward {
    inputs = ["image"], outputs = ["logits"], dispatch = "kernels",
    args = ["logits", "params", "image"]
  }
  rhal.buffer @image {space = "host", type = tensor<1x1x28x28xf32>}
  rhal.buffer @logits {space = "host", type = tensor<1x10xf32>}
  rhal.constant @params {
    id = 7 : i32, storage = "external",
    type = tensor<44426xf32>, uri = "file:arg0.data"
  }
  rhal.codeobj @kernels {
    id = 1 : i32, kind = "host_shared_lib", backend = "cpu",
    uri = "file:model.so", entry_symbol = "_mlir_ciface_forward"
  }
  rhal.codeobj @runtime {
    id = 2 : i32, kind = "host_shared_lib", backend = "cpu",
    uri = "file:runtime.so"
  }
}
