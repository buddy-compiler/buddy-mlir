// RUN: rax-pack %s -o %t.rax
// RUN: rax-inspect %t.rax | FileCheck %s

rhal.module @stage1a attributes {version = "0.1.0"} {
  rhal.constant @weights {id = 7 : i32, storage = "external",
                          type = tensor<4xf32>, uri = "file:weights.bin"}

  rhal.codeobj @stage0 {id = 10 : i32, kind = "host_shared_lib",
                        backend = "cpu", uri = "file:kernels.so",
                        entry_symbol = "kernel_a"}
  rhal.codeobj @stage1 {id = 11 : i32, kind = "host_shared_lib",
                        backend = "cpu", uri = "file:kernels.so",
                        entry_symbol = "kernel_b"}
  rhal.codeobj @stage2 {id = 12 : i32, kind = "host_shared_lib",
                        backend = "cpu", uri = "file:kernels.so"}

  rhal.buffer @input {space = "host", type = tensor<4xf32>}
  rhal.buffer @scratch {space = "dram", type = tensor<4xf32>}
  rhal.buffer @output {space = "host", type = tensor<4xf32>}
  rhal.buffer @cos {space = "dram", type = tensor<4xf32>}
  rhal.buffer @sin {space = "dram", type = tensor<4xf32>}

  rhal.func @legacy {inputs = ["input"], outputs = ["output"],
                     dispatch = "stage0", args = ["input", "output"]}

  rhal.func @pipeline {inputs = ["input"], outputs = ["output"]} body {
    rhal.dispatch @stage0 [@weights, @input, @scratch]
    rhal.dispatch @stage1 [@scratch, @weights, @output]
  }

  rhal.func @ordered {inputs = ["input"], outputs = ["scratch"]} body {
    rhal.dispatch @stage0 [@input]
    rhal.dispatch @stage1 [@scratch]
    rhal.collective [@input, @scratch] {
      kind = "all_reduce",
      reduction = "sum"
    }
    rhal.dispatch @stage2 [@scratch]
  }

  rhal.func @broadcast {inputs = ["output", "cos"], outputs = ["sin"]} body {
    rhal.collective [@output, @cos, @sin] {
      kind = "broadcast",
      root = 2 : i32
    }
  }

  rhal.func @all_gatherv {inputs = ["input"], outputs = ["scratch"]} body {
    rhal.collective [@input] {
      kind = "all_gatherv",
      output_buffers = [@scratch],
      recv_counts = array<i64: 2, 3>,
      displacements = array<i64: 0, 2>
    }
  }

  rhal.func @reduce_scatter {inputs = ["output"], outputs = ["cos"]} body {
    rhal.collective [@output] {
      kind = "reduce_scatter",
      output_buffers = [@cos],
      recv_counts = array<i64: 2, 2>,
      reduction = "sum"
    }
  }
}

// CHECK: code_objects: 3
// CHECK-NEXT: [10] @stage0  kind=HostSharedLib  uri=file:kernels.so  entry_symbol=kernel_a
// CHECK-NEXT: [11] @stage1  kind=HostSharedLib  uri=file:kernels.so  entry_symbol=kernel_b
// CHECK-NEXT: [12] @stage2  kind=HostSharedLib  uri=file:kernels.so
// CHECK: functions: 6
// CHECK-NEXT: @legacy
// CHECK-NEXT: [0] Dispatch code_object_id=10 args=[buffer:1, buffer:3]
// CHECK-NEXT: [1] Barrier
// CHECK-NEXT: @pipeline
// CHECK-NEXT: [0] Dispatch code_object_id=10 args=[constant:7, buffer:1, buffer:2]
// CHECK-NEXT: [1] Dispatch code_object_id=11 args=[buffer:2, constant:7, buffer:3]
// CHECK-NEXT: @ordered
// CHECK-NEXT: [0] Dispatch code_object_id=10 args=[buffer:1]
// CHECK-NEXT: [1] Dispatch code_object_id=11 args=[buffer:2]
// CHECK-NEXT: [2] Collective kind=AllReduce reduction=Sum operands=[1->1, 2->2]
// CHECK-NEXT: [3] Dispatch code_object_id=12 args=[buffer:2]
// CHECK-NEXT: @broadcast
// CHECK-NEXT: [0] Collective kind=Broadcast root=2 operands=[3->3, 4->4, 5->5]
// CHECK-NEXT: @all_gatherv
// CHECK-NEXT: [0] Collective kind=AllGatherV operands=[1->2 recv_counts=[2,3] displacements=[0,2]]
// CHECK-NEXT: @reduce_scatter
// CHECK-NEXT: [0] Collective kind=ReduceScatter reduction=Sum operands=[3->4 recv_counts=[2,2]]
