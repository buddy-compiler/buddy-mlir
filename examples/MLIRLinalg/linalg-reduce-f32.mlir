// RUN: buddy-opt -reduce-vectorize="vector-size=16" -verify-diagnostics -lower-affine -expand-strided-metadata -convert-vector-to-scf -convert-vector-to-llvm -finalize-memref-to-llvm -convert-scf-to-cf -convert-arith-to-llvm -convert-func-to-llvm -lower-affine -llvm-request-c-wrappers -convert-arith-to-llvm -reconcile-unrealized-casts %s \
// RUN: | mlir-cpu-runner -O0 -e buddy_reduce_f32 -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

// 创建一个12x40x48的输入张量
memref.global "private" @A : memref<12x40x48xf32> = dense<1.0>

func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @buddy_reduce_f32() {
  %a = memref.get_global @A : memref<12x40x48xf32>
  %b = memref.alloc() : memref<12x40xf32>  // 输出张量，对最后一维进行规约

  %t_start = call @rtclock() : () -> f64

  // 初始化输出张量为0
  %c0 = arith.constant 0.0 : f32
  linalg.fill ins(%c0 : f32) outs(%b : memref<12x40xf32>)

  // 对最后一维进行规约操作
  linalg.reduce 
      ins(%a: memref<12x40x48xf32>)
      outs(%b: memref<12x40xf32>)
      dimensions = [2]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  %printed_b = memref.cast %b : memref<12x40xf32> to memref<*xf32>
  call @printMemrefF32(%printed_b) : (memref<*xf32>) -> ()
  vector.print %time : f64
  memref.dealloc %b : memref<12x40xf32>
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[12, 40\] strides = \[40, 1\] data =}}
  return
}
