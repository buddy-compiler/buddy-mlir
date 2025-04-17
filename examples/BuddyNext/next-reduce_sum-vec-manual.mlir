// RUN: buddy-opt -reduce-vectorize="vector-size=16" -verify-diagnostics -lower-affine -expand-strided-metadata -convert-vector-to-scf -convert-vector-to-llvm -finalize-memref-to-llvm -convert-scf-to-cf -convert-arith-to-llvm -convert-func-to-llvm -lower-affine -llvm-request-c_wrappers -convert-arith-to-llvm -reconcile-unrealized-casts %s \
// RUN: | mlir-cpu-runner -O0 -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

// 创建一个12x40x40的输入张量
memref.global "private" @A : memref<12x40x40xf32> = dense<3.0>

func.func @kernel(%a : memref<12x40x40xf32>) {
  %t_start = call @rtclock() : () -> f64
  
  %b = memref.alloc() : memref<12x40xf32>  // 输出张量

  // 初始化常量
  %c0 = arith.constant 0.0 : f32
  %c16 = arith.constant 16 : index
  %c12 = arith.constant 12 : index
  %c40 = arith.constant 40 : index
  %c0_idx = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  // 使用step 1的外层循环和8x8分块
  affine.for %i0 = 0 to 12 step 1 {
    affine.for %j0 = 0 to 40 step 8 {
      // 使用1维并行处理
      affine.parallel (%idx) = (0) to (8) {
        // 计算j1
        %j1 = arith.remui %idx, %c8 : index
        
        %j = affine.apply affine_map<(d0, d1) -> (d0 + d1)> (%j0, %j1)
        
        // 检查是否在有效范围内
        %j_in_range = arith.cmpi slt, %j, %c40 : index
        
        // 只在有效范围内进行计算
        scf.if %j_in_range {
          // 初始化累加器
          %init_acc = arith.constant 0.0 : f32
          
          // 在k维度上使用16元素向量化
          %result_acc = affine.for %k = 0 to 40 step 16 iter_args(%acc = %init_acc) -> f32 {
            // 预取下一个数据块
            %next_k = arith.addi %k, %c16 : index
            %next_valid = arith.cmpi slt, %next_k, %c40 : index
            scf.if %next_valid {
              memref.prefetch %a[%i0, %j, %next_k], read, locality<3>, data : memref<12x40x40xf32>
            }
            
            // 计算当前块大小和掩码
            %remaining = arith.subi %c40, %k : index
            %vl = arith.minsi %remaining, %c16 : index
            %mask = vector.create_mask %vl : vector<16xi1>
            
            // 使用向量化读取数据
            %vec = vector.transfer_read %a[%i0, %j, %k], %c0, %mask : memref<12x40x40xf32>, vector<16xf32>
            
            // 向量规约求和
            %block_sum = vector.reduction <add>, %vec : vector<16xf32> into f32
            %next_acc = arith.addf %acc, %block_sum : f32
            affine.yield %next_acc : f32
          }

          // 写入结果
          memref.store %result_acc, %b[%i0, %j] : memref<12x40xf32>
        }
      }
    }
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // 打印结果
  %printed_b = memref.cast %b : memref<12x40xf32> to memref<*xf32>
  call @printMemrefF32(%printed_b) : (memref<*xf32>) -> ()
  
  // 打印时间
  vector.print %time : f64
  
  memref.dealloc %b : memref<12x40xf32>
  return
}

func.func @main() {
  %a = memref.get_global @A : memref<12x40x40xf32>
  call @kernel(%a) : (memref<12x40x40xf32>) -> ()
  return
}