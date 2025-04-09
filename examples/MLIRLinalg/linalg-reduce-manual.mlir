// 定义全局输入张量 A，初始化为全 1.0
memref.global "private" @A : memref<12x40x48xf32> = dense<1.0>

// 声明外部函数
func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

// 向量化的规约函数
func.func @reduce_last_dim_vectorized() {
  %a = memref.get_global @A : memref<12x40x48xf32>  // 获取输入张量 A
  %b = memref.alloc() : memref<12x40xf32>           // 分配输出张量 B
  %c0 = arith.constant 0.0 : f32                    // 零值常量，用于填充

  // 测量开始时间
  %t_start = call @rtclock() : () -> f64

  // 外层循环：遍历 i 和 j
  affine.for %i = 0 to 12 {
    affine.for %j = 0 to 40 {
      // 初始化累加器
      %acc_init = arith.constant 0.0 : f32
      %acc = affine.for %k = 0 to 48 step 16 iter_args(%acc_iter = %acc_init) -> (f32) {
        // 读取 A[i][j][k..k+15] 作为一个向量
        %vec = vector.transfer_read %a[%i, %j, %k], %c0 {permutation_map = affine_map<(d0, d1, d2) -> (d2)>}
          : memref<12x40x48xf32>, vector<16xf32>

        // 对向量进行规约求和
        %vec_sum = vector.reduction <add>, %vec : vector<16xf32> into f32

        // 累加到当前结果
        %acc_next = arith.addf %acc_iter, %vec_sum : f32
        affine.yield %acc_next : f32
      }

      // 存储最终结果到 B[i][j]
      memref.store %acc, %b[%i, %j] : memref<12x40xf32>
    }
  }

  // 测量结束时间
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // 打印输出张量 B
  %printed_b = memref.cast %b : memref<12x40xf32> to memref<*xf32>
  call @printMemrefF32(%printed_b) : (memref<*xf32>) -> ()

  // 打印执行时间
  vector.print %time : f64

  // 释放内存
  memref.dealloc %b : memref<12x40xf32>
  return
}

// 主函数入口
func.func @main() {
  call @reduce_last_dim_vectorized() : () -> ()
  return
}