#map = affine_map<(d0, d1) -> (d0 + d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (0)>
#map3 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 mod 16 - 1 >= 0)>

module {
  memref.global "private" constant @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  
  // 通用的张量转置优化实现
  func.func @kernel(%arg0: memref<1x32x40x128xf32>) {
    %0 = call @rtclock() : () -> f64
    
    // 分配输出内存
    %alloc = memref.alloc() : memref<1x40x32x128xf32>
    
    // 初始化常量
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c40 = arith.constant 40 : index
    %c128 = arith.constant 128 : index
    %c8 = arith.constant 8 : index  // 使用更小的向量长度
    %f0 = arith.constant 0.0 : f32
    
    // 主循环 - 处理可以向量化的部分
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 40 {
        affine.for %k = 0 to 32 {
          // 处理可以8个元素向量化的部分
          affine.for %l = 0 to 128 step 8 {
            // 检查是否还有足够的元素进行向量化
            affine.if affine_set<(d0) : (128 - d0 >= 8)>(%l) {
              // 向量化读取和存储
              %vec = vector.transfer_read %arg0[%i, %k, %j, %l], %f0 : 
                memref<1x32x40x128xf32>, vector<8xf32>
              vector.transfer_write %vec, %alloc[%i, %j, %k, %l] : 
                vector<8xf32>, memref<1x40x32x128xf32>
            } else {
              // 处理剩余元素
              affine.for %r = 0 to 8 {
                %l_idx = affine.apply #map(%l, %r)
                // 确保不越界
                affine.if affine_set<(d0) : (128 - d0 >= 0)>(%l_idx) {
                  %val = memref.load %arg0[%i, %k, %j, %l_idx] : memref<1x32x40x128xf32>
                  memref.store %val, %alloc[%i, %j, %k, %l_idx] : memref<1x40x32x128xf32>
                }
              }
            }
          }
        }
      }
    }
    
    %1 = call @rtclock() : () -> f64
    %2 = arith.subf %1, %0 : f64
    
    // 打印结果
    %cast = memref.cast %alloc : memref<1x40x32x128xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    vector.print %2 : f64
    
    // 释放内存
    memref.dealloc %alloc : memref<1x40x32x128xf32>
    return
  }
  
  func.func @main() {
    %0 = memref.get_global @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32>
    call @kernel(%0) : (memref<1x32x40x128xf32>) -> ()
    return
  }
} 