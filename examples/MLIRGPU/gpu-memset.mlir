module attributes {gpu.container_module} {
  func.func @main(%dst : memref<7xf32, 1>, %value : f32) {
    %t0 = gpu.wait async
    %t1 = gpu.memset async [%t0] %dst, %value : memref<7xf32, 1>, f32
    gpu.wait [%t1]
    return
  }
}