module attributes {gpu.container_module} {
  func.func @main(%dst : memref<7xf32, 1>, %src : memref<7xf32>) {
    %t0 = gpu.wait async
    %t1 = gpu.memcpy async [%t0] %dst, %src : memref<7xf32, 1>, memref<7xf32>
    gpu.wait [%t1]
    return
  }
}
