module attributes {gpu.container_module} {
  func.func @foo() {
    %t0 = gpu.wait async
    %t1 = gpu.wait async [%t0]
    gpu.wait [%t0, %t1]
    return
  }
}