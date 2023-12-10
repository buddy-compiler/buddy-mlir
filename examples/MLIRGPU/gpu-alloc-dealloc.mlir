module attributes {gpu.container_module} {
  func.func @main(%size : index) {
    %0 = gpu.wait async
    %1, %2 = gpu.alloc async [%0] (%size) : memref<?xf32>
    %3 = gpu.dealloc async [%2] %1 : memref<?xf32>
    gpu.wait [%3]
    return
  }
  func.func @alloc_sync(%size : index) {
    %0 = gpu.alloc host_shared (%size) : memref<?xf32>
    %1 = gpu.wait async
    %2 = gpu.dealloc async [%1] %0 : memref<?xf32>
    gpu.wait [%2]
    return
  }
}
