func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mem0 = memref.alloc() : memref<2x3xf32>
  %mem1 = memref.cast %mem0 : memref<2x3xf32> to memref<?x?xf32>
  %dim0 = memref.dim %mem0, %c0 : memref<2x3xf32>
  %dim1 = memref.dim %mem0, %c1 : memref<2x3xf32>
  %dim2 = memref.dim %mem1, %c0 : memref<?x?xf32>
  %dim3 = memref.dim %mem1, %c1 : memref<?x?xf32>
  vector.print %dim0 : index
  vector.print %dim1 : index
  vector.print %dim2 : index
  vector.print %dim3 : index  
  memref.dealloc %mem0 : memref<2x3xf32>
  func.return

}
