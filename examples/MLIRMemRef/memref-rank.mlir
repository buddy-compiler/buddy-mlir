func @main() {
    %mem0 = memref.alloc() : memref<4xf32>
    %mem1 = memref.alloc() : memref<4x4xf32>
    %mem2 = memref.cast %mem1 : memref<4x4xf32> to memref<*xf32>
    %i0 = memref.rank %mem0 : memref<4xf32>
    %i1 = memref.rank %mem1 : memref<4x4xf32>
    %i2 = memref.rank %mem2 : memref<*xf32>
    vector.print %i0 : index
    vector.print %i1 : index
    vector.print %i2 : index
    memref.dealloc %mem0 : memref<4xf32>
    memref.dealloc %mem1 : memref<4x4xf32>
    return 
}