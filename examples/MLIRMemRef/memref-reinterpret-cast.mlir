module {

    func private @print_memref_f32(memref<*xf32>)
    func @main() {
        %mem0 = memref.alloc() : memref<8xf32, offset : 0, strides : [1]>
        %print_out0 = memref.cast %mem0 : memref<8xf32, offset : 0, strides : [1]> to memref<*xf32>
        call @print_memref_f32(%print_out0) : (memref<*xf32>) -> ()
        %mem1 = memref.reinterpret_cast %mem0 to 
        offset : [1],
        sizes  : [4, 2],
        strides : [1, 1]
        : memref<8xf32, offset : 0 , strides : [1]> to memref<4x2xf32, offset : 1 , strides : [1, 1]>
        %print_out1 = memref.cast %mem1 : memref<4x2xf32, offset : 1 , strides : [1, 1]> to memref<*xf32>
        call @print_memref_f32(%print_out1) : (memref<*xf32>) -> ()
        %c0 = arith.constant 2 : index
        %mem2 = memref.reinterpret_cast %mem1 to 
        offset : [0],
        sizes : [%c0, 4],
        strides : [1, %c0]
        : memref<4x2xf32, offset : 1 , strides : [1, 1]> to memref<?x4xf32, offset :0, strides : [1, ?]>
        %print_out2 = memref.cast %mem2 : memref<?x4xf32, offset :0, strides : [1, ?]> to memref<*xf32>
        call @print_memref_f32(%print_out2) : (memref<*xf32>) -> ()
        
        %c1 = arith.constant 0 : index
        %f0 = arith.constant 520. : f32 
        %f1 = arith.constant 1314. : f32
        affine.store %f1, %mem0[%c0] :  memref<8xf32, offset : 0, strides : [1]>
        affine.store %f0, %mem0[%c1] :  memref<8xf32, offset : 0, strides : [1]> 
        %print_out3 =  memref.cast %mem0 : memref<8xf32, offset : 0, strides : [1]> to memref<*xf32>
        %print_out4 =  memref.cast %mem1 : memref<4x2xf32, offset : 1 , strides : [1, 1]> to memref<*xf32>
        %print_out5 =  memref.cast %mem2 : memref<?x4xf32, offset :0, strides : [1, ?]> to memref<*xf32>
        call @print_memref_f32(%print_out3) : (memref<*xf32>) -> ()
        call @print_memref_f32(%print_out4) : (memref<*xf32>) -> ()
        call @print_memref_f32(%print_out5) : (memref<*xf32>) -> ()
        memref.dealloc  %mem0 : memref<8xf32, offset : 0, strides : [1]>
        return
    }
}