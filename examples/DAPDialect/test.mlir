memref.global "private" @kernel_0 : memref<6xf32> = dense<[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]>
memref.global "private" @input_audio3 : memref<20xf32> = dense<1.0>


// func.func @buddy_iir(%in : memref<?xf32>, %filter : memref<?x?xf32>, %out : memref<?xf32>) -> () {
//   dap.iir %in, %filter, %out : memref<?xf32>, memref<?x?xf32>, memref<?xf32>
//   return
// } 

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @main() -> () {
  %krn0 = memref.get_global @kernel_0 : memref<6xf32>
  %data0 = memref.get_global @input_audio3 : memref<20xf32>
  %output0 = memref.alloc() : memref<20xf32>
  %krn = memref.cast %krn0 : memref<6xf32> to memref<?xf32>
  %data = memref.cast %data0: memref<20xf32> to memref<?xf32>
  %output = memref.cast %output0: memref<20xf32> to memref<?xf32>
  dap.biquad %data, %krn, %output : memref<?xf32>, memref<?xf32>, memref<?xf32>
  %print_output = memref.cast %output : memref<?xf32> to memref<*xf32>
  func.call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()
  return
}