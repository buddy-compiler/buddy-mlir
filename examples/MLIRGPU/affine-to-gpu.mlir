module {
func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
    %a = memref.alloc() : memref<8192xf32>
    %b = memref.alloc() : memref<8192xf32>
    %c = memref.alloc() : memref<8192xf32>
    %cf1 = arith.constant 1.0 : f32
    %cf2 = arith.constant 2.0 : f32
    linalg.fill
    ins(%cf1: f32)
    outs(%a: memref<8192xf32>)
    linalg.fill
    ins(%cf2: f32)
    outs(%b: memref<8192xf32>)
    %cast_a = memref.cast %a : memref<8192xf32> to memref<*xf32>
    %cast_b = memref.cast %b : memref<8192xf32> to memref<*xf32>
    %cast_c = memref.cast %c : memref<8192xf32> to memref<*xf32>
    gpu.host_register %cast_a : memref<*xf32>
    gpu.host_register %cast_b : memref<*xf32>
    gpu.host_register %cast_c : memref<*xf32>
    affine.for %idx0 = 0 to 8192 {
        %1 = affine.load %a[%idx0] : memref<8192xf32>
        %2 = affine.load %b[%idx0] : memref<8192xf32>
        %3 = arith.addf %1, %2 : f32
        affine.store %3, %c[%idx0] : memref<8192xf32>
    }
    // %print_out = memref.cast %c : memref<8192xf32> to memref<*xf32>
    func.call @printMemrefF32(%cast_c) : (memref<*xf32>) -> ()
    func.return
}
}