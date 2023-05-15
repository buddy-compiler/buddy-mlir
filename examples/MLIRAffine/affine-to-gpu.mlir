// use affine dialect to write a vector add program

memref.global "private" @A: memref<16xf32> = dense<[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]>
memref.global "private" @B: memref<16xf32> = dense<[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]>

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
    %a = memref.get_global @A: memref<16xf32>
    %b = memref.get_global @B: memref<16xf32>
    %c = memref.alloca() : memref<16xf32>
    affine.for %idx0 = 0 to 16 {
        %1 = affine.load %a[%idx0] : memref<16xf32>
        %2 = affine.load %b[%idx0] : memref<16xf32>
        %3 = arith.addf %1, %2 : f32
        affine.store %3, %c[%idx0] : memref<16xf32>
    }
    %print_out = memref.cast %c : memref<16xf32> to memref<*xf32>
    func.call @printMemrefF32(%print_out) : (memref<*xf32>) -> ()
    func.return
}