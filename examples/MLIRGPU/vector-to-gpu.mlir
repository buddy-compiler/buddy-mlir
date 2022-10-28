#map0 = affine_map<(i, j, k) -> (i, k)>
#map1 = affine_map<(i, j, k) -> (k, j)>
#map2 = affine_map<(i, j, k) -> (i, j)> 
#map3 = affine_map<(i, j) -> (j, i) >
module attributes {gpu.container_module} {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index 
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index 
    %c16 = arith.constant 16 : index
    %f2 = arith.constant 2.0 : f16
    %f0f32 = arith.constant 0.0 : f32
    %input0 = memref.alloc() : memref<16x16xf16>
    %input1 = memref.alloc() : memref<16x16xf16>
    %output0 = memref.alloc() : memref<16x16xf32>
    %input0_cast = memref.cast %input0 : memref<16x16xf16> to memref<*xf16>
    %input1_cast = memref.cast %input1 : memref<16x16xf16> to memref<*xf16>
    %output0_cast = memref.cast %output0 : memref<16x16xf32> to memref<*xf32>
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %f0f32, %output0[%i, %j] : memref<16x16xf32>
        memref.store %f2, %input0[%i, %j] : memref<16x16xf16>
        memref.store %f2, %input1[%i, %j] : memref<16x16xf16>
      }
    }  
    call @printMemrefF32(%output0_cast) : (memref<*xf32>) -> ()
    gpu.host_register %input0_cast : memref<*xf16>
    gpu.host_register %input1_cast : memref<*xf16>
    gpu.host_register %output0_cast : memref<*xf32>
    gpu.launch_func @kernels::@kernel1 blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%input0: memref<16x16xf16>, %input1 : memref<16x16xf16>, %output0 : memref<16x16xf32>)
    call @printMemrefF32(%output0_cast) : (memref<*xf32>) -> ()
    memref.dealloc %input0 : memref<16x16xf16>
    memref.dealloc %input1 : memref<16x16xf16>
    memref.dealloc %output0 : memref<16x16xf32>
    return 
  } 
  gpu.module @kernels { 
    gpu.func @kernel1(%arg0 : memref<16x16xf16>, %arg1 : memref<16x16xf16>, %arg2 : memref<16x16xf32>) kernel {
        %f0 = arith.constant 0.0 : f16
        %f0f32 = arith.constant 0.0 : f32
        %c0 = arith.constant 0 : index
        %arg_v0 = vector.transfer_read %arg0[%c0, %c0], %f0 : memref<16x16xf16>, vector<16x16xf16>
        %arg_v1 = vector.transfer_read %arg1[%c0, %c0], %f0 : memref<16x16xf16>, vector<16x16xf16>
        %arg_v2 = vector.transfer_read %arg2[%c0, %c0], %f0f32 : memref<16x16xf32>, vector<16x16xf32>
        %result = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %arg_v0, %arg_v1, %arg_v2 : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
        vector.transfer_write %result, %arg2[%c0, %c0] : vector<16x16xf32>, memref<16x16xf32>
        gpu.return
      }
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}

