#map = affine_map<(d0, d1) -> (d1, d0)>
module attributes {gpu.container_module} {
  func.func @subgraph1(%arg0: tensor<120x256xf32>, %arg1: tensor<84x120xf32>, %arg2: tensor<10x84xf32>) -> (memref<256x120xf32>, memref<120x84xf32>, memref<84x10xf32>) {
    %0 = bufferization.to_memref %arg0 : memref<120x256xf32>
    %1 = bufferization.to_memref %arg1 : memref<84x120xf32>
    %2 = bufferization.to_memref %arg2 : memref<10x84xf32>
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %3 = vector.transfer_read %0[%c0, %c0], %cst {permutation_map = #map} : memref<120x256xf32>, vector<256x120xf32>
    %alloc = memref.alloc() : memref<256x120xf32>
    vector.transfer_write %3, %alloc[%c0, %c0] : vector<256x120xf32>, memref<256x120xf32>
    %c0_0 = arith.constant 0 : index
    %cst_1 = arith.constant 0.000000e+00 : f32
    %4 = vector.transfer_read %1[%c0_0, %c0_0], %cst_1 {permutation_map = #map} : memref<84x120xf32>, vector<120x84xf32>
    %alloc_2 = memref.alloc() : memref<120x84xf32>
    vector.transfer_write %4, %alloc_2[%c0_0, %c0_0] : vector<120x84xf32>, memref<120x84xf32>
    %c0_3 = arith.constant 0 : index
    %cst_4 = arith.constant 0.000000e+00 : f32
    %5 = vector.transfer_read %2[%c0_3, %c0_3], %cst_4 {permutation_map = #map} : memref<10x84xf32>, vector<84x10xf32>
    %alloc_5 = memref.alloc() : memref<84x10xf32>
    vector.transfer_write %5, %alloc_5[%c0_3, %c0_3] : vector<84x10xf32>, memref<84x10xf32>
    return %alloc, %alloc_2, %alloc_5 : memref<256x120xf32>, memref<120x84xf32>, memref<84x10xf32>
  }
}

