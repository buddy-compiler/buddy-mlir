#map = affine_map<(d0) -> ((d0 floordiv 16) * 16)>
#map1 = affine_map<(d0) -> (d0 mod 16)>
#map2 = affine_map<(d0) -> (0)>
#map3 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 mod 16 - 1 >= 0)>
module {
  memref.global "private" constant @__constant_1x32x40x120xf32 : memref<1x32x40x120xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @kernel(%arg0: memref<1x32x40x120xf32>) {
    %cast = memref.cast %arg0 : memref<1x32x40x120xf32> to memref<1x32x40x120xf32, strided<[?, ?, ?, ?], offset: ?>>
    %0 = call @rtclock() : () -> f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x40x32x120xf32>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<1x32x40x120xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg0, %c1 : memref<1x32x40x120xf32>
    %c2 = arith.constant 2 : index
    %dim_1 = memref.dim %arg0, %c2 : memref<1x32x40x120xf32>
    %c3 = arith.constant 3 : index
    %dim_2 = memref.dim %arg0, %c3 : memref<1x32x40x120xf32>
    %c0_3 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %1 = affine.apply #map(%dim_2)
    %2 = affine.apply #map1(%dim_2)
    %c1_4 = arith.constant 1 : index
    %c1_5 = arith.constant 1 : index
    %c1_6 = arith.constant 1 : index
    %3 = vector.create_mask %c1_4, %c1_5, %c1_6, %2 : vector<1x1x1x16xi1>
    %4 = vector.create_mask %c1_4, %c1_6, %c1_5, %2 : vector<1x1x1x16xi1>
    affine.for %arg1 = #map2(%c0_3) to #map3(%dim) {
      affine.for %arg2 = #map2(%c0_3) to #map3(%dim_0) {
        affine.for %arg3 = #map2(%c0_3) to #map3(%dim_1) {
          affine.for %arg4 = #map2(%c0_3) to #map3(%1) step 16 {
            %8 = vector.transfer_read %arg0[%arg1, %arg2, %arg3, %arg4], %cst {in_bounds = [true, true, true, true]} : memref<1x32x40x120xf32>, vector<1x1x1x16xf32>
            vector.transfer_write %8, %alloc[%arg1, %arg3, %arg2, %arg4] {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf32>, memref<1x40x32x120xf32>
            affine.if #set(%dim_2) {
              %9 = vector.transfer_read %arg0[%arg1, %arg2, %arg3, %1], %cst, %3 {in_bounds = [true, true, true, true]} : memref<1x32x40x120xf32>, vector<1x1x1x16xf32>
              vector.transfer_write %9, %alloc[%arg1, %arg3, %arg2, %1], %4 {in_bounds = [true, true, true, true]} : vector<1x1x1x16xf32>, memref<1x40x32x120xf32>
            }
          }
        }
      }
    }
    %5 = bufferization.to_tensor %alloc : memref<1x40x32x120xf32>
    %6 = call @rtclock() : () -> f64
    %7 = arith.subf %6, %0 : f64
    %cast_7 = memref.cast %alloc : memref<1x40x32x120xf32> to memref<*xf32>
    call @printMemrefF32(%cast_7) : (memref<*xf32>) -> ()
    vector.print %7 : f64
    return
  }
  func.func @main() {
    %0 = memref.get_global @__constant_1x32x40x120xf32 : memref<1x32x40x120xf32>
    call @kernel(%0) : (memref<1x32x40x120xf32>) -> ()
    return
  }
}

