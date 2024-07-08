module {
  memref.global "private" constant @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32> = dense<8.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x1x40x40xf32 : memref<1x1x40x40xf32> = dense<4.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_32x128x40xf32 : memref<32x128x40xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_32x40x128xf32 : memref<32x40x128xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x32x40x40xf32 : memref<1x32x40x40xf32> = dense<11.3137083> {alignment = 64 : i64}
  func.func @veckenerl(%arg0: tensor<32x40x128xf32>, %arg1: tensor<32x128x40xf32>, %arg2: tensor<1x1x40x40xf32>, %arg3: tensor<1x32x40x128xf32>) {
    %cst = arith.constant 0.0883883461 : f32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant -3.40282347E+38 : f32
    %0 = bufferization.to_memref %arg3 : memref<1x32x40x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg2 : memref<1x1x40x40xf32, strided<[?, ?, ?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg1 : memref<32x128x40xf32, strided<[?, ?, ?], offset: ?>>
    %3 = bufferization.to_memref %arg0 : memref<32x40x128xf32, strided<[?, ?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x40x40xf32>
    
    // 0（cst_0）-> alloc
    affine.parallel (%arg4) = (0) to (32) {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 40 step 40{
          %t1 = vector.broadcast %cst_0 : f32 to vector<40xf32>
          vector.transfer_write %t1,%alloc[%arg4, %arg5, %arg6] : vector<40xf32> , memref<32x40x40xf32>
        }
      }
    }

    affine.parallel (%arg4) = (0) to (32) {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 40 step 40{
          affine.for %arg7 = 0 to 128 {
            %c1 = arith.constant 0.000000e+00 : f32
            %t1 = vector.transfer_read %3[%arg4, %arg5, %arg7],%c1 {permutation_map = affine_map<(d0,d1,d2)->(0)> } : memref<32x40x128xf32, strided<[?, ?, ?], offset: ?>> , vector<40xf32>
            %c2 = arith.constant 0.000000e+00 : f32
            %t2 = vector.transfer_read %2[%arg4, %arg7, %arg6],%c2 : memref<32x128x40xf32, strided<[?, ?, ?], offset: ?>> , vector<40xf32>
            %c3 = arith.constant 0.000000e+00 : f32
            %t3 = vector.transfer_read %alloc[%arg4, %arg5, %arg6],%c3 : memref<32x40x40xf32> , vector<40xf32>
            %r1 = arith.mulf %t1, %t2 : vector<40xf32>
            %r2 = arith.addf %r1, %t3 : vector<40xf32>
            vector.transfer_write %r2, %alloc[%arg4, %arg5, %arg6] : vector<40xf32>, memref<32x40x40xf32>
          }
        }
      }
    }

    %expand_shape = memref.expand_shape %alloc [[0, 1], [2], [3]] : memref<32x40x40xf32> into memref<1x32x40x40xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.parallel (%arg5) = (0) to (32) {
        affine.for %arg6 = 0 to 40 step 40{
          %t1 = vector.broadcast %cst_2 : f32 to vector<40xf32>
          vector.transfer_write %t1,%alloc_6[%arg4, %arg5, %arg6] : vector<40xf32> , memref<1x32x40xf32>
        }
      }
    }
    
    // 0.0883(cst)->alloc_3   0.883 = 11.313.reciporocal
    // mul->alloc_4
    // add->alloc_5
    //reduce_max->alloc_6
    affine.for %arg4 = 0 to 1 {
      affine.parallel (%arg5) = (0) to (32) {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 step 40{
            //mul 
            %t1 = vector.broadcast %cst : f32 to vector<40xf32>
            %c1 = arith.constant 0.000000e+00 : f32 
            %5 = vector.transfer_read %expand_shape[%c0, %arg5, %arg6, %arg7],%c1 : memref<1x32x40x40xf32> , vector<40xf32>
            %7 = arith.mulf %5, %t1 : vector<40xf32>
            //%5 = affine.load %expand_shape[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            //%7 = arith.mulf %5, %cst : f32

            //add
            %c2 = arith.constant 0.000000e+00 : f32 
            %9 = vector.transfer_read %1[%c0, %c0, %arg6, %arg7],%c2 : memref<1x1x40x40xf32, strided<[?, ?, ?, ?], offset: ?>> , vector<40xf32>
            %10 = arith.addf %7, %9 : vector<40xf32>
            vector.transfer_write %10, %alloc_5[%arg4, %arg5, %arg6, %arg7] : vector<40xf32>, memref<1x32x40x40xf32>
            //%9 = affine.load %1[%c0, %c0, %arg6, %arg7] : memref<1x1x40x40xf32, strided<[?, ?, ?, ?], offset: ?>>
            //%10 = arith.addf %7, %9 : f32
            //affine.store %10, %alloc_5[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>

            //reduce_max
            %11 = vector.reduction<maxf>, %10 : vector<40xf32> into f32
            %13 = arith.cmpf ugt, %11, %cst_2 : f32
            %14 = arith.select %13, %11, %cst_2 : f32
            affine.store %14, %alloc_6[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
            //%12 = affine.load %alloc_6[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
            //%13 = arith.cmpf ugt, %10, %12 : f32
            //%14 = arith.select %13, %10, %12 : f32
            //%15 = arith.cmpf uno, %12, %12 : f32
            //%16 = arith.select %15, %12, %14 : f32
            //affine.store %16, %alloc_6[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
          }
        }
      }
    }


    %expand_shape_7 = memref.expand_shape %alloc_6 [[0], [1], [2, 3]] : memref<1x32x40xf32> into memref<1x32x40x1xf32>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.parallel (%arg5) = (0) to (32) {
        affine.for %arg6 = 0 to 40 step 40{
          %t1 = vector.broadcast %cst_0 : f32 to vector<40xf32>
          vector.transfer_write %t1,%alloc_10[%arg4, %arg5, %arg6] : vector<40xf32> , memref<1x32x40xf32>
          //affine.store %cst_0, %alloc_10[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
        }
      }
    }
    // sub->alloc_8
    affine.for %arg4 = 0 to 1 {
      affine.parallel (%arg5) = (0) to (32) {
        affine.for %arg6 = 0 to 40 {
          affine.for %arg7 = 0 to 40 step 40{
            //sub
            %c1 = arith.constant 0.000000e+00 : f32
            %5 = vector.transfer_read %alloc_5[%c0, %arg5, %arg6, %arg7],%c1 : memref<1x32x40x40xf32>,vector<40xf32>
            %c2 = arith.constant 0.000000e+00 : f32
            %6 = vector.transfer_read %expand_shape_7[%c0, %arg5, %arg6, %c0],%c2 : memref<1x32x40x1xf32>, vector<40xf32>
            //%5 = affine.load %alloc_5[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            //%6 = affine.load %expand_shape_7[%c0, %arg5, %arg6, %c0] : memref<1x32x40x1xf32>
            %7 = arith.subf %5, %6 : vector<40xf32>

            //exp
            %9 = math.exp %7 : vector<40xf32>
            vector.transfer_write %9,%alloc_9[%arg4, %arg5, %arg6, %arg7] : vector<40xf32> , memref<1x32x40x40xf32>
            //affine.store %9, %alloc_9[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            
            //reduce_sum
            %10 = vector.reduction<add>,%9 : vector<40xf32> into f32
            //%11 = affine.load %alloc_10[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
            //%12 = arith.addf %10, %11 : f32
            affine.store %10, %alloc_10[%arg4, %arg5, %arg6] : memref<1x32x40xf32>
          }
        }
      }
    }

    // Fusion: Reciprocal + Multiplication
    %expand_shape_11 = memref.expand_shape %alloc_10 [[0], [1], [2, 3]] : memref<1x32x40xf32> into memref<1x32x40x1xf32>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x40xf32>
    affine.for %arg4 = 0 to 1 {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 40 {
          // Fusion point: reciprocal
          %5 = affine.load %expand_shape_11[%c0, %arg5, %arg6, %c0] : memref<1x32x40x1xf32>
          %6 = arith.divf %cst_1, %5 : f32
          affine.for %arg7 = 0 to 40 {
            // Fusion point: multiplication
            %7 = affine.load %alloc_9[%c0, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
            %8 = arith.mulf %6, %7 : f32
            affine.store %8, %alloc_13[%arg4, %arg5, %arg6, %arg7] : memref<1x32x40x40xf32>
          }
        }
      }
    }

    // reshape✖️2
    %collapse_shape = memref.collapse_shape %alloc_13 [[0, 1], [2], [3]] : memref<1x32x40x40xf32> into memref<32x40x40xf32>
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x32x40x128xf32>
    memref.copy %0, %alloc_14 : memref<1x32x40x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x32x40x128xf32>
    %collapse_shape_15 = memref.collapse_shape %alloc_14 [[0, 1], [2], [3]] : memref<1x32x40x128xf32> into memref<32x40x128xf32>
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32x40x128xf32>
    affine.parallel (%arg4) = (0) to (32) {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 128 step 128{
          %t1 = vector.broadcast %cst_0 : f32 to vector<128xf32>
          vector.transfer_write %t1, %alloc_16[%arg4, %arg5, %arg6] : vector<128xf32>, memref<32x40x128xf32>
          //affine.store %cst_0, %alloc_16[%arg4, %arg5, %arg6] : memref<32x40x128xf32>
        }
      }
    }
    // matmul->alloc_16
    affine.parallel (%arg4) = (0) to (32) {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 128 step 128{
          affine.for %arg7 = 0 to 40 {
            %c1 = arith.constant 0.000000e+00 : f32
            %t1 = vector.transfer_read %collapse_shape[%arg4, %arg5, %arg7],%c1 {permutation_map = affine_map<(d0,d1,d2)->(0)> } : memref<32x40x40xf32> , vector<128xf32>
            %c2 = arith.constant 0.000000e+00 : f32
            %t2 = vector.transfer_read %collapse_shape_15[%arg4, %arg7, %arg6],%c2 : memref<32x40x128xf32> , vector<128xf32>
            %c3 = arith.constant 0.000000e+00 : f32
            %t3 = vector.transfer_read %alloc_16[%arg4, %arg5, %arg6],%c3 : memref<32x40x128xf32> , vector<128xf32>
            %r1 = arith.mulf %t1, %t2 : vector<128xf32>
            %r2 = arith.addf %r1, %t3 : vector<128xf32>
            vector.transfer_write %r2, %alloc_16[%arg4, %arg5, %arg6] : vector<128xf32>, memref<32x40x128xf32>
          }
        }
      }
    }
    
    return
  }
  func.func @main3() {
    %0 = memref.get_global @__constant_32x40x128xf32 : memref<32x40x128xf32>
    %1 = bufferization.to_tensor %0 : memref<32x40x128xf32>
    %2 = memref.get_global @__constant_32x128x40xf32 : memref<32x128x40xf32>
    %3 = bufferization.to_tensor %2 : memref<32x128x40xf32>
    %4 = memref.get_global @__constant_1x1x40x40xf32 : memref<1x1x40x40xf32>
    %5 = bufferization.to_tensor %4 : memref<1x1x40x40xf32>
    %6 = memref.get_global @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32>
    %7 = bufferization.to_tensor %6 : memref<1x32x40x128xf32>
    call @veckenerl(%1, %3, %5, %7) : (tensor<32x40x128xf32>, tensor<32x128x40xf32>, tensor<1x1x40x40xf32>, tensor<1x32x40x128xf32>) -> ()
    return
  }
  
}

