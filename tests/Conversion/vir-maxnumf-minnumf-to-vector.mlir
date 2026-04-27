// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4' -cse | FileCheck %s

func.func @vir_maxnumf_to_vector(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
  %c128 = arith.constant 128 : index
  vir.set_vl %c128 : index {
    %a = vir.load %arg0[] : memref<128xf32> -> !vir.vec<?xf32>
    %b = vir.load %arg1[] : memref<128xf32> -> !vir.vec<?xf32>
    %max = arith.maxnumf %a, %b : !vir.vec<?xf32>
    vir.store %max, %arg0[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return
}

func.func @vir_minnumf_to_vector(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
  %c128 = arith.constant 128 : index
  vir.set_vl %c128 : index {
    %a = vir.load %arg0[] : memref<128xf32> -> !vir.vec<?xf32>
    %b = vir.load %arg1[] : memref<128xf32> -> !vir.vec<?xf32>
    %min = arith.minnumf %a, %b : !vir.vec<?xf32>
    vir.store %min, %arg0[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @vir_maxnumf_to_vector
// CHECK: arith.maxnumf {{.*}} : vector<4xf32>
// CHECK: arith.maxnumf {{.*}} : f32

// CHECK-LABEL: func.func @vir_minnumf_to_vector
// CHECK: arith.minnumf {{.*}} : vector<4xf32>
// CHECK: arith.minnumf {{.*}} : f32
