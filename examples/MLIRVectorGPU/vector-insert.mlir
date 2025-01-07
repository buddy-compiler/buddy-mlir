module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_insert(%result0: memref<9xi32>, %result1: memref<9xi32>, %result2: memref<9xi32>, %the_base: memref<9xi32>) kernel{
      %base = arith.constant dense<[[0, 1, 2],
                                    [10, 11, 12],
                                    [20, 21, 22]]> : vector<3x3xi32>
      // insert a scalar
      %c0 = arith.constant 0 : index
      %src0 = arith.constant 100 : i32
      %v0 = vector.insert %src0, %base[0, 0] : i32 into vector<3x3xi32>
      %v0_shape_casted = vector.shape_cast %v0 : vector<3x3xi32> to vector<9xi32>
      vector.store %v0_shape_casted, %result0[%c0] : memref<9xi32>, vector<9xi32>
      
      // insert a vector
      %src1 = arith.constant dense<[101, 102, 103]> : vector<3xi32>
      %v1 = vector.insert %src1, %base[1] : vector<3xi32> into vector<3x3xi32>
      %v1_shape_casted = vector.shape_cast %v1 : vector<3x3xi32> to vector<9xi32>
      vector.store %v1_shape_casted, %result1[%c0] : memref<9xi32>, vector<9xi32>

      // insert a vector with exactly the same rank
      %src2 = arith.constant dense<[[201, 202, 203],
                                    [211, 212, 213],
                                    [221, 222, 223]]> : vector<3x3xi32>
      %v2 = vector.insert %src2, %base[] : vector<3x3xi32> into vector<3x3xi32>
      %v2_shape_casted = vector.shape_cast %v2 : vector<3x3xi32> to vector<9xi32>
      vector.store %v2_shape_casted, %result2[%c0] : memref<9xi32>, vector<9xi32>

      %the_base_shape_casted = vector.shape_cast %base : vector<3x3xi32> to vector<9xi32>
      vector.store %the_base_shape_casted, %the_base[%c0] : memref<9xi32>, vector<9xi32>

      gpu.return
    }
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result0 = memref.alloc() : memref<9xi32>
    %result1 = memref.alloc() : memref<9xi32>
    %result2 = memref.alloc() : memref<9xi32>
    %the_base = memref.alloc() : memref<9xi32>
    %result0_cast = memref.cast %result0 : memref<9xi32> to memref<*xi32>
    %result1_cast = memref.cast %result1 : memref<9xi32> to memref<*xi32>
    %result2_cast = memref.cast %result2 : memref<9xi32> to memref<*xi32>
    %the_base_cast = memref.cast %the_base : memref<9xi32> to memref<*xi32>

    gpu.host_register %result0_cast : memref<*xi32>
    gpu.host_register %result1_cast : memref<*xi32>
    gpu.host_register %result2_cast : memref<*xi32>
    gpu.host_register %the_base_cast : memref<*xi32>

    gpu.launch_func @kernels::@vector_insert blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%result0 : memref<9xi32>, %result1 : memref<9xi32>, %result2 : memref<9xi32>, %the_base : memref<9xi32>)

    %result0_v = vector.load %result0[%c0] : memref<9xi32>, vector<9xi32>
    %result0_v_reshape = vector.shape_cast %result0_v : vector<9xi32> to vector<3x3xi32>
    vector.print %result0_v_reshape : vector<3x3xi32>

    %result1_v = vector.load %result1[%c0] : memref<9xi32>, vector<9xi32>
    %result1_v_reshape = vector.shape_cast %result1_v : vector<9xi32> to vector<3x3xi32>
    vector.print %result1_v_reshape : vector<3x3xi32>

    %result2_v = vector.load %result2[%c0] : memref<9xi32>, vector<9xi32>
    %result2_v_reshape = vector.shape_cast %result2_v : vector<9xi32> to vector<3x3xi32>
    vector.print %result2_v_reshape : vector<3x3xi32>

    %the_base_v = vector.load %the_base[%c0] : memref<9xi32>, vector<9xi32>
    %the_base_v_reshape = vector.shape_cast %the_base_v : vector<9xi32> to vector<3x3xi32>
    vector.print %the_base_v_reshape : vector<3x3xi32>

    func.return
  }
}
