module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_reduction() kernel {
      %v0 = arith.constant dense<[12, 13, 14, 15, 16, 90]> : vector<6xi32>
      %sum = vector.reduction <add>, %v0 : vector<6xi32> into i32
      gpu.printf "sum: %d\n" %sum : i32
      %mul = vector.reduction <mul>, %v0 : vector<6xi32> into i32
      gpu.printf "mul: %d\n" %mul : i32
      %xor = vector.reduction <xor>, %v0 : vector<6xi32> into i32
      gpu.printf "xor: %d\n" %xor : i32
      %and = vector.reduction <and>, %v0 : vector<6xi32> into i32
      gpu.printf "and: %d\n" %and : i32
      %or = vector.reduction <or>, %v0 : vector<6xi32> into i32
      gpu.printf "or: %d\n" %or : i32

      %v1 = arith.constant dense<[1., 2., 3., 4., 5., 6.]> : vector<6xf32>
      %sum_f = vector.reduction <add>, %v1 : vector<6xf32> into f32
      gpu.printf "sum_f: %f\n" %sum_f : f32
      %mul_f = vector.reduction <mul>, %v1 : vector<6xf32> into f32
      gpu.printf "mul_f: %f\n" %mul_f : f32
      %min_f = vector.reduction <minimumf>, %v1 : vector<6xf32> into f32
      gpu.printf "min_f: %f\n" %min_f : f32
      %max_f = vector.reduction <maximumf>, %v1 : vector<6xf32> into f32
      gpu.printf "max_f: %f\n" %max_f : f32
      gpu.return
    }
  }

  func.func @main() {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@vector_reduction blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args()

    func.return 
  }
}
