module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_extract() kernel {
      
      %base = arith.constant dense<[[0, 1, 2],
                                    [10, 11, 12],
                                    [20, 21, 22]]> : vector<3x3xi32>

      %c0 = vector.extract %base[1, 1] : i32 from vector<3x3xi32>
      gpu.printf "%d\n" %c0 : i32

      %w1 = vector.extract %base[1] : vector<3xi32> from vector<3x3xi32>
      %w1_0 = vector.extract %w1[0] : i32 from vector<3xi32>
      %w1_1 = vector.extract %w1[1] : i32 from vector<3xi32>
      %w1_2 = vector.extract %w1[2] : i32 from vector<3xi32>
      gpu.printf "( %d, %d, %d )\n" %w1_0, %w1_1, %w1_2 : i32, i32, i32
      gpu.return
    }
  }

  func.func @main() {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@vector_extract blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args()
    func.return
  }
}
