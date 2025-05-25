// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -O0 -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

// Create a 1x40x1536 input tensor
memref.global "private" @A : memref<1x40x1536xf32> = dense<3.0>

func.func @kernel(%a : memref<1x40x1536xf32>) {
  %t_start = call @rtclock() : () -> f64
  
  %b = memref.alloc() : memref<1x40xf32>  // Output tensor

  // Initialize constants
  %c0 = arith.constant 0.0 : f32
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c40 = arith.constant 40 : index
  %c1536 = arith.constant 1536 : index
  %c0_idx = arith.constant 0 : index
  %c8 = arith.constant 8 : index

  // Use blocking and vectorization
  affine.for %j0 = 0 to 40 step 8 {
    // Process 8 elements at a time
    affine.for %j1 = 0 to 8 {
      %j = affine.apply affine_map<(d0, d1) -> (d0 + d1)> (%j0, %j1)
      
      // Check if within valid range
      %j_in_range = arith.cmpi slt, %j, %c40 : index
      
      // Only compute within valid range
      scf.if %j_in_range {
        // Initialize accumulator
        %init_acc = arith.constant 0.0 : f32
        
        // Vectorize along k dimension with 32 elements
        %result_acc = affine.for %k = 0 to 1536 step 32 iter_args(%acc = %init_acc) -> f32 {
          // Prefetch next data block
          %next_k = arith.addi %k, %c32 : index
          %next_valid = arith.cmpi slt, %next_k, %c1536 : index
          scf.if %next_valid {
            memref.prefetch %a[%c0_idx, %j, %next_k], read, locality<3>, data : memref<1x40x1536xf32>
          }
          
          // Compute current block size and mask
          %remaining = arith.subi %c1536, %k : index
          %vl = arith.minsi %remaining, %c32 : index
          %mask = vector.create_mask %vl : vector<32xi1>
          
          // Vectorized data read
          %vec = vector.transfer_read %a[%c0_idx, %j, %k], %c0, %mask : memref<1x40x1536xf32>, vector<32xf32>
          
          // Vector reduction sum
          %block_sum = vector.reduction <add>, %vec : vector<32xf32> into f32
          %next_acc = arith.addf %acc, %block_sum : f32
          affine.yield %next_acc : f32
        }

        // Write result
        memref.store %result_acc, %b[%c0_idx, %j] : memref<1x40xf32>
      }
    }
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // Print result
  %printed_b = memref.cast %b : memref<1x40xf32> to memref<*xf32>
  
  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [1, 40] strides = [40, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [4608{{(, 4608)*}}]
  
  call @printMemrefF32(%printed_b) : (memref<*xf32>) -> ()
  
  // Print time
  vector.print %time : f64
  
  memref.dealloc %b : memref<1x40xf32>
  return
}

func.func @main() {
  %a = memref.get_global @A : memref<1x40x1536xf32>
  call @kernel(%a) : (memref<1x40x1536xf32>) -> ()
  return
}

