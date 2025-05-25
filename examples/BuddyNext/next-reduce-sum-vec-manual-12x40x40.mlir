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

// Create a 12x40x40 input tensor
memref.global "private" @A : memref<12x40x40xf32> = dense<3.0>

func.func @kernel(%a : memref<12x40x40xf32>) {
  %t_start = call @rtclock() : () -> f64
  
  %b = memref.alloc() : memref<12x40xf32>  // Output tensor

  // Initialize constants
  %c0 = arith.constant 0.0 : f32
  %c16 = arith.constant 16 : index
  %c12 = arith.constant 12 : index
  %c40 = arith.constant 40 : index
  %c0_idx = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  // Use outer loop with step 1 and 8x8 blocking
  affine.for %i0 = 0 to 12 step 1 {
    affine.for %j0 = 0 to 40 step 8 {
      // Use 1D parallel processing
      affine.parallel (%idx) = (0) to (8) {
        // Compute j1
        %j1 = arith.remui %idx, %c8 : index
        
        %j = affine.apply affine_map<(d0, d1) -> (d0 + d1)> (%j0, %j1)
        
        // Check if within valid range
        %j_in_range = arith.cmpi slt, %j, %c40 : index
        
        // Only compute within valid range
        scf.if %j_in_range {
          // Initialize accumulator
          %init_acc = arith.constant 0.0 : f32
          
          // Vectorize along k dimension with 16 elements
          %result_acc = affine.for %k = 0 to 40 step 16 iter_args(%acc = %init_acc) -> f32 {
            // Prefetch next data block
            %next_k = arith.addi %k, %c16 : index
            %next_valid = arith.cmpi slt, %next_k, %c40 : index
            scf.if %next_valid {
              memref.prefetch %a[%i0, %j, %next_k], read, locality<3>, data : memref<12x40x40xf32>
            }
            
            // Compute current block size and mask
            %remaining = arith.subi %c40, %k : index
            %vl = arith.minsi %remaining, %c16 : index
            %mask = vector.create_mask %vl : vector<16xi1>
            
            // Vectorized data read
            %vec = vector.transfer_read %a[%i0, %j, %k], %c0, %mask : memref<12x40x40xf32>, vector<16xf32>
            
            // Vector reduction sum
            %block_sum = vector.reduction <add>, %vec : vector<16xf32> into f32
            %next_acc = arith.addf %acc, %block_sum : f32
            affine.yield %next_acc : f32
          }

          // Write result
          memref.store %result_acc, %b[%i0, %j] : memref<12x40xf32>
        }
      }
    }
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // Print result
  %printed_b = memref.cast %b : memref<12x40xf32> to memref<*xf32>
  
  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [12, 40] strides = [40, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [120{{(, 120)*}}]
  
  call @printMemrefF32(%printed_b) : (memref<*xf32>) -> ()
  
  // Print timings
  vector.print %time : f64
  
  memref.dealloc %b : memref<12x40xf32>
  return
}

func.func @main() {
  %a = memref.get_global @A : memref<12x40x40xf32>
  call @kernel(%a) : (memref<12x40x40xf32>) -> ()
  return
}