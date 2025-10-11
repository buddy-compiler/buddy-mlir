// RUN: buddy-opt  %s \
// RUN:   -convert-linalg-to-loops \
// RUN:   -cse \
// RUN:   -lower-affine \
// RUN:   -convert-vector-to-scf \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-vector-to-llvm \
// RUN:   -expand-strided-metadata \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#cond_set_a = affine_set<(i, j)[M, K] : (M - 1 - i >= 0, K - 1 - j >= 0)>
#cond_set_b = affine_set<(i, j)[N, K] : (K - 1 - i >= 0, N - 1 - j >= 0)>
#micro_kernel_offset = affine_map<(off, idx, p)[KC]->(off + idx*KC + p)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64
  
  func.func @pack_a(%a : memref<?x?xf32>, %a_packed : memref<?xf32>, %i_c: index, %p_c: index) {
    %cf0 = arith.constant 0. : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %KC = arith.constant 128 : index
    %MC = arith.constant 256 : index

    %M = memref.dim %a, %c0 : memref<?x?xf32>
    %K = memref.dim %a, %c1 : memref<?x?xf32>

    affine.for %j = 0 to %KC { // KC
      affine.for %i = 0 to %MC { // MC
        %i_global = affine.apply affine_map<(i_c,i)->(i_c+i)>(%i_c, %i)
        %j_global = affine.apply affine_map<(p_c,j)->(p_c+j)>(%p_c, %j)

        %idx = affine.apply affine_map<(i,j)[KC]->(i*KC + j)>(%i, %j)[%KC]

        affine.if #cond_set_a(%i_global, %j_global)[%M, %K] {
          %val = affine.load %a[%i_global, %j_global] : memref<?x?xf32>
          affine.store %val, %a_packed[%idx] : memref<?xf32>
        } else {
          affine.store %cf0, %a_packed[%idx] : memref<?xf32>
        }
      }
    }
    return
  }

  func.func @pack_b(%b : memref<?x?xf32>, %b_packed : memref<?xf32>, %j_c: index, %p_c: index) {
    %cf0 = arith.constant 0. : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %KC = arith.constant 128 : index
    %NC = arith.constant 1024 : index

    %K = memref.dim %b, %c0 : memref<?x?xf32>
    %N = memref.dim %b, %c1 : memref<?x?xf32>

    affine.for %i = 0 to %KC { // KC
      affine.for %j = 0 to %NC { // MC
        %i_global = affine.apply affine_map<(p_c,i)->(p_c+i)>(%p_c, %i)
        %j_global = affine.apply affine_map<(j_c,j)->(j_c+j)>(%j_c, %j)

        %idx = affine.apply affine_map<(i,j)[KC]->(i + j*KC)>(%i, %j)[%KC]

        affine.if #cond_set_b(%i_global, %j_global)[%N, %K] {
          %val = affine.load %b[%i_global, %j_global] : memref<?x?xf32>
          affine.store %val, %b_packed[%idx] : memref<?xf32>
        } else {
          affine.store %cf0, %b_packed[%idx] : memref<?xf32>
        }
      }
    }
    return
  }

  func.func @micro_kernel(%a_packed : memref<?xf32>, %b_packed : memref<?xf32>, %c_sub : memref<?xf32>, %a_offset: index, %b_offset: index) {
    %cf0 = arith.constant 0. : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %NC = arith.constant 1024 : index
    %MC = arith.constant 256 : index
    %KC = arith.constant 128 : index

    %MR = arith.constant 4 : index
    %NR = arith.constant 8 : index

    affine.for %i = 0 to %MR {
      affine.for %j = 0 to %NR {
        %idx = affine.apply affine_map<(i,j)[NR]->(i*NR + j)>(%i, %j)[%NR]

        %acc_0 = arith.constant 0.0 : f32
        %c_val_final = scf.for %p = %c0 to %KC step %c1 iter_args(%acc = %acc_0) -> f32 {
          
          %a_idx = affine.apply #micro_kernel_offset(%a_offset, %i, %p)[%KC]
          %b_idx = affine.apply #micro_kernel_offset(%b_offset, %j, %p)[%KC]

          %a_val = memref.load %a_packed[%a_idx] : memref<?xf32>
          %b_val = memref.load %b_packed[%b_idx] : memref<?xf32>

          %prod = arith.mulf %a_val, %b_val : f32
          %new_acc = arith.addf %acc, %prod : f32

          scf.yield %new_acc : f32
        }

        affine.store %c_val_final, %c_sub[%idx] : memref<?xf32>
      }
    }
    return
  }

  func.func @sgemm_blis_32(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    %t_start = func.call @rtclock() : () -> f64

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %M = memref.dim %a, %c0 : memref<?x?xf32>
    %N = memref.dim %c, %c1 : memref<?x?xf32>
    %K = memref.dim %a, %c1 : memref<?x?xf32>

    %NC = arith.constant 1024 : index
    %MC = arith.constant 256 : index
    %KC = arith.constant 128 : index

    %MR = arith.constant 4 : index
    %NR = arith.constant 8 : index

    %a_packed_size = arith.muli %MC, %KC : index
    %b_packed_size = arith.muli %KC, %NC : index

    %a_packed = memref.alloc(%a_packed_size) : memref<?xf32>
    %b_packed = memref.alloc(%b_packed_size) : memref<?xf32>

    %c_sub_size = arith.muli %MR, %NR: index
    %c_sub = memref.alloc(%c_sub_size) : memref<?xf32>

    affine.for %j_c = 0 to %N step 1024 { // NC
      affine.for %p_c = 0 to %K step 128 { // KC
        affine.for %i_c = 0 to %M step 256 { // MC
          func.call @pack_a(%a, %a_packed, %i_c, %p_c) : (memref<?x?xf32>, memref<?xf32>, index, index) -> ()
          func.call @pack_b(%b, %b_packed, %j_c, %p_c) : (memref<?x?xf32>, memref<?xf32>, index, index) -> ()

          affine.for %j_r = 0 to %NC step 8 { // NR
            affine.for %i_r = 0 to %MC step 4 { // MR
              %a_offset = affine.apply affine_map<(i_r)[KC]->(i_r*KC)>(%i_r)[%KC]
              %b_offset = affine.apply affine_map<(j_r)[KC]->(j_r*KC)>(%j_r)[%KC]

              func.call @micro_kernel(%a_packed, %b_packed, %c_sub, %a_offset, %b_offset) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index, index) -> ()

              affine.for %i = 0 to %MR {
                affine.for %j = 0 to %NR {
                  %c_sub_idx = affine.apply affine_map<(i,j)[NR]->(i*NR + j)>(%i, %j)[%NR]
                  %c_sub_val = memref.load %c_sub[%c_sub_idx] : memref<?xf32>

                  %i_global = affine.apply affine_map<(i_c,i_r,i)->(i_c+i_r+i)>(%i_c, %i_r, %i)
                  %j_global = affine.apply affine_map<(j_c,j_r,j)->(j_c+j_r+j)>(%j_c, %j_r, %j)

                  affine.if #cond_set_a(%i_global, %j_global)[%M, %N] {
                    %c_val = memref.load %c[%i_global, %j_global] : memref<?x?xf32>
                    %c_new = arith.addf %c_val, %c_sub_val : f32
                    memref.store %c_new, %c[%i_global, %j_global] : memref<?x?xf32>
                  }
                }
              }
            }
          }
        }
      }
    }

    memref.dealloc %c_sub : memref<?xf32>
    memref.dealloc %a_packed : memref<?xf32>
    memref.dealloc %b_packed : memref<?xf32>

    %t_end = func.call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
    vector.print %time : f64
    // CHECK: {{[0-9]+\.[0-9]+}}
    func.return
  }

  func.func @main(){
    // Set up dims.
    %cM = arith.constant 1024 : index
    %cN = arith.constant 1536 : index
    %cK = arith.constant 8960 : index

    // Set Init Value.
    %cf1 = arith.constant 1.0 : f32
    %cf2 = arith.constant 2.0 : f32
    %c0 = arith.constant 0.0 : f32

    %A = memref.alloc(%cM, %cK) : memref<?x?xf32>
    %B = memref.alloc(%cK, %cN) : memref<?x?xf32>
    %C = memref.alloc(%cM, %cN) : memref<?x?xf32>

    linalg.fill
    ins(%cf1 : f32)
    outs(%A:memref<?x?xf32>)

    linalg.fill
    ins(%cf2 : f32)
    outs(%B:memref<?x?xf32>)

    linalg.fill
    ins(%c0 : f32)
    outs(%C:memref<?x?xf32>)

    call @sgemm_blis_32(%A, %B, %C) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // %print_C = memref.cast %C : memref<?x?xf32> to memref<*xf32>
    // call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

    %i = arith.constant 0 : index
    %j = arith.constant 0 : index
    %val = memref.load %C[%i, %j] : memref<?x?xf32>
    vector.print %val : f32

    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<?x?xf32>
    return
  }
}
