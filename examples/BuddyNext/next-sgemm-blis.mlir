// RUN: buddy-opt  %s \
// RUN:   -convert-scf-to-openmp \
// RUN:   -convert-vector-to-scf \
// RUN:   -expand-strided-metadata \
// RUN:   -lower-affine \
// RUN:   -cse \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-vector-to-llvm \
// RUN:   -memref-expand \
// RUN:   -arith-expand \
// RUN:   -convert-arith-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:	  -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-openmp-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-math-to-llvm \
// RUN:   -convert-math-to-libm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libomp%shlibext \
// RUN: | FileCheck %s



func.func private @printMemrefF32(memref<*xf32>)
func.func private @rtclock() -> f64

func.func @alloc_memref_2d_f32(%arg0: index, %arg1: index, %arg3: f32) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.for %idx0 = %c0 to %arg0 step %c1 {
        scf.for %idx1 = %c0 to %arg1 step %c1 {
            memref.store %arg3, %0[%idx0, %idx1] : memref<?x?xf32>
        }
    }
    return %0 : memref<?x?xf32>
}

// Scalar kernel for tail processing
// A: [m, k], B: [k, n], C: [m, n]
func.func @micro_kernel_scalar(%a: memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                               %b: memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                               %c: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    %m = memref.dim %a, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
    %k = memref.dim %a, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
    %n = memref.dim %b, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
    
    // Standard scalar processing: C[i,j] += A[i,k] * B[k,j]
    scf.for %i = %c0 to %m step %c1 {
        scf.for %j = %c0 to %n step %c1 {
            %c_val = memref.load %c[%i, %j] : memref<?x?xf32, strided<[?, 1], offset: ?>>
            %sum = scf.for %kk = %c0 to %k step %c1 iter_args(%acc = %c_val) -> (f32) {
                %a_val = memref.load %a[%i, %kk] : memref<?x?xf32, strided<[?, 1], offset: ?>>
                %b_val = memref.load %b[%kk, %j] : memref<?x?xf32, strided<[?, 1], offset: ?>>
                %prod = arith.mulf %a_val, %b_val : f32
                %new_acc = arith.addf %acc, %prod : f32
                scf.yield %new_acc : f32
            }
            memref.store %sum, %c[%i, %j] : memref<?x?xf32, strided<[?, 1], offset: ?>>
        }
    }
    return
}

// Vectorized micro-kernel with tail processing
// A: [m, k], B: [k, n], C: [m, n]
func.func @micro_kernel_vectorized_with_tail(%a: memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                             %b: memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                             %c: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index  // Vector size
    %f0 = arith.constant 0.0 : f32  // Padding value for vector reads
    
    %m = memref.dim %a, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
    %k = memref.dim %a, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
    %n = memref.dim %b, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
    
    // Process main 8x8 blocks
    %m_main = arith.floordivsi %m, %c8 : index
    %n_main = arith.floordivsi %n, %c8 : index
    %m_main_end = arith.muli %m_main, %c8 : index
    %n_main_end = arith.muli %n_main, %c8 : index
    
    // Main 8x8 vectorized processing
    scf.for %i = %c0 to %m_main_end step %c8 {
        scf.for %j = %c0 to %n_main_end step %c8 {
            %c_vec_0 = vector.transfer_read %c[%i, %j], %f0 {in_bounds = [true, true]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8x8xf32>
            
            %result = scf.for %kk = %c0 to %k step %c1 iter_args(%c_acc = %c_vec_0) -> (vector<8x8xf32>) {
                %a_vec = vector.transfer_read %a[%i, %kk], %f0 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8xf32>
                %b_vec = vector.transfer_read %b[%kk, %j], %f0 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8xf32>
                %prod = vector.outerproduct %a_vec, %b_vec, %c_acc : vector<8xf32>, vector<8xf32>
                scf.yield %prod : vector<8x8xf32>
            }
            
            vector.transfer_write %result, %c[%i, %j] {in_bounds = [true, true]} : vector<8x8xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        }
        
        // Process horizontal tail (remaining columns) with 8x4 blocks
        %n_tail = arith.subi %n, %n_main_end : index
        %has_tail = arith.cmpi sgt, %n_tail, %c0 : index
        scf.if %has_tail {
            %n_tail_4 = arith.floordivsi %n_tail, %c4 : index
            %n_tail_4_end = arith.muli %n_tail_4, %c4 : index
            %n_tail_4_end_abs = arith.addi %n_main_end, %n_tail_4_end : index
            
            // Process 8x4 blocks
            scf.for %j = %n_main_end to %n_tail_4_end_abs step %c4 {
                %c_vec_4 = vector.transfer_read %c[%i, %j], %f0 {in_bounds = [true, false]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8x4xf32>
                
                %result_4 = scf.for %kk = %c0 to %k step %c1 iter_args(%c_acc = %c_vec_4) -> (vector<8x4xf32>) {
                    %a_vec = vector.transfer_read %a[%i, %kk], %f0 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8xf32>
                    %b_vec = vector.transfer_read %b[%kk, %j], %f0 {in_bounds = [false]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<4xf32>
                    %prod = vector.outerproduct %a_vec, %b_vec, %c_acc : vector<8xf32>, vector<4xf32>
                    scf.yield %prod : vector<8x4xf32>
                }
                
                vector.transfer_write %result_4, %c[%i, %j] {in_bounds = [true, false]} : vector<8x4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
            }
            
            // Process remaining scalar columns
            scf.for %j = %n_tail_4_end_abs to %n step %c1 {
                %i_end = arith.addi %i, %c8 : index
                scf.for %ii = %i to %i_end step %c1 {
                    %c_val = memref.load %c[%ii, %j] : memref<?x?xf32, strided<[?, 1], offset: ?>>
                    %sum = scf.for %kk = %c0 to %k step %c1 iter_args(%acc = %c_val) -> (f32) {
                        %a_val = memref.load %a[%ii, %kk] : memref<?x?xf32, strided<[?, 1], offset: ?>>
                        %b_val = memref.load %b[%kk, %j] : memref<?x?xf32, strided<[?, 1], offset: ?>>
                        %prod = arith.mulf %a_val, %b_val : f32
                        %new_acc = arith.addf %acc, %prod : f32
                        scf.yield %new_acc : f32
                    }
                    memref.store %sum, %c[%ii, %j] : memref<?x?xf32, strided<[?, 1], offset: ?>>
                }
            }
        }
    }
    
    // Process vertical tail (remaining rows) with 4x8 blocks
    %m_tail = arith.subi %m, %m_main_end : index
    %has_m_tail = arith.cmpi sgt, %m_tail, %c0 : index
    scf.if %has_m_tail {
        %m_tail_4 = arith.floordivsi %m_tail, %c4 : index
        %m_tail_4_end = arith.muli %m_tail_4, %c4 : index
        %m_tail_4_end_abs = arith.addi %m_main_end, %m_tail_4_end : index
        
        // Process 4x8 blocks
        scf.for %i = %m_main_end to %m_tail_4_end_abs step %c4 {
            scf.for %j = %c0 to %n_main_end step %c8 {
                %c_vec_4 = vector.transfer_read %c[%i, %j], %f0 {in_bounds = [false, true]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<4x8xf32>
                
                %result_4 = scf.for %kk = %c0 to %k step %c1 iter_args(%c_acc = %c_vec_4) -> (vector<4x8xf32>) {
                    %a_vec = vector.transfer_read %a[%i, %kk], %f0 {in_bounds = [false]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<4xf32>
                    %b_vec = vector.transfer_read %b[%kk, %j], %f0 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8xf32>
                    %prod = vector.outerproduct %a_vec, %b_vec, %c_acc : vector<4xf32>, vector<8xf32>
                    scf.yield %prod : vector<4x8xf32>
                }
                
                vector.transfer_write %result_4, %c[%i, %j] {in_bounds = [false, true]} : vector<4x8xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
            }
        }
        
        // Process remaining rows with scalar operations
        scf.for %i = %m_tail_4_end_abs to %m step %c1 {
            scf.for %j = %c0 to %n step %c1 {
                %c_val = memref.load %c[%i, %j] : memref<?x?xf32, strided<[?, 1], offset: ?>>
                %sum = scf.for %kk = %c0 to %k step %c1 iter_args(%acc = %c_val) -> (f32) {
                    %a_val = memref.load %a[%i, %kk] : memref<?x?xf32, strided<[?, 1], offset: ?>>
                    %b_val = memref.load %b[%kk, %j] : memref<?x?xf32, strided<[?, 1], offset: ?>>
                    %prod = arith.mulf %a_val, %b_val : f32
                    %new_acc = arith.addf %acc, %prod : f32
                    scf.yield %new_acc : f32
                }
                memref.store %sum, %c[%i, %j] : memref<?x?xf32, strided<[?, 1], offset: ?>>
            }
        }
    }
    
    return
}

// Unified multi-level tiled GEMM function with tail processing
// Standard GEMM: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
func.func @gemm(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) {
    %t_start = call @rtclock() : () -> f64

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    
    // Tile sizes for different cache levels
    %nc = arith.constant 128 : index  // L3 cache blocking for N
    %kc = arith.constant 64 : index   // L2 cache blocking for K
    %mc = arith.constant 64 : index   // L2 cache blocking for M
    %mr = arith.constant 16 : index   // Register blocking M
    %nr = arith.constant 16 : index   // Register blocking N
    
    %m = memref.dim %a, %c0 : memref<?x?xf32>
    %k = memref.dim %a, %c1 : memref<?x?xf32>
    %n = memref.dim %b, %c1 : memref<?x?xf32>
    
    // Level 0: N-dimension cache blocking (nc) - L3 cache
    scf.parallel (%jc) = (%c0) to (%n) step (%nc) {
        %n_size = affine.min affine_map<(d0, d1, d2) -> (d0, d2 - d1)>(%nc, %jc, %n)
        %jc_end = arith.addi %jc, %n_size : index
        
        // Level 1: K-dimension cache blocking (kc) - L2 cache
        scf.for %pc = %c0 to %k step %kc {
            %k_size = affine.min affine_map<(d0, d1, d2) -> (d0, d2 - d1)>(%kc, %pc, %k)
            %pc_end = arith.addi %pc, %k_size : index
            
            // Level 2: M-dimension cache blocking (mc) - L2 cache
            scf.parallel (%ic) = (%c0) to (%m) step (%mc) {
                %m_size = affine.min affine_map<(d0, d1, d2) -> (d0, d2 - d1)>(%mc, %ic, %m)
                %ic_end = arith.addi %ic, %m_size : index
                
                // Level 3: Register blocking with adaptive tile sizes
                // Process main 16x16 blocks
                %m_main_16 = arith.floordivsi %m_size, %c16 : index
                %n_main_16 = arith.floordivsi %n_size, %c16 : index
                %m_16_end = arith.muli %m_main_16, %c16 : index
                %n_16_end = arith.muli %n_main_16, %c16 : index
                %m_16_end_abs = arith.addi %ic, %m_16_end : index
                %n_16_end_abs = arith.addi %jc, %n_16_end : index
                
                // Process full 16x16 blocks
                scf.for %ir = %ic to %m_16_end_abs step %mr {
                    scf.for %jr = %jc to %n_16_end_abs step %nr {
                        %a_micro = memref.subview %a[%ir, %pc][%mr, %k_size][1, 1] : 
                            memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                        %b_micro = memref.subview %b[%pc, %jr][%k_size, %nr][1, 1] : 
                            memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                        %c_micro = memref.subview %c[%ir, %jr][%mr, %nr][1, 1] : 
                            memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                        
                        func.call @micro_kernel_vectorized_with_tail(%a_micro, %b_micro, %c_micro) : 
                            (memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                             memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                             memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
                    }
                    
                    // Handle horizontal tail (remaining N)
                    %n_tail = arith.subi %jc_end, %n_16_end_abs : index
                    %has_n_tail = arith.cmpi sgt, %n_tail, %c0 : index
                    scf.if %has_n_tail {
                        // Try 16x8 blocks first
                        %n_tail_8 = arith.cmpi sge, %n_tail, %c8 : index
                        scf.if %n_tail_8 {
                            %a_micro = memref.subview %a[%ir, %pc][%mr, %k_size][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            %b_micro = memref.subview %b[%pc, %n_16_end_abs][%k_size, %c8][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            %c_micro = memref.subview %c[%ir, %n_16_end_abs][%mr, %c8][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            
                            func.call @micro_kernel_vectorized_with_tail(%a_micro, %b_micro, %c_micro) : 
                                (memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                 memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                 memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
                            
                            // Handle remaining columns after 8
                            %n_16_8_end = arith.addi %n_16_end_abs, %c8 : index
                            %n_remaining = arith.subi %jc_end, %n_16_8_end : index
                            %has_remaining = arith.cmpi sgt, %n_remaining, %c0 : index
                            scf.if %has_remaining {
                                %a_micro2 = memref.subview %a[%ir, %pc][%mr, %k_size][1, 1] : 
                                    memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                                %b_micro2 = memref.subview %b[%pc, %n_16_8_end][%k_size, %n_remaining][1, 1] : 
                                    memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                                %c_micro2 = memref.subview %c[%ir, %n_16_8_end][%mr, %n_remaining][1, 1] : 
                                    memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                                
                                func.call @micro_kernel_scalar(%a_micro2, %b_micro2, %c_micro2) : 
                                    (memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                     memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                     memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
                            }
                        } else {
                            // Less than 8 columns, use scalar
                            %a_micro = memref.subview %a[%ir, %pc][%mr, %k_size][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            %b_micro = memref.subview %b[%pc, %n_16_end_abs][%k_size, %n_tail][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            %c_micro = memref.subview %c[%ir, %n_16_end_abs][%mr, %n_tail][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            
                            func.call @micro_kernel_scalar(%a_micro, %b_micro, %c_micro) : 
                                (memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                 memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                 memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
                        }
                    }
                }
                
                // Handle vertical tail (remaining M)
                %m_tail = arith.subi %ic_end, %m_16_end_abs : index
                %has_m_tail = arith.cmpi sgt, %m_tail, %c0 : index
                scf.if %has_m_tail {
                    // Try 8xN blocks for M tail
                    %m_tail_8 = arith.cmpi sge, %m_tail, %c8 : index
                    scf.if %m_tail_8 {
                        // Process 8xN blocks
                        scf.for %jr = %jc to %jc_end step %nr {
                            %n_micro = affine.min affine_map<(d0, d1, d2) -> (d0, d2 - d1)>(%nr, %jr, %jc_end)
                            
                            %a_micro = memref.subview %a[%m_16_end_abs, %pc][%c8, %k_size][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            %b_micro = memref.subview %b[%pc, %jr][%k_size, %n_micro][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            %c_micro = memref.subview %c[%m_16_end_abs, %jr][%c8, %n_micro][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            
                            %use_vector = arith.cmpi sge, %n_micro, %c8 : index
                            scf.if %use_vector {
                                func.call @micro_kernel_vectorized_with_tail(%a_micro, %b_micro, %c_micro) : 
                                    (memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                     memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                     memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
                            } else {
                                func.call @micro_kernel_scalar(%a_micro, %b_micro, %c_micro) : 
                                    (memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                     memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                     memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
                            }
                        }
                        
                        // Handle remaining rows after 8
                        %m_8_end = arith.addi %m_16_end_abs, %c8 : index
                        %m_remaining = arith.subi %ic_end, %m_8_end : index
                        %has_remaining = arith.cmpi sgt, %m_remaining, %c0 : index
                        scf.if %has_remaining {
                            scf.for %jr = %jc to %jc_end step %c1 {
                                %a_micro = memref.subview %a[%m_8_end, %pc][%m_remaining, %k_size][1, 1] : 
                                    memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                                %b_micro = memref.subview %b[%pc, %jr][%k_size, %c1][1, 1] : 
                                    memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                                %c_micro = memref.subview %c[%m_8_end, %jr][%m_remaining, %c1][1, 1] : 
                                    memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                                
                                func.call @micro_kernel_scalar(%a_micro, %b_micro, %c_micro) : 
                                    (memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                     memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                     memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
                            }
                        }
                    } else {
                        // Less than 8 rows, use scalar for all
                        scf.for %jr = %jc to %jc_end step %c1 {
                            %a_micro = memref.subview %a[%m_16_end_abs, %pc][%m_tail, %k_size][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            %b_micro = memref.subview %b[%pc, %jr][%k_size, %c1][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            %c_micro = memref.subview %c[%m_16_end_abs, %jr][%m_tail, %c1][1, 1] : 
                                memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                            
                            func.call @micro_kernel_scalar(%a_micro, %b_micro, %c_micro) : 
                                (memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                 memref<?x?xf32, strided<[?, 1], offset: ?>>, 
                                 memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
                        }
                    }
                }
            }
        }
    }

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
    vector.print %time : f64
    // CHECK: {{[0-9]+\.[0-9]+}}
    return
}

func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %M = arith.constant 1024 : index
    %N = arith.constant 1536 : index
    %K = arith.constant 8960 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f3 = arith.constant 3.0 : f32
    
    // A: [M, K], B: [K, N], C: [M, N]
    %a = call @alloc_memref_2d_f32(%M, %K, %f2) : (index, index, f32) -> memref<?x?xf32>
    %b = call @alloc_memref_2d_f32(%K, %N, %f3) : (index, index, f32) -> memref<?x?xf32>
    %c = call @alloc_memref_2d_f32(%M, %N, %f0) : (index, index, f32) -> memref<?x?xf32>
    
    call @gemm(%a, %b, %c) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    
    // %print_C = memref.cast %c : memref<?x?xf32> to memref<*xf32>
    // call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()
    
    return
}