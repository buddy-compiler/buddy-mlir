func.func @main() {
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  %cf9 = arith.constant 9.0 : f32
  %size = arith.constant 4 : index 
  %t0 = gpu.wait async
  // CHECK llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
  // CHECK llvm.mlir.constant(1 : index) : i64
  // CHECK llvm.mlir.zero : !llvm.ptr
  // CHECK llvm.getelementptr %7[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  // CEHCK llvm.ptrtoint %8 : !llvm.ptr to i64
  // CEHCK llvm.mlir.zero : !llvm.ptr
  // CHECK llvm.mlir.constant(0 : i8) : i8
  // CHECK llvm.call @mgpuMemAlloc(%9, %5, %11) : (i64, !llvm.ptr, i8) -> !llvm.ptr
  %m, %t1 = gpu.alloc async [%t0] (%size): memref<?xf32>
  gpu.wait [%t1]
  affine.for %i0 = 0 to 4 {
    affine.store %cf7, %m[%i0] : memref<?xf32>
    %v0 = affine.load %m[%i0] : memref<?xf32>
    %v1 = arith.addf %v0, %v0 : f32
    affine.store %cf8, %m[%i0] : memref<?xf32>
    affine.store %cf9, %m[%i0] : memref<?xf32>
    %v2 = affine.load %m[%i0] : memref<?xf32>
    %v3 = affine.load %m[%i0] : memref<?xf32>
    %v4 = arith.mulf %v2, %v3 : f32
  }
  %t2 = gpu.dealloc async [%t1] %m : memref<?xf32>
  call @printMemrefF32(%m) : (memref<?xf32>) -> ()
  gpu.wait [%t2]
  return
}
func.func private @printMemrefF32(memref<?xf32>)
