// RUN: buddy-opt %s --vectorize-dequantize --canonicalize --cse --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 | FileCheck %s
// CHECK: 0

// Execute the transformed vector code against a scalar oracle, bit for bit.
// Lengths 0..35 cover empty tensors, every short tail, a full vector, and two
// vectors plus a tail. Both source memrefs have nonzero descriptor offsets.
// The signed accumulators straddle f32's exact-integer boundary, and scales
// are deliberately inexact in binary, exercising both cast and multiply
// rounding. Only the linalg function is eligible for the optimization; the
// oracle remains scalar SCF arithmetic and has separate ordered multiplies.

#id = affine_map<(d0) -> (d0)>
func.func @compute(%acc: memref<?xi32, strided<[1], offset: ?>>,
                   %ws: memref<?xf32, strided<[1], offset: ?>>,
                   %scale: f32) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %n = memref.dim %acc, %c0 : memref<?xi32, strided<[1], offset: ?>>
  %out = memref.alloc(%n) : memref<?xf32>
  linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
      ins(%acc, %ws : memref<?xi32, strided<[1], offset: ?>>,
                      memref<?xf32, strided<[1], offset: ?>>)
      outs(%out : memref<?xf32>) {
    ^bb0(%a: i32, %w: f32, %unused: f32):
      %f = arith.sitofp %a : i32 to f32
      %fa = arith.mulf %f, %scale : f32
      %faw = arith.mulf %fa, %w : f32
      linalg.yield %faw : f32
  }
  return %out : memref<?xf32>
}

func.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c36 = arith.constant 36 : index
  %c40 = arith.constant 40 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %base = arith.constant 16777217 : i32
  %scale = arith.constant 0.1 : f32
  %weightScale = arith.constant 0.3 : f32
  %acc = memref.alloc() : memref<40xi32>
  %ws = memref.alloc() : memref<40xf32>
  scf.for %i = %c0 to %c40 step %c1 {
    %ii = arith.index_cast %i : index to i32
    %value = arith.addi %ii, %base : i32
    %negative = arith.subi %zero, %value : i32
    %odd = arith.andi %ii, %one : i32
    %isOdd = arith.cmpi ne, %odd, %zero : i32
    %signedValue = arith.select %isOdd, %negative, %value : i32
    memref.store %signedValue, %acc[%i] : memref<40xi32>
    memref.store %weightScale, %ws[%i] : memref<40xf32>
  }
  %errors = scf.for %n = %c0 to %c36 step %c1 iter_args(%total = %zero) -> i32 {
    %a = memref.subview %acc[3] [%n] [1] : memref<40xi32> to memref<?xi32, strided<[1], offset: 3>>
    %w = memref.subview %ws[3] [%n] [1] : memref<40xf32> to memref<?xf32, strided<[1], offset: 3>>
    %ac = memref.cast %a : memref<?xi32, strided<[1], offset: 3>> to memref<?xi32, strided<[1], offset: ?>>
    %wc = memref.cast %w : memref<?xf32, strided<[1], offset: 3>> to memref<?xf32, strided<[1], offset: ?>>
    %out = func.call @compute(%ac, %wc, %scale) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, f32) -> memref<?xf32>
    %e = scf.for %j = %c0 to %n step %c1 iter_args(%count = %total) -> i32 {
      %av = memref.load %a[%j] : memref<?xi32, strided<[1], offset: 3>>
      %wv = memref.load %w[%j] : memref<?xf32, strided<[1], offset: 3>>
      %f = arith.sitofp %av : i32 to f32
      %fa = arith.mulf %f, %scale : f32
      %expected = arith.mulf %fa, %wv : f32
      %actual = memref.load %out[%j] : memref<?xf32>
      %actualBits = arith.bitcast %actual : f32 to i32
      %expectedBits = arith.bitcast %expected : f32 to i32
      %different = arith.cmpi ne, %actualBits, %expectedBits : i32
      %delta = arith.extui %different : i1 to i32
      %next = arith.addi %count, %delta : i32
      scf.yield %next : i32
    }
    memref.dealloc %out : memref<?xf32>
    scf.yield %e : i32
  }
  memref.dealloc %acc : memref<40xi32>
  memref.dealloc %ws : memref<40xf32>
  return %errors : i32
}
