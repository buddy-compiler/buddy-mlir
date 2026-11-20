// RUN: buddy-opt %s --lower-linalg-to-ame='target=gem5' --lower-ame --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --expand-strided-metadata --lower-affine --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir -o %t.ll
// RUN: buddy-llc %t.ll -mtriple=riscv64 -mattr=+m,+xame,+xame-gem5 -target-abi=lp64 -verify-machineinstrs -filetype=obj -o %t.o
// RUN: buddy-llc %t.ll -mtriple=riscv64 -mattr=+m,+xame,+xame-gem5 -target-abi=lp64 -verify-machineinstrs -o - | FileCheck %s
// RUN: buddy-opt %s --lower-linalg-to-ame --lower-ame --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --expand-strided-metadata --lower-affine --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir -o %t.generic.ll
// RUN: buddy-llc %t.generic.ll -mtriple=riscv64 -mattr=+m,+xame -target-abi=lp64 -verify-machineinstrs -filetype=obj -o %t.generic.o

// This standalone main also runs with the AME gem5 Linux-SE startup. It checks
// tails in all three dimensions, dynamic shapes, nonzero offsets, non-unit
// strides, integer overflow, zero K and preservation of the initial C. The
// expected result uses ordinary scalar arithmetic; AME does not compute it.
func.func @matmul_strided(
    %a: memref<?x?xi32, strided<[?, ?], offset: ?>>,
    %b: memref<?x?xi32, strided<[?, ?], offset: ?>>,
    %c: memref<?x?xi32, strided<[?, ?], offset: ?>>) {
  linalg.matmul ins(%a, %b : memref<?x?xi32, strided<[?, ?], offset: ?>>,
                             memref<?x?xi32, strided<[?, ?], offset: ?>>)
                outs(%c : memref<?x?xi32, strided<[?, ?], offset: ?>>)
  return
}

// CHECK-LABEL: matmul_strided:
// CHECK: ame.acquire
// CHECK: msettyp
// CHECK: asettyp
// CHECK: mmov.a.m
// CHECK: mmulacc.2d
// CHECK: mss.rm
// CHECK: ame.release

func.func @main() -> i32 {
  %a_storage = memref.alloca() : memref<16x16xi32>
  %b_storage = memref.alloca() : memref<16x16xi32>
  %c_storage = memref.alloca() : memref<16x16xi32>
  %a = memref.subview %a_storage[2, 3] [5, 7] [2, 2] :
      memref<16x16xi32> to memref<5x7xi32, strided<[32, 2], offset: 35>>
  %b = memref.subview %b_storage[1, 2] [7, 6] [1, 2] :
      memref<16x16xi32> to memref<7x6xi32, strided<[16, 2], offset: 18>>
  %c = memref.subview %c_storage[3, 1] [5, 6] [1, 2] :
      memref<16x16xi32> to memref<5x6xi32, strided<[16, 2], offset: 49>>
  %ad = memref.cast %a : memref<5x7xi32, strided<[32, 2], offset: 35>>
      to memref<?x?xi32, strided<[?, ?], offset: ?>>
  %bd = memref.cast %b : memref<7x6xi32, strided<[16, 2], offset: 18>>
      to memref<?x?xi32, strided<[?, ?], offset: ?>>
  %cd = memref.cast %c : memref<5x6xi32, strided<[16, 2], offset: 49>>
      to memref<?x?xi32, strided<[?, ?], offset: ?>>
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i5 = arith.constant 5 : index
  %i6 = arith.constant 6 : index
  %i7 = arith.constant 7 : index
  %i16 = arith.constant 16 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %three = arith.constant 3 : i32
  %seven = arith.constant 7 : i32
  %scale = arith.constant 1073741825 : i32
  %sentinel = arith.constant 99 : i32
  %false = arith.constant false

  scf.for %i = %i0 to %i16 step %i1 {
    scf.for %j = %i0 to %i16 step %i1 {
      memref.store %sentinel, %a_storage[%i, %j] : memref<16x16xi32>
      memref.store %sentinel, %b_storage[%i, %j] : memref<16x16xi32>
      memref.store %sentinel, %c_storage[%i, %j] : memref<16x16xi32>
    }
  }
  scf.for %i = %i0 to %i5 step %i1 {
    %iv = arith.index_cast %i : index to i32
    scf.for %k = %i0 to %i7 step %i1 {
      %kv = arith.index_cast %k : index to i32
      %difference = arith.subi %iv, %kv : i32
      %unscaled = arith.subi %difference, %two : i32
      %value = arith.muli %unscaled, %scale : i32
      memref.store %value, %a[%i, %k] :
          memref<5x7xi32, strided<[32, 2], offset: 35>>
    }
  }
  scf.for %k = %i0 to %i7 step %i1 {
    %kv = arith.index_cast %k : index to i32
    scf.for %j = %i0 to %i6 step %i1 {
      %jv = arith.index_cast %j : index to i32
      %difference = arith.subi %kv, %jv : i32
      %value = arith.addi %difference, %three : i32
      memref.store %value, %b[%k, %j] :
          memref<7x6xi32, strided<[16, 2], offset: 18>>
    }
  }
  scf.for %i = %i0 to %i5 step %i1 {
    scf.for %j = %i0 to %i6 step %i1 {
      memref.store %seven, %c[%i, %j] :
          memref<5x6xi32, strided<[16, 2], offset: 49>>
    }
  }
  func.call @matmul_strided(%ad, %bd, %cd) :
      (memref<?x?xi32, strided<[?, ?], offset: ?>>,
       memref<?x?xi32, strided<[?, ?], offset: ?>>,
       memref<?x?xi32, strided<[?, ?], offset: ?>>) -> ()

  // An empty reduction must leave the existing C unchanged.
  %a0 = memref.subview %a[0, 0] [5, 0] [1, 1] :
      memref<5x7xi32, strided<[32, 2], offset: 35>> to
      memref<5x0xi32, strided<[32, 2], offset: 35>>
  %b0 = memref.subview %b[0, 0] [0, 6] [1, 1] :
      memref<7x6xi32, strided<[16, 2], offset: 18>> to
      memref<0x6xi32, strided<[16, 2], offset: 18>>
  %ad0 = memref.cast %a0 : memref<5x0xi32, strided<[32, 2], offset: 35>>
      to memref<?x?xi32, strided<[?, ?], offset: ?>>
  %bd0 = memref.cast %b0 : memref<0x6xi32, strided<[16, 2], offset: 18>>
      to memref<?x?xi32, strided<[?, ?], offset: ?>>
  func.call @matmul_strided(%ad0, %bd0, %cd) :
      (memref<?x?xi32, strided<[?, ?], offset: ?>>,
       memref<?x?xi32, strided<[?, ?], offset: ?>>,
       memref<?x?xi32, strided<[?, ?], offset: ?>>) -> ()

  %bad = scf.for %i = %i0 to %i5 step %i1
      iter_args(%bad_rows = %false) -> (i1) {
    %iv = arith.index_cast %i : index to i32
    %bad_row = scf.for %j = %i0 to %i6 step %i1
        iter_args(%bad_columns = %bad_rows) -> (i1) {
      %jv = arith.index_cast %j : index to i32
      %expected = scf.for %k = %i0 to %i7 step %i1
          iter_args(%sum = %seven) -> (i32) {
        %kv = arith.index_cast %k : index to i32
        %adifference = arith.subi %iv, %kv : i32
        %unscaled = arith.subi %adifference, %two : i32
        %av = arith.muli %unscaled, %scale : i32
        %bdifference = arith.subi %kv, %jv : i32
        %bv = arith.addi %bdifference, %three : i32
        %product = arith.muli %av, %bv : i32
        %next_sum = arith.addi %sum, %product : i32
        scf.yield %next_sum : i32
      }
      %actual = memref.load %c[%i, %j] :
          memref<5x6xi32, strided<[16, 2], offset: 49>>
      %mismatch = arith.cmpi ne, %actual, %expected : i32
      %bad_element = arith.ori %bad_columns, %mismatch : i1
      scf.yield %bad_element : i1
    }
    scf.yield %bad_row : i1
  }
  // A location outside the strided C view must remain unchanged.
  %outside = memref.load %c_storage[%i0, %i0] : memref<16x16xi32>
  %outside_changed = arith.cmpi ne, %outside, %sentinel : i32
  %failed = arith.ori %bad, %outside_changed : i1
  %status = arith.select %failed, %one, %zero : i32
  return %status : i32
}
