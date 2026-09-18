// RUN: buddy-opt %s --vectorize-quantize --canonicalize --cse | FileCheck %s --check-prefix=IR
// RUN: buddy-opt %s --vectorize-quantize --convert-linalg-to-loops --canonicalize --cse --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --convert-math-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 | FileCheck %s
// CHECK: 0
// Exercise every tail (lengths 1..35), a nonzero input offset, all-zero rows,
// non-power-of-two scales, and exact agreement with independent scalar SCF.
#id = affine_map<(d0) -> (d0)>
// IR-LABEL: func.func @compute
// IR: math.absf {{.*}} : vector<16xf32>
// IR: vector.reduction <maxnumf>
// IR: arith.divf {{.*}} : vector<16xf32>
// IR: arith.fptosi {{.*}} : vector<16xf32> to vector<16xi32>
// IR: arith.trunci {{.*}} : vector<16xi32> to vector<16xi8>
// IR: arith.fptosi {{.*}} : f32 to i32
func.func @compute(%x: memref<?xf32, strided<[1], offset: ?>>) -> (memref<?xi8>, f32) {
  %c0 = arith.constant 0 : index
  %n = memref.dim %x, %c0 : memref<?xf32, strided<[1], offset: ?>>
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %half = arith.constant 0.5 : f32
  %nhalf = arith.constant -0.5 : f32
  %high = arith.constant 127.0 : f32
  %low = arith.constant -127.0 : f32
  %inf = arith.constant 0xFF800000 : f32
  %abs = memref.alloc(%n) : memref<?xf32>
  %max = memref.alloc() : memref<f32>
  %out = memref.alloc(%n) : memref<?xi8>
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<?xf32, strided<[1], offset: ?>>) outs(%abs : memref<?xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %a = math.absf %v : f32
      linalg.yield %a : f32
  }
  linalg.fill ins(%inf : f32) outs(%max : memref<f32>)
  linalg.reduce ins(%abs : memref<?xf32>) outs(%max : memref<f32>) dimensions = [0]
      (%a: f32, %b: f32) {
    %m = arith.maxnumf %a, %b : f32
    linalg.yield %m : f32
  }
  %maximum = memref.load %max[] : memref<f32>
  %s = arith.divf %maximum, %high : f32
  %iszero = arith.cmpf oeq, %maximum, %zero : f32
  %scale = arith.select %iszero, %one, %s : f32
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<?xf32, strided<[1], offset: ?>>) outs(%out : memref<?xi8>) {
    ^bb0(%v: f32, %unused: i8):
      %d = arith.divf %v, %scale : f32
      %positive = arith.cmpf oge, %d, %zero : f32
      %round = arith.select %positive, %half, %nhalf : f32
      %r = arith.addf %d, %round : f32
      %lo = arith.maxnumf %r, %low : f32
      %hi = arith.minnumf %lo, %high : f32
      %i = arith.fptosi %hi : f32 to i32
      %q = arith.trunci %i : i32 to i8
      linalg.yield %q : i8
  }
  memref.dealloc %abs : memref<?xf32>
  memref.dealloc %max : memref<f32>
  return %out, %scale : memref<?xi8>, f32
}
func.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c36 = arith.constant 36 : index
  %c40 = arith.constant 40 : index
  %zero = arith.constant 0 : i32
  %twenty = arith.constant 20 : i32
  %fzero = arith.constant 0.0 : f32
  %fone = arith.constant 1.0 : f32
  %half = arith.constant 0.5 : f32
  %nhalf = arith.constant -0.5 : f32
  %factor = arith.constant 0.49999997 : f32
  %high = arith.constant 127.0 : f32
  %low = arith.constant -127.0 : f32
  %input = memref.alloc() : memref<40xf32>
  %errors = scf.for %test = %c0 to %c2 step %c1 iter_args(%total = %zero) -> i32 {
    %allzero = arith.cmpi eq, %test, %c0 : index
    scf.for %i = %c0 to %c40 step %c1 {
      %ii = arith.index_cast %i : index to i32
      %signed = arith.subi %ii, %twenty : i32
      %f = arith.sitofp %signed : i32 to f32
      %v = arith.mulf %f, %factor : f32
      %value = arith.select %allzero, %fzero, %v : f32
      memref.store %value, %input[%i] : memref<40xf32>
    }
    %e = scf.for %n = %c1 to %c36 step %c1 iter_args(%count = %total) -> i32 {
      %sub = memref.subview %input[3] [%n] [1] : memref<40xf32> to memref<?xf32, strided<[1], offset: 3>>
      %x = memref.cast %sub : memref<?xf32, strided<[1], offset: 3>> to memref<?xf32, strided<[1], offset: ?>>
      %out, %scale = func.call @compute(%x) : (memref<?xf32, strided<[1], offset: ?>>) -> (memref<?xi8>, f32)
      %maximum = scf.for %i = %c0 to %n step %c1 iter_args(%m = %fzero) -> f32 {
        %v = memref.load %x[%i] : memref<?xf32, strided<[1], offset: ?>>
        %a = math.absf %v : f32
        %next = arith.maxnumf %a, %m : f32
        scf.yield %next : f32
      }
      %s = arith.divf %maximum, %high : f32
      %iszero = arith.cmpf oeq, %maximum, %fzero : f32
      %expectedscale = arith.select %iszero, %fone, %s : f32
      %wrongscale = arith.cmpf une, %expectedscale, %scale : f32
      %delta = arith.extui %wrongscale : i1 to i32
      %start = arith.addi %count, %delta : i32
      %err = scf.for %j = %c0 to %n step %c1 iter_args(%current = %start) -> i32 {
        %v = memref.load %x[%j] : memref<?xf32, strided<[1], offset: ?>>
        %d = arith.divf %v, %expectedscale : f32
        %positive = arith.cmpf oge, %d, %fzero : f32
        %round = arith.select %positive, %half, %nhalf : f32
        %r = arith.addf %d, %round : f32
        %lo = arith.maxnumf %r, %low : f32
        %hi = arith.minnumf %lo, %high : f32
        %ii = arith.fptosi %hi : f32 to i32
        %q = arith.trunci %ii : i32 to i8
        %got = memref.load %out[%j] : memref<?xi8>
        %wrong = arith.cmpi ne, %q, %got : i8
        %change = arith.extui %wrong : i1 to i32
        %next = arith.addi %current, %change : i32
        scf.yield %next : i32
      }
      memref.dealloc %out : memref<?xi8>
      scf.yield %err : i32
    }
    scf.yield %e : i32
  }
  memref.dealloc %input : memref<40xf32>
  return %errors : i32
}
