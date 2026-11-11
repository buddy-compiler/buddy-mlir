// RUN: buddy-opt %s -split-input-file -verify-diagnostics

// A vir.set_vl without results runs once per chunk of lanes, chunks the
// lowering chooses: a scalar it writes to memory from outside the region is
// stored only as an accumulator (a vir.reduce of a memref.load, in the
// region, of the element stored).

// A count of the runs of the region depends on the chunks.
func.func @count(%a: memref<51xf32>, %o: memref<51xf32>, %c: memref<1xi32>) {
  %vl = arith.constant 51 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : i32
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<51xf32> -> !vir.vec<?xf32>
    vir.store %x, %o[] : !vir.vec<?xf32> -> memref<51xf32>
    %old = memref.load %c[%zero] : memref<1xi32>
    %next = arith.addi %old, %one : i32
    // expected-error @below {{'memref.store' op writes a scalar that is not the result of a vir.reduce in a vir.set_vl without results}}
    memref.store %next, %c[%zero] : memref<1xi32>
    vector.yield
  }
  return
}

// -----

// An accumulator loaded before the region: each chunk combines its lanes
// with the value of before the region, the chunks before are lost.
func.func @acc_outside(%a: memref<51xf32>, %s: memref<1xf32>) {
  %vl = arith.constant 51 : index
  %zero = arith.constant 0 : index
  %acc = memref.load %s[%zero] : memref<1xf32>
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<51xf32> -> !vir.vec<?xf32>
    %r = vir.reduce %x, %acc {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    // expected-error @below {{'memref.store' op stores a vir.reduce whose accumulator is not a memref.load in the region of the element stored}}
    memref.store %r, %s[%zero] : memref<1xf32>
    vector.yield
  }
  return
}

// -----

// An accumulator loaded from another element than the one stored.
func.func @acc_other(%a: memref<51xf32>, %s: memref<2xf32>) {
  %vl = arith.constant 51 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<51xf32> -> !vir.vec<?xf32>
    %acc = memref.load %s[%zero] : memref<2xf32>
    %r = vir.reduce %x, %acc {kind = "maxnum"} : !vir.vec<?xf32>, f32 -> f32
    // expected-error @below {{'memref.store' op stores a vir.reduce whose accumulator is not a memref.load in the region of the element stored}}
    memref.store %r, %s[%one] : memref<2xf32>
    vector.yield
  }
  return
}

// -----

// An op of unknown memory effects.
func.func private @log(f32)
func.func @call(%a: memref<51xf32>, %o: memref<51xf32>) {
  %vl = arith.constant 51 : index
  %one = arith.constant 1.0 : f32
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<51xf32> -> !vir.vec<?xf32>
    vir.store %x, %o[] : !vir.vec<?xf32> -> memref<51xf32>
    // expected-error @below {{'func.call' op has unknown memory effects}}
    func.call @log(%one) : (f32) -> ()
    vector.yield
  }
  return
}

// -----

// An accumulator: each chunk combines its lanes with what the chunks before
// stored (add or maxnum).
func.func @accumulate(%a: memref<51xf32>, %s: memref<1xf32>) {
  %vl = arith.constant 51 : index
  %zero = arith.constant 0 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<51xf32> -> !vir.vec<?xf32>
    %acc = memref.load %s[%zero] : memref<1xf32>
    %r = vir.reduce %x, %acc {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    memref.store %r, %s[%zero] : memref<1xf32>
    vector.yield
  }
  return
}

// -----

// A scalar of a buffer of the region itself.
func.func @local(%a: memref<51xf32>, %o: memref<51xf32>) {
  %vl = arith.constant 51 : index
  %one = arith.constant 1.0 : f32
  vir.set_vl %vl : index {
    %t = memref.alloca() : memref<f32>
    memref.store %one, %t[] : memref<f32>
    %x = vir.load %a[] : memref<51xf32> -> !vir.vec<?xf32>
    vir.store %x, %o[] : !vir.vec<?xf32> -> memref<51xf32>
    vector.yield
  }
  return
}

// -----

// A region with results computes on whole vectors once: any write.
func.func @with_results(%a: memref<8xf32>, %c: memref<1xi32>) -> f32 {
  %vl = arith.constant 8 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : i32
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<8xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    memref.store %one, %c[%zero] : memref<1xi32>
    vector.yield %s : f32
  } -> f32
  return %r : f32
}
