// RUN: buddy-opt %s -polynomial-approximate-math | FileCheck %s --check-prefix=EXP
// RUN: buddy-opt %s -polynomial-approximate-math='ops=math.exp' | FileCheck %s --check-prefix=EXP
// RUN: buddy-opt %s -polynomial-approximate-math='ops=tanh' | FileCheck %s --check-prefix=TANH

func.func @scalar_exp(%x: f32) -> f32 {
  %y = math.exp %x : f32
  return %y : f32
}

func.func @vector_exp(%x: vector<8xf32>) -> vector<8xf32> {
  %y = math.exp %x : vector<8xf32>
  return %y : vector<8xf32>
}

// EXP-LABEL: func.func @scalar_exp
// EXP-NOT: math.exp
// EXP: math.fma
// EXP-LABEL: func.func @vector_exp
// EXP-NOT: math.exp
// EXP: math.fma
// TANH-LABEL: func.func @scalar_exp
// TANH: math.exp
// TANH-LABEL: func.func @vector_exp
// TANH: math.exp

// RUN: buddy-opt %s -polynomial-approximate-math='ops=log,log2,sin,cos,atan2,cbrt' | FileCheck %s --check-prefix=FIX
// RUN: buddy-opt %s -polynomial-approximate-math='ops=atan2' | FileCheck %s --check-prefix=ATAN2

// FIX-LABEL: func.func @scalar_log
// FIX: arith.cmpf olt
// FIX: arith.mulf
// FIX: arith.select
// FIX: arith.bitcast
// FIX: math.fma
// FIX-NOT: math.log
// FIX: return
func.func @scalar_log(%x: f32) -> f32 {
  %r = math.log %x : f32
  return %r : f32
}

// FIX-LABEL: func.func @vector_log2
// FIX: arith.cmpf olt
// FIX: arith.mulf
// FIX: arith.select
// FIX: math.fma
// FIX-NOT: math.log2
// FIX: return
func.func @vector_log2(%x: vector<4xf32>) -> vector<4xf32> {
  %r = math.log2 %x : vector<4xf32>
  return %r : vector<4xf32>
}

// The safe operand is selected before conversion to integer. Constants and
// predicates must retain the scalable shape.
// FIX-LABEL: func.func @scalable_sin
// FIX: math.absf
// FIX: arith.cmpi ugt
// FIX: vector.reduction <or>
// FIX: scf.if
// FIX: scf.for
// FIX: vector.extract
// FIX: arith.muli {{.*}} : vector<[4]xi64>
// FIX: scf.yield
// FIX: arith.select
// FIX: arith.mulf
// FIX: math.roundeven
// FIX: math.fma
// FIX: math.fma
// FIX: math.fma
// FIX: arith.fptosi {{.*}} to vector<[4]xi32>
// FIX: scf.yield
// FIX: } else {
// FIX: math.fma
// FIX-NOT: arith.muli
// FIX: scf.yield
// FIX: arith.cmpf uge
// FIX-NOT: math.sin
// FIX: return
func.func @scalable_sin(%x: vector<[4]xf32>) -> vector<[4]xf32> {
  %r = math.sin %x : vector<[4]xf32>
  return %r : vector<[4]xf32>
}

// FIX-LABEL: func.func @half_cos
// FIX: arith.extf
// FIX: math.fma
// FIX: arith.fptosi
// FIX: arith.truncf
// FIX-NOT: math.cos
// FIX: return
func.func @half_cos(%x: vector<4xf16>) -> vector<4xf16> {
  %r = math.cos %x : vector<4xf16>
  return %r : vector<4xf16>
}

// atan2 alone also expands its internal atan; atan is not in this ops list.
// FIX-LABEL: func.func @bf16_atan2
// FIX: arith.extf
// FIX: math.absf
// FIX: arith.divf
// FIX: [[T:%.*]] = arith.divf
// FIX: [[TT:%.*]] = arith.mulf [[T]], [[T]]
// FIX: arith.addf [[TT]],
// FIX: arith.cmpi slt
// FIX: math.copysign
// FIX: arith.truncf
// FIX-NOT: math.atan
// FIX: return
// ATAN2-LABEL: func.func @bf16_atan2
// ATAN2-NOT: math.atan
// ATAN2: return
func.func @bf16_atan2(%y: bf16, %x: bf16) -> bf16 {
  %r = math.atan2 %y, %x : bf16
  return %r : bf16
}

// FIX-LABEL: func.func @scalar_cbrt
// FIX: math.absf
// FIX: arith.cmpf olt
// FIX: arith.mulf
// FIX: arith.select
// FIX: arith.cmpf uge
// FIX: arith.select
// FIX: arith.bitcast
// FIX: arith.divf
// FIX: math.fma
// FIX: arith.divf
// FIX: math.fma
// FIX: arith.divf
// FIX: math.fma
// FIX: math.copysign
// FIX-NOT: math.cbrt
// FIX: return
func.func @scalar_cbrt(%x: f32) -> f32 {
  %r = math.cbrt %x : f32
  return %r : f32
}

// FIX-LABEL: func.func @scalable_cbrt
// FIX: arith.shrui
// FIX: math.copysign {{.*}} : vector<[4]xf32>
// FIX-NOT: math.cbrt
// FIX: return
func.func @scalable_cbrt(%x: vector<[4]xf32>) -> vector<[4]xf32> {
  %r = math.cbrt %x : vector<[4]xf32>
  return %r : vector<[4]xf32>
}

// The new rules leave unsupported precision and unselected ops alone.
// FIX-LABEL: func.func @double_log
// FIX: math.log {{.*}} : f64
// FIX: return
func.func @double_log(%x: f64) -> f64 {
  %r = math.log %x : f64
  return %r : f64
}

// ATAN2-LABEL: func.func @unselected_atan
// ATAN2: math.atan
// ATAN2: return
func.func @unselected_atan(%x: f32) -> f32 {
  %r = math.atan %x : f32
  return %r : f32
}

// FIX-LABEL: func.func @scalar_sin
// FIX: arith.cmpi ugt
// FIX-NOT: vector.reduction
// FIX: scf.if
// FIX: arith.muli
// FIX: } else {
// FIX: math.fma
// FIX-NOT: arith.muli
// FIX: scf.yield
// FIX: return
func.func @scalar_sin(%x: f32) -> f32 {
  %r = math.sin %x : f32
  return %r : f32
}
