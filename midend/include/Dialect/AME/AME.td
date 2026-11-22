//====------ AME.td - AME dialect operation definitions --- tablegen ------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Attached Matrix Extension dialect for ZTT v0.6.
//
//===----------------------------------------------------------------------===//

#ifndef AME_DIALECT
#define AME_DIALECT

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// AME Dialect Definition
//===----------------------------------------------------------------------===//

def AME_Dialect : Dialect {
  let name = "ame";
  let cppNamespace = "::buddy::ame";
  let summary = "Attached Matrix Extension operations based on ZTT v0.6";
  let description = [{
    The AME dialect models the provisional Attached Matrix Extension (ZTT v0.6)
    implemented by the RISC-V XAME backend. It provides SSA operations for
    acquisition, descriptors, matrix memory access, M/ACC transfers,
    element-wise computation and matrix multiply-accumulate.

    Matrix and accumulator values use the LLVM target-extension types
    `riscv.ztt.matrix` and `riscv.ztt.acc`. Destination passthrough operands
    preserve register contents and descriptor dependencies. The caller must
    own the AME and supply descriptors consistent with the SSA value types.

    Ordinary operations named `ame.<mnemonic>` provide typed SSA interfaces
    and custom assembly syntax. `--lower-ame` converts them through the
    corresponding intrinsic bridge operations.

    Operations named `ame.intr.<mnemonic>` lower to `llvm.call_intrinsic`
    through `--lower-ame`, or translate directly with `buddy-translate`.
    Both paths emit the corresponding `llvm.riscv.ztt.*` intrinsic.
  }];
}

//===----------------------------------------------------------------------===//
// AME Types and Operation Base Class
//===----------------------------------------------------------------------===//

def AME_Matrix : Type<
  And<[LLVM_AnyTargetExt.predicate,
       CPred<"::llvm::cast<::mlir::LLVM::LLVMTargetExtType>($_self)"
             ".getExtTypeName() == \"riscv.ztt.matrix\"">]>,
  "AME matrix target-extension type">;

def AME_Accumulator : Type<
  And<[LLVM_AnyTargetExt.predicate,
       CPred<"::llvm::cast<::mlir::LLVM::LLVMTargetExtType>($_self)"
             ".getExtTypeName() == \"riscv.ztt.acc\"">]>,
  "AME accumulator target-extension type">;

// The backend legalizes these integer widths to architectural XLEN operands.
def AME_Scalar : AnyTypeOf<[I8, I16, I32, I64]>;

// AME operations access ordered architectural state, including descriptors.
// Leave effects conservative even for element-wise arithmetic.
class AME_Op<string mnemonic, list<Trait> traits = []> :
  Op<AME_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// AME Instruction Operations (ZTT v0.6)
//
// Operand order and overloads follow IntrinsicsRISCVXAME.td. Only a destination
// passthrough must match its result; independently overloaded sources may differ.
// Zip/unzip pairs are represented by a single, literal LLVM struct result.
//===----------------------------------------------------------------------===//

def AME_AmeAcquireOp : AME_Op<"acquire", [AllTypesMatch<["rs1", "res"]>]> {
  let summary = "Acquire the attached matrix extension";
  let description = [{
    Requests ownership using the integer acquisition descriptor and returns
    the integer acquisition status. Success is encoded in bit zero of the
    result; test that bit rather than testing whether the complete result is
    nonzero. Ownership transitions also order ordinary state-save and state-
    restore memory accesses. Scalar operands and results may use i8, i16, i32
    or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Scalar:$rs1);
  let results = (outs AME_Scalar:$res);
  let assemblyFormat =
    "$rs1 attr-dict `:` functional-type(operands, results)";
}

def AME_AmeReleaseOp : AME_Op<"release"> {
  let summary = "Release the attached matrix extension";
  let description = [{
    Releases AME ownership. This operation has no operands or results and
    orders ownership state and ordinary state-save and state-restore memory
    accesses. No live AME SSA value may rely on an ownership transition
    preserving its register state.
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat =
    "attr-dict `:` functional-type(operands, results)";
}

def AME_AgettypOp : AME_Op<"agettyp"> {
  let summary = "Read an AME accumulator datatype descriptor";
  let description = [{
    Reads the runtime datatype descriptor coupled to the input accumulator and
    returns it as an integer value. This reads descriptor state; it does not
    extract an element of the accumulator. The caller must own the AME.
    Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties. Scalar operands and
    results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Accumulator:$acc);
  let results = (outs AME_Scalar:$res);
  let assemblyFormat =
    "$acc attr-dict `:` functional-type(operands, results)";
}

def AME_AsettypOp : AME_Op<"asettyp", [AllTypesMatch<["acc_in", "res"]>]> {
  let summary = "Set an AME accumulator datatype descriptor";
  let description = [{
    Programs the runtime datatype descriptor of the destination accumulator
    from the integer descriptor operand and returns its updated SSA value. The
    old destination is explicit: an unsupported descriptor preserves its data
    and descriptor. On success the instruction clears its architectural
    destination span; a larger SSA register group does not extend that write
    footprint. Use an undefined old value only for initial construction when
    no prior state must survive. The caller must own the AME. Runtime datatype
    descriptors must agree with the static matrix or accumulator types;
    signedness, rounding, saturation and operation availability remain
    architectural runtime properties. Scalar operands and results may use i8,
    i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Accumulator:$acc_in, AME_Scalar:$rs1);
  let results = (outs AME_Accumulator:$res);
  let assemblyFormat =
    "$acc_in `,` $rs1 attr-dict `:` functional-type(operands, results)";
}

def AME_MabsEwOp : AME_Op<"mabs.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise absolute value";
  let description = [{
    Selects the ZTT v0.6 `mabs.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MabsdiffEwOp : AME_Op<"mabsdiff.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise absolute difference";
  let description = [{
    Selects the ZTT v0.6 `mabsdiff.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MabsdiffEwXOp : AME_Op<"mabsdiff.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise absolute difference with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mabsdiff.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MaddEwOp : AME_Op<"madd.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise addition";
  let description = [{
    Selects the ZTT v0.6 `madd.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MaddEwXOp : AME_Op<"madd.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise addition with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `madd.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MandEwOp : AME_Op<"mand.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise bitwise AND";
  let description = [{
    Selects the ZTT v0.6 `mand.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MandEwXOp : AME_Op<"mand.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise bitwise AND with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mand.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MandnotEwOp : AME_Op<"mandnot.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise bitwise AND-NOT";
  let description = [{
    Selects the ZTT v0.6 `mandnot.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MandnotEwXOp : AME_Op<"mandnot.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise bitwise AND-NOT with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mandnot.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_McmovgeEwOp : AME_Op<"mcmovge.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise conditional move (GE)";
  let description = [{
    Selects the ZTT v0.6 `mcmovge.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_McmovltEwOp : AME_Op<"mcmovlt.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise conditional move (LT)";
  let description = [{
    Selects the ZTT v0.6 `mcmovlt.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_McmpgeEwOp : AME_Op<"mcmpge.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise comparison (GE)";
  let description = [{
    Selects the ZTT v0.6 `mcmpge.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_McmpgeEwXOp : AME_Op<"mcmpge.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise comparison (GE) with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mcmpge.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_McmpltEwOp : AME_Op<"mcmplt.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise comparison (LT)";
  let description = [{
    Selects the ZTT v0.6 `mcmplt.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_McmpltEwXOp : AME_Op<"mcmplt.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise comparison (LT) with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mcmplt.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_McolbcastEwXOp : AME_Op<"mcolbcast.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise column broadcast with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mcolbcast.ew.x` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_McolgatherEwOp : AME_Op<"mcolgather.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise column gather";
  let description = [{
    Selects the ZTT v0.6 `mcolgather.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_McolidEwOp : AME_Op<"mcolid.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise column index generation";
  let description = [{
    Selects the ZTT v0.6 `mcolid.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in attr-dict `:` functional-type(operands, results)";
}

def AME_McolshiftEwXOp : AME_Op<"mcolshift.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise column shift with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mcolshift.ew.x` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_McolunzipEwOp : AME_Op<"mcolunzip.ew"> {
  let summary = "Unzip two AME matrices by column";
  let description = [{
    Selects `mcolunzip.ew` and updates both matrix operands. The operation
    returns one LLVM struct containing the two updated matrix values in
    operand order. Extract the fields with `llvm.extractvalue` to use the
    matrices separately. Both inputs and the two struct fields have the same
    non-packed matrix type. The inputs carry the existing data and descriptors
    for both architectural destinations. The caller must own the AME. Runtime
    datatype descriptors must agree with the static matrix types; signedness,
    rounding, saturation and operation availability remain architectural
    runtime properties.
  }];
  let arguments = (ins AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs LLVM_AnyStruct:$res);
  let assemblyFormat =
    "$ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def AME_McolzipEwOp : AME_Op<"mcolzip.ew"> {
  let summary = "Zip two AME matrices by column";
  let description = [{
    Selects `mcolzip.ew` and updates both matrix operands. The operation
    returns one LLVM struct containing the two updated matrix values in
    operand order. Extract the fields with `llvm.extractvalue` to use the
    matrices separately. Both inputs and the two struct fields have the same
    non-packed matrix type. The inputs carry the existing data and descriptors
    for both architectural destinations. The caller must own the AME. Runtime
    datatype descriptors must agree with the static matrix types; signedness,
    rounding, saturation and operation availability remain architectural
    runtime properties.
  }];
  let arguments = (ins AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs LLVM_AnyStruct:$res);
  let assemblyFormat =
    "$ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def AME_MconvEwOp : AME_Op<"mconv.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise datatype conversion";
  let description = [{
    Selects the ZTT v0.6 `mconv.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MbcastMXOp : AME_Op<"mbcast.m.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME scalar broadcast";
  let description = [{
    Selects the ZTT v0.6 `mbcast.m.x` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Scalar:$rs2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $rs2 attr-dict `:` functional-type(operands, results)";
}

def AME_McosEwOp : AME_Op<"mcos.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise cosine";
  let description = [{
    Selects the ZTT v0.6 `mcos.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mexp2EwOp : AME_Op<"mexp2.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise base-two exponential";
  let description = [{
    Selects the ZTT v0.6 `mexp2.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MfrintmEwOp : AME_Op<"mfrintm.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise frintm rounding";
  let description = [{
    Selects the ZTT v0.6 `mfrintm.ew` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MfrintnEwOp : AME_Op<"mfrintn.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise frintn rounding";
  let description = [{
    Selects the ZTT v0.6 `mfrintn.ew` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MfrintpEwOp : AME_Op<"mfrintp.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise frintp rounding";
  let description = [{
    Selects the ZTT v0.6 `mfrintp.ew` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MfrintzEwOp : AME_Op<"mfrintz.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise frintz rounding";
  let description = [{
    Selects the ZTT v0.6 `mfrintz.ew` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MgettypOp : AME_Op<"mgettyp"> {
  let summary = "Read an AME matrix datatype descriptor";
  let description = [{
    Reads the runtime datatype descriptor coupled to the input matrix and
    returns it as an integer value. This reads descriptor state; it does not
    extract an element of the matrix. The caller must own the AME. Runtime
    datatype descriptors must agree with the static matrix or accumulator
    types; signedness, rounding, saturation and operation availability remain
    architectural runtime properties. Scalar operands and results may use i8,
    i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$ms1);
  let results = (outs AME_Scalar:$res);
  let assemblyFormat =
    "$ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MhdiffEwOp : AME_Op<"mhdiff.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise hdiff";
  let description = [{
    Selects the ZTT v0.6 `mhdiff.ew` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MhdiffEwXOp : AME_Op<"mhdiff.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise hdiff with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mhdiff.ew.x` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The scalar integer bits are
    interpreted using the scalar datatype state in `amestype`, which must be
    programmed separately from matrix descriptors. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties. Scalar operands and
    results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MldexpEwOp : AME_Op<"mldexp.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise ldexp";
  let description = [{
    Selects the ZTT v0.6 `mldexp.ew` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MldexpEwXOp : AME_Op<"mldexp.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise ldexp with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mldexp.ew.x` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The scalar integer bits are
    interpreted using the scalar datatype state in `amestype`, which must be
    programmed separately from matrix descriptors. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties. Scalar operands and
    results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MldexpaccEwOp : AME_Op<"mldexpacc.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise ldexp accumulation";
  let description = [{
    Selects the ZTT v0.6 `mldexpacc.ew` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MldexpaccEwXOp : AME_Op<"mldexpacc.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise ldexp accumulation with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mldexpacc.ew.x` instruction and returns the updated
    M matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The scalar integer bits are
    interpreted using the scalar datatype state in `amestype`, which must be
    programmed separately from matrix descriptors. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties. Scalar operands and
    results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mlog2EwOp : AME_Op<"mlog2.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise base-two logarithm";
  let description = [{
    Selects the ZTT v0.6 `mlog2.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mlog2subEwOp : AME_Op<"mlog2sub.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise log2-subtract";
  let description = [{
    Selects the ZTT v0.6 `mlog2sub.ew` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_Mlog2subEwXOp : AME_Op<"mlog2sub.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise log2-subtract with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mlog2sub.ew.x` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The scalar integer bits are
    interpreted using the scalar datatype state in `amestype`, which must be
    programmed separately from matrix descriptors. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties. Scalar operands and
    results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mls1rOp : AME_Op<"mls.1r", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Load an AME matrix using the one-register instruction";
  let description = [{
    Selects `mls.1r` and loads matrix data from the supplied pointer. The old
    destination supplies its already programmed descriptor and is returned
    unchanged when the architectural operation does not write the destination;
    an undefined input does not program a datatype descriptor. This whole-
    register instruction transfers exactly one physical M register. The caller
    must own the AME. Runtime datatype descriptors must agree with the static
    matrix or accumulator types; signedness, rounding, saturation and
    operation availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, LLVM_DefaultPointer:$base);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $base attr-dict `:` functional-type(operands, results)";
}

def AME_MlsCmOp : AME_Op<"mls.cm", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Load an AME matrix using the column-major instruction";
  let description = [{
    Selects `mls.cm` and loads matrix data from the supplied pointer. The old
    destination supplies its already programmed descriptor and is returned
    unchanged when the architectural operation does not write the destination;
    an undefined input does not program a datatype descriptor. The caller must
    own the AME. Runtime datatype descriptors must agree with the static
    matrix or accumulator types; signedness, rounding, saturation and
    operation availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, LLVM_DefaultPointer:$base);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $base attr-dict `:` functional-type(operands, results)";
}

def AME_MlsRmOp : AME_Op<"mls.rm", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Load an AME matrix using the row-major instruction";
  let description = [{
    Selects `mls.rm` and loads matrix data from the supplied pointer. The old
    destination supplies its already programmed descriptor and is returned
    unchanged when the architectural operation does not write the destination;
    an undefined input does not program a datatype descriptor. The caller must
    own the AME. Runtime datatype descriptors must agree with the static
    matrix or accumulator types; signedness, rounding, saturation and
    operation availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, LLVM_DefaultPointer:$base);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $base attr-dict `:` functional-type(operands, results)";
}

def AME_MlsStOp : AME_Op<"mls.st", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Load an AME matrix using the strided instruction";
  let description = [{
    Selects `mls.st` and loads matrix data from the supplied pointer. The old
    destination supplies its already programmed descriptor and is returned
    unchanged when the architectural operation does not write the destination;
    an undefined input does not program a datatype descriptor. The trailing
    integer operand supplies the instruction stride; the lowering forwards it
    without reshaping or synthesizing a loop. The caller must own the AME.
    Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties. Scalar operands and
    results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, LLVM_DefaultPointer:$base, AME_Scalar:$rs1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $base `,` $rs1 attr-dict `:` functional-type(operands, results)";
}

def AME_MlsTstOp : AME_Op<"mls.tst", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Load an AME matrix using the transposed-strided instruction";
  let description = [{
    Selects `mls.tst` and loads matrix data from the supplied pointer. The old
    destination supplies its already programmed descriptor and is returned
    unchanged when the architectural operation does not write the destination;
    an undefined input does not program a datatype descriptor. The trailing
    integer operand supplies the instruction stride; the lowering forwards it
    without reshaping or synthesizing a loop. The caller must own the AME.
    Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties. Scalar operands and
    results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, LLVM_DefaultPointer:$base, AME_Scalar:$rs1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $base `,` $rs1 attr-dict `:` functional-type(operands, results)";
}

def AME_MmaxEwOp : AME_Op<"mmax.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise maximum";
  let description = [{
    Selects the ZTT v0.6 `mmax.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MmaxEwXOp : AME_Op<"mmax.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise maximum with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mmax.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MmeanEwOp : AME_Op<"mmean.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise mean";
  let description = [{
    Selects the ZTT v0.6 `mmean.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MmeanEwXOp : AME_Op<"mmean.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise mean with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mmean.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MminEwOp : AME_Op<"mmin.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise minimum";
  let description = [{
    Selects the ZTT v0.6 `mmin.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MminEwXOp : AME_Op<"mmin.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise minimum with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mmin.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MmovMAOp : AME_Op<"mmov.m.a", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Move an AME ACC accumulator value to M matrix";
  let description = [{
    Selects `mmov.m.a` with an explicit old M matrix destination and the ACC
    accumulator source. The result has exactly the destination type. M/ACC
    transfers access the packed accumulator group dictated by the datatype and
    profile; a single-square ACC SSA type does not reduce that architectural
    transfer footprint. The caller must own the AME. Runtime datatype
    descriptors must agree with the static matrix or accumulator types;
    signedness, rounding, saturation and operation availability remain
    architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Accumulator:$acc);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $acc attr-dict `:` functional-type(operands, results)";
}

def AME_MmovAMOp : AME_Op<"mmov.a.m", [AllTypesMatch<["acc_in", "res"]>]> {
  let summary = "Move an AME M matrix value to ACC accumulator";
  let description = [{
    Selects `mmov.a.m` with an explicit old ACC accumulator destination and
    the M matrix source. The result has exactly the destination type. M/ACC
    transfers access the packed accumulator group dictated by the datatype and
    profile; a single-square ACC SSA type does not reduce that architectural
    transfer footprint. The caller must own the AME. Runtime datatype
    descriptors must agree with the static matrix or accumulator types;
    signedness, rounding, saturation and operation availability remain
    architectural runtime properties.
  }];
  let arguments = (ins AME_Accumulator:$acc_in, AME_Matrix:$ms1);
  let results = (outs AME_Accumulator:$res);
  let assemblyFormat =
    "$acc_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MmovMMOp : AME_Op<"mmov.m.m", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Move an AME M matrix value to M matrix";
  let description = [{
    Selects `mmov.m.m` with an explicit old M destination and the M source.
    The result has exactly the destination type. The destination carries its
    existing descriptor and preserves architectural no-write behavior for
    unsupported tuples. The caller must own the AME. Runtime datatype
    descriptors must agree with the static matrix or accumulator types;
    signedness, rounding, saturation and operation availability remain
    architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmove8MXOp : AME_Op<"mmove8.m.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Move a scalar integer to a 8-bit matrix element";
  let description = [{
    Selects `mmove8.m.x`. The old destination, scalar integer and integer
    index are forwarded in the intrinsic operand order; the result is the
    updated matrix. The move operates on one physical M register and preserves
    the other destination contents according to the architecture. The caller
    must own the AME. Runtime datatype descriptors must agree with the static
    matrix or accumulator types; signedness, rounding, saturation and
    operation availability remain architectural runtime properties. Scalar
    operands and results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Scalar:$rs2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $rs2 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmove16MXOp : AME_Op<"mmove16.m.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Move a scalar integer to a 16-bit matrix element";
  let description = [{
    Selects `mmove16.m.x`. The old destination, scalar integer and integer
    index are forwarded in the intrinsic operand order; the result is the
    updated matrix. The move operates on one physical M register and preserves
    the other destination contents according to the architecture. The caller
    must own the AME. Runtime datatype descriptors must agree with the static
    matrix or accumulator types; signedness, rounding, saturation and
    operation availability remain architectural runtime properties. Scalar
    operands and results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Scalar:$rs2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $rs2 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmove32MXOp : AME_Op<"mmove32.m.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Move a scalar integer to a 32-bit matrix element";
  let description = [{
    Selects `mmove32.m.x`. The old destination, scalar integer and integer
    index are forwarded in the intrinsic operand order; the result is the
    updated matrix. The move operates on one physical M register and preserves
    the other destination contents according to the architecture. The caller
    must own the AME. Runtime datatype descriptors must agree with the static
    matrix or accumulator types; signedness, rounding, saturation and
    operation availability remain architectural runtime properties. Scalar
    operands and results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Scalar:$rs2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $rs2 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmove64MXOp : AME_Op<"mmove64.m.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Move a scalar integer to a 64-bit matrix element";
  let description = [{
    Selects `mmove64.m.x`. The old destination, scalar integer and integer
    index are forwarded in the intrinsic operand order; the result is the
    updated matrix. The move operates on one physical M register and preserves
    the other destination contents according to the architecture. The caller
    must own the AME. Runtime datatype descriptors must agree with the static
    matrix or accumulator types; signedness, rounding, saturation and
    operation availability remain architectural runtime properties. Scalar
    operands and results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Scalar:$rs2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $rs2 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmove8XMOp : AME_Op<"mmove8.x.m", [AllTypesMatch<["rs1", "res"]>]> {
  let summary = "Move a 8-bit matrix element to a scalar integer";
  let description = [{
    Selects the `mmove8.x.m` scalar extraction instruction. The input matrix
    and integer index remain in the intrinsic operand order, and the result is
    an integer value even for a 8-bit move. The scalar debug move operates on
    one physical M register. The caller must own the AME. Runtime datatype
    descriptors must agree with the static matrix or accumulator types;
    signedness, rounding, saturation and operation availability remain
    architectural runtime properties. Scalar operands and results may use i8,
    i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$ms1, AME_Scalar:$rs1);
  let results = (outs AME_Scalar:$res);
  let assemblyFormat =
    "$ms1 `,` $rs1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmove16XMOp : AME_Op<"mmove16.x.m", [AllTypesMatch<["rs1", "res"]>]> {
  let summary = "Move a 16-bit matrix element to a scalar integer";
  let description = [{
    Selects the `mmove16.x.m` scalar extraction instruction. The input matrix
    and integer index remain in the intrinsic operand order, and the result is
    an integer value even for a 16-bit move. The scalar debug move operates on
    one physical M register. The caller must own the AME. Runtime datatype
    descriptors must agree with the static matrix or accumulator types;
    signedness, rounding, saturation and operation availability remain
    architectural runtime properties. Scalar operands and results may use i8,
    i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$ms1, AME_Scalar:$rs1);
  let results = (outs AME_Scalar:$res);
  let assemblyFormat =
    "$ms1 `,` $rs1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmove32XMOp : AME_Op<"mmove32.x.m", [AllTypesMatch<["rs1", "res"]>]> {
  let summary = "Move a 32-bit matrix element to a scalar integer";
  let description = [{
    Selects the `mmove32.x.m` scalar extraction instruction. The input matrix
    and integer index remain in the intrinsic operand order, and the result is
    an integer value even for a 32-bit move. The scalar debug move operates on
    one physical M register. The caller must own the AME. Runtime datatype
    descriptors must agree with the static matrix or accumulator types;
    signedness, rounding, saturation and operation availability remain
    architectural runtime properties. Scalar operands and results may use i8,
    i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$ms1, AME_Scalar:$rs1);
  let results = (outs AME_Scalar:$res);
  let assemblyFormat =
    "$ms1 `,` $rs1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmove64XMOp : AME_Op<"mmove64.x.m", [AllTypesMatch<["rs1", "res"]>]> {
  let summary = "Move a 64-bit matrix element to a scalar integer";
  let description = [{
    Selects the `mmove64.x.m` scalar extraction instruction. The input matrix
    and integer index remain in the intrinsic operand order, and the result is
    an integer value even for a 64-bit move. The scalar debug move operates on
    one physical M register. The caller must own the AME. Runtime datatype
    descriptors must agree with the static matrix or accumulator types;
    signedness, rounding, saturation and operation availability remain
    architectural runtime properties. Scalar operands and results may use i8,
    i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$ms1, AME_Scalar:$rs1);
  let results = (outs AME_Scalar:$res);
  let assemblyFormat =
    "$ms1 `,` $rs1 attr-dict `:` functional-type(operands, results)";
}

def AME_MmulEwOp : AME_Op<"mmul.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise multiplication";
  let description = [{
    Selects the ZTT v0.6 `mmul.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MmulEwXOp : AME_Op<"mmul.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise multiplication with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mmul.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmulacc2dOp : AME_Op<"mmulacc.2d", [AllTypesMatch<["acc_in", "res"]>]> {
  let summary = "AME two-dimensional multiply-accumulate";
  let description = [{
    Selects the ZTT v0.6 `mmulacc.2d` instruction and returns the updated ACC
    accumulator. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Accumulator:$acc_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Accumulator:$res);
  let assemblyFormat =
    "$acc_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MmulaccEwOp : AME_Op<"mmulacc.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise multiply-accumulate";
  let description = [{
    Selects the ZTT v0.6 `mmulacc.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MmulaccEwXOp : AME_Op<"mmulacc.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise multiply-accumulate with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mmulacc.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmulaccneg2dOp : AME_Op<"mmulaccneg.2d", [AllTypesMatch<["acc_in", "res"]>]> {
  let summary = "AME two-dimensional negative multiply-accumulate";
  let description = [{
    Selects the ZTT v0.6 `mmulaccneg.2d` instruction and returns the updated
    ACC accumulator. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Accumulator:$acc_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Accumulator:$res);
  let assemblyFormat =
    "$acc_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MmulaccnegEwOp : AME_Op<"mmulaccneg.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise negative multiply-accumulate";
  let description = [{
    Selects the ZTT v0.6 `mmulaccneg.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MmulaccnegEwXOp : AME_Op<"mmulaccneg.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise negative multiply-accumulate with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mmulaccneg.ew.x` instruction and returns the updated
    M matrix. The scalar integer bits are interpreted using the scalar
    datatype state in `amestype`, which must be programmed separately from
    matrix descriptors. The old destination is explicit so that
    read/modify/write behavior, its descriptor and unsupported-tuple no-write
    behavior survive lowering. The caller must own the AME. Runtime datatype
    descriptors must agree with the static matrix or accumulator types;
    signedness, rounding, saturation and operation availability remain
    architectural runtime properties. Scalar operands and results may use i8,
    i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MmuladdEwOp : AME_Op<"mmuladd.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise multiply-add";
  let description = [{
    Selects the ZTT v0.6 `mmuladd.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MmuladdEwXOp : AME_Op<"mmuladd.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise multiply-add with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mmuladd.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmulatacc2dOp : AME_Op<"mmulatacc.2d", [AllTypesMatch<["acc_in", "res"]>]> {
  let summary = "AME two-dimensional A-transposed multiply-accumulate";
  let description = [{
    Selects the ZTT v0.6 `mmulatacc.2d` instruction and returns the updated
    ACC accumulator. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Accumulator:$acc_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Accumulator:$res);
  let assemblyFormat =
    "$acc_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmulataccneg2dOp : AME_Op<"mmulataccneg.2d", [AllTypesMatch<["acc_in", "res"]>]> {
  let summary = "AME two-dimensional negative A-transposed multiply-accumulate";
  let description = [{
    Selects the ZTT v0.6 `mmulataccneg.2d` instruction and returns the updated
    ACC accumulator. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Accumulator:$acc_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Accumulator:$res);
  let assemblyFormat =
    "$acc_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmulbtacc2dOp : AME_Op<"mmulbtacc.2d", [AllTypesMatch<["acc_in", "res"]>]> {
  let summary = "AME two-dimensional B-transposed multiply-accumulate";
  let description = [{
    Selects the ZTT v0.6 `mmulbtacc.2d` instruction and returns the updated
    ACC accumulator. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Accumulator:$acc_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Accumulator:$res);
  let assemblyFormat =
    "$acc_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_Mmulbtaccneg2dOp : AME_Op<"mmulbtaccneg.2d", [AllTypesMatch<["acc_in", "res"]>]> {
  let summary = "AME two-dimensional negative B-transposed multiply-accumulate";
  let description = [{
    Selects the ZTT v0.6 `mmulbtaccneg.2d` instruction and returns the updated
    ACC accumulator. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Accumulator:$acc_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Accumulator:$res);
  let assemblyFormat =
    "$acc_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MmulnegEwOp : AME_Op<"mmulneg.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise negated multiplication";
  let description = [{
    Selects the ZTT v0.6 `mmulneg.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MmulnegEwXOp : AME_Op<"mmulneg.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise negated multiplication with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mmulneg.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MmulsubEwOp : AME_Op<"mmulsub.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise multiply-subtract";
  let description = [{
    Selects the ZTT v0.6 `mmulsub.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MmulsubEwXOp : AME_Op<"mmulsub.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise multiply-subtract with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mmulsub.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MorEwOp : AME_Op<"mor.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise bitwise OR";
  let description = [{
    Selects the ZTT v0.6 `mor.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MorEwXOp : AME_Op<"mor.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise bitwise OR with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mor.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MornotEwOp : AME_Op<"mornot.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise bitwise OR-NOT";
  let description = [{
    Selects the ZTT v0.6 `mornot.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MornotEwXOp : AME_Op<"mornot.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise bitwise OR-NOT with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mornot.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MpackEwXOp : AME_Op<"mpack.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise packing with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mpack.ew.x` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MprefixaddColOp : AME_Op<"mprefixadd.col", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME prefix sum (column form)";
  let description = [{
    Selects the ZTT v0.6 `mprefixadd.col` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MprefixaddRowOp : AME_Op<"mprefixadd.row", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME prefix sum (row form)";
  let description = [{
    Selects the ZTT v0.6 `mprefixadd.row` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MprefixmaxColOp : AME_Op<"mprefixmax.col", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME prefix maximum (column form)";
  let description = [{
    Selects the ZTT v0.6 `mprefixmax.col` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MprefixmaxRowOp : AME_Op<"mprefixmax.row", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME prefix maximum (row form)";
  let description = [{
    Selects the ZTT v0.6 `mprefixmax.row` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MrdexpEwOp : AME_Op<"mrdexp.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise rdexp";
  let description = [{
    Selects the ZTT v0.6 `mrdexp.ew` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MrdexpaccEwOp : AME_Op<"mrdexpacc.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise rdexp accumulation";
  let description = [{
    Selects the ZTT v0.6 `mrdexpacc.ew` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MrecEwOp : AME_Op<"mrec.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise reciprocal";
  let description = [{
    Selects the ZTT v0.6 `mrec.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MreduceaddColOp : AME_Op<"mreduceadd.col", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME sum reduction (column form)";
  let description = [{
    Selects the ZTT v0.6 `mreduceadd.col` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MreduceaddRowOp : AME_Op<"mreduceadd.row", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME sum reduction (row form)";
  let description = [{
    Selects the ZTT v0.6 `mreduceadd.row` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MreducemaxColOp : AME_Op<"mreducemax.col", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME maximum reduction (column form)";
  let description = [{
    Selects the ZTT v0.6 `mreducemax.col` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MreducemaxRowOp : AME_Op<"mreducemax.row", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME maximum reduction (row form)";
  let description = [{
    Selects the ZTT v0.6 `mreducemax.row` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MreduceminColOp : AME_Op<"mreducemin.col", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME minimum reduction (column form)";
  let description = [{
    Selects the ZTT v0.6 `mreducemin.col` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MreduceminRowOp : AME_Op<"mreducemin.row", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME minimum reduction (row form)";
  let description = [{
    Selects the ZTT v0.6 `mreducemin.row` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MrowbcastEwXOp : AME_Op<"mrowbcast.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise row broadcast with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mrowbcast.ew.x` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MrowgatherEwOp : AME_Op<"mrowgather.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise row gather";
  let description = [{
    Selects the ZTT v0.6 `mrowgather.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MrowidEwOp : AME_Op<"mrowid.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise row index generation";
  let description = [{
    Selects the ZTT v0.6 `mrowid.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in attr-dict `:` functional-type(operands, results)";
}

def AME_MrowshiftEwXOp : AME_Op<"mrowshift.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise row shift with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mrowshift.ew.x` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MrowunzipEwOp : AME_Op<"mrowunzip.ew"> {
  let summary = "Unzip two AME matrices by row";
  let description = [{
    Selects `mrowunzip.ew` and updates both matrix operands. The operation
    returns one LLVM struct containing the two updated matrix values in
    operand order. Extract the fields with `llvm.extractvalue` to use the
    matrices separately. Both inputs and the two struct fields have the same
    non-packed matrix type. The inputs carry the existing data and descriptors
    for both architectural destinations. The caller must own the AME. Runtime
    datatype descriptors must agree with the static matrix types; signedness,
    rounding, saturation and operation availability remain architectural
    runtime properties.
  }];
  let arguments = (ins AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs LLVM_AnyStruct:$res);
  let assemblyFormat =
    "$ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def AME_MrowzipEwOp : AME_Op<"mrowzip.ew"> {
  let summary = "Zip two AME matrices by row";
  let description = [{
    Selects `mrowzip.ew` and updates both matrix operands. The operation
    returns one LLVM struct containing the two updated matrix values in
    operand order. Extract the fields with `llvm.extractvalue` to use the
    matrices separately. Both inputs and the two struct fields have the same
    non-packed matrix type. The inputs carry the existing data and descriptors
    for both architectural destinations. The caller must own the AME. Runtime
    datatype descriptors must agree with the static matrix types; signedness,
    rounding, saturation and operation availability remain architectural
    runtime properties.
  }];
  let arguments = (ins AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs LLVM_AnyStruct:$res);
  let assemblyFormat =
    "$ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def AME_MrsqrtEwOp : AME_Op<"mrsqrt.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise reciprocal square root";
  let description = [{
    Selects the ZTT v0.6 `mrsqrt.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MrowscataddEwOp : AME_Op<"mrowscatadd.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise row scatter-add";
  let description = [{
    Selects the ZTT v0.6 `mrowscatadd.ew` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_McolscataddEwOp : AME_Op<"mcolscatadd.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise column scatter-add";
  let description = [{
    Selects the ZTT v0.6 `mcolscatadd.ew` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MrowscatmaxEwOp : AME_Op<"mrowscatmax.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise row scatter-maximum";
  let description = [{
    Selects the ZTT v0.6 `mrowscatmax.ew` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_McolscatmaxEwOp : AME_Op<"mcolscatmax.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise column scatter-maximum";
  let description = [{
    Selects the ZTT v0.6 `mcolscatmax.ew` instruction and returns the updated
    M matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MselgeEwOp : AME_Op<"mselge.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise selection (GE)";
  let description = [{
    Selects the ZTT v0.6 `mselge.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MselltEwOp : AME_Op<"msellt.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise selection (LT)";
  let description = [{
    Selects the ZTT v0.6 `msellt.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MsettypOp : AME_Op<"msettyp", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Set an AME matrix datatype descriptor";
  let description = [{
    Programs the runtime datatype descriptor of the destination matrix from
    the integer descriptor operand and returns its updated SSA value. The old
    destination is explicit: an unsupported descriptor preserves its data and
    descriptor. On success the instruction clears its architectural
    destination span; a larger SSA register group does not extend that write
    footprint. Use an undefined old value only for initial construction when
    no prior state must survive. The caller must own the AME. Runtime datatype
    descriptors must agree with the static matrix or accumulator types;
    signedness, rounding, saturation and operation availability remain
    architectural runtime properties. Scalar operands and results may use i8,
    i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 attr-dict `:` functional-type(operands, results)";
}

def AME_MsinEwOp : AME_Op<"msin.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise sine";
  let description = [{
    Selects the ZTT v0.6 `msin.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MsllEwOp : AME_Op<"msll.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise logical left shift";
  let description = [{
    Selects the ZTT v0.6 `msll.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MsllEwXOp : AME_Op<"msll.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise logical left shift with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `msll.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MsqrtEwOp : AME_Op<"msqrt.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise square root";
  let description = [{
    Selects the ZTT v0.6 `msqrt.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MsraEwOp : AME_Op<"msra.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise arithmetic right shift";
  let description = [{
    Selects the ZTT v0.6 `msra.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MsraEwXOp : AME_Op<"msra.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise arithmetic right shift with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `msra.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MsrlEwOp : AME_Op<"msrl.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise logical right shift";
  let description = [{
    Selects the ZTT v0.6 `msrl.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MsrlEwXOp : AME_Op<"msrl.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise logical right shift with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `msrl.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mss1rOp : AME_Op<"mss.1r"> {
  let summary = "Store an AME matrix using the one-register instruction";
  let description = [{
    Selects `mss.1r` and writes matrix data to the supplied pointer. The
    matrix is a source and the operation has no SSA results. This whole-
    register instruction transfers exactly one physical M register. The caller
    must own the AME. Runtime datatype descriptors must agree with the static
    matrix or accumulator types; signedness, rounding, saturation and
    operation availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$ms, LLVM_DefaultPointer:$base);
  let results = (outs);
  let assemblyFormat =
    "$ms `,` $base attr-dict `:` functional-type(operands, results)";
}

def AME_MssCmOp : AME_Op<"mss.cm"> {
  let summary = "Store an AME matrix using the column-major instruction";
  let description = [{
    Selects `mss.cm` and writes matrix data to the supplied pointer. The
    matrix is a source and the operation has no SSA results. The caller must
    own the AME. Runtime datatype descriptors must agree with the static
    matrix or accumulator types; signedness, rounding, saturation and
    operation availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$ms, LLVM_DefaultPointer:$base);
  let results = (outs);
  let assemblyFormat =
    "$ms `,` $base attr-dict `:` functional-type(operands, results)";
}

def AME_MssRmOp : AME_Op<"mss.rm"> {
  let summary = "Store an AME matrix using the row-major instruction";
  let description = [{
    Selects `mss.rm` and writes matrix data to the supplied pointer. The
    matrix is a source and the operation has no SSA results. The caller must
    own the AME. Runtime datatype descriptors must agree with the static
    matrix or accumulator types; signedness, rounding, saturation and
    operation availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$ms, LLVM_DefaultPointer:$base);
  let results = (outs);
  let assemblyFormat =
    "$ms `,` $base attr-dict `:` functional-type(operands, results)";
}

def AME_MssStOp : AME_Op<"mss.st"> {
  let summary = "Store an AME matrix using the strided instruction";
  let description = [{
    Selects `mss.st` and writes matrix data to the supplied pointer. The
    matrix is a source and the operation has no SSA results. The trailing
    integer operand supplies the instruction stride; the lowering forwards it
    without reshaping or synthesizing a loop. The caller must own the AME.
    Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties. Scalar operands and
    results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$ms, LLVM_DefaultPointer:$base, AME_Scalar:$rs1);
  let results = (outs);
  let assemblyFormat =
    "$ms `,` $base `,` $rs1 attr-dict `:` functional-type(operands, results)";
}

def AME_MssTstOp : AME_Op<"mss.tst"> {
  let summary = "Store an AME matrix using the transposed-strided instruction";
  let description = [{
    Selects `mss.tst` and writes matrix data to the supplied pointer. The
    matrix is a source and the operation has no SSA results. The trailing
    integer operand supplies the instruction stride; the lowering forwards it
    without reshaping or synthesizing a loop. The caller must own the AME.
    Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties. Scalar operands and
    results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$ms, LLVM_DefaultPointer:$base, AME_Scalar:$rs1);
  let results = (outs);
  let assemblyFormat =
    "$ms `,` $base `,` $rs1 attr-dict `:` functional-type(operands, results)";
}

def AME_MsubEwOp : AME_Op<"msub.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise subtraction";
  let description = [{
    Selects the ZTT v0.6 `msub.ew` instruction and returns the updated M
    matrix. The draft operand order is `ms2 - ms1`: the second explicit matrix
    source minus the first. The old destination is explicit so that
    read/modify/write behavior, its descriptor and unsupported-tuple no-write
    behavior survive lowering. The caller must own the AME. Runtime datatype
    descriptors must agree with the static matrix or accumulator types;
    signedness, rounding, saturation and operation availability remain
    architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MsubEwXOp : AME_Op<"msub.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise subtraction with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `msub.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_Msublog2EwOp : AME_Op<"msublog2.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise subtract-log2";
  let description = [{
    Selects the ZTT v0.6 `msublog2.ew` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_Msublog2EwXOp : AME_Op<"msublog2.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise subtract-log2 with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `msublog2.ew.x` instruction and returns the updated M
    matrix. Its numerical behavior is the behavior of that architectural
    opcode under the runtime datatype descriptors; the dialect does not
    replace it with a generic arithmetic formula. The scalar integer bits are
    interpreted using the scalar datatype state in `amestype`, which must be
    programmed separately from matrix descriptors. The old destination is
    explicit so that read/modify/write behavior, its descriptor and
    unsupported-tuple no-write behavior survive lowering. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties. Scalar operands and
    results may use i8, i16, i32 or i64 in IR. The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MtanhEwOp : AME_Op<"mtanh.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise hyperbolic tangent";
  let description = [{
    Selects the ZTT v0.6 `mtanh.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MunpackEwXOp : AME_Op<"munpack.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise unpacking with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `munpack.ew.x` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_MxorEwOp : AME_Op<"mxor.ew", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise bitwise XOR";
  let description = [{
    Selects the ZTT v0.6 `mxor.ew` instruction and returns the updated M
    matrix. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Matrix:$ms1, AME_Matrix:$ms2);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $ms1 `,` $ms2 attr-dict `:` functional-type(operands, results)";
}

def AME_MxorEwXOp : AME_Op<"mxor.ew.x", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "AME element-wise bitwise XOR with a scalar integer";
  let description = [{
    Selects the ZTT v0.6 `mxor.ew.x` instruction and returns the updated M
    matrix. The scalar integer bits are interpreted using the scalar datatype
    state in `amestype`, which must be programmed separately from matrix
    descriptors. The old destination is explicit so that read/modify/write
    behavior, its descriptor and unsupported-tuple no-write behavior survive
    lowering. The caller must own the AME. Runtime datatype descriptors must
    agree with the static matrix or accumulator types; signedness, rounding,
    saturation and operation availability remain architectural runtime
    properties. Scalar operands and results may use i8, i16, i32 or i64 in IR.
    The backend normalizes them to one XLEN GPR: narrow inputs are
    zero-extended and wider inputs are truncated. Narrow results are truncated;
    results wider than XLEN are zero-extended. Explicitly sign-extend an input
    to XLEN when a signed interpretation of its upper bits is required.
  }];
  let arguments = (ins AME_Matrix:$md_in, AME_Scalar:$rs1, AME_Matrix:$ms1);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in `,` $rs1 `,` $ms1 attr-dict `:` functional-type(operands, results)";
}

def AME_Mzero2dAccOp : AME_Op<"mzero.2d.acc", [AllTypesMatch<["acc_in", "res"]>]> {
  let summary = "Zero an AME ACC accumulator square";
  let description = [{
    Selects `mzero.2d.acc` and returns the updated destination with its
    runtime descriptor dependency preserved. The operation clears the
    architectural base square; a larger SSA register-group type does not cause
    additional instructions or imply that every square is cleared. The caller
    must own the AME. Runtime datatype descriptors must agree with the static
    matrix or accumulator types; signedness, rounding, saturation and
    operation availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Accumulator:$acc_in);
  let results = (outs AME_Accumulator:$res);
  let assemblyFormat =
    "$acc_in attr-dict `:` functional-type(operands, results)";
}

def AME_Mzero2dMOp : AME_Op<"mzero.2d.m", [AllTypesMatch<["md_in", "res"]>]> {
  let summary = "Zero an AME M matrix square";
  let description = [{
    Selects `mzero.2d.m` and returns the updated destination with its runtime
    descriptor dependency preserved. The operation clears the architectural
    base square; a larger SSA register-group type does not cause additional
    instructions or imply that every square is cleared. The caller must own
    the AME. Runtime datatype descriptors must agree with the static matrix or
    accumulator types; signedness, rounding, saturation and operation
    availability remain architectural runtime properties.
  }];
  let arguments = (ins AME_Matrix:$md_in);
  let results = (outs AME_Matrix:$res);
  let assemblyFormat =
    "$md_in attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// AME Intrinsic Operations
//
// One operation per intrinsic in IntrinsicsRISCVXAME.td. Keep overload lists
// in intrinsic result/operand order. Multi-result intrinsics use one LLVM
// struct result, as required by LLVM_IntrOpBase.
//===----------------------------------------------------------------------===//

class AME_IntrOpBase<string intrinsicName,
                     list<int> overloadedResults = [],
                     list<int> overloadedOperands = [],
                     int numResults = 0> :
  LLVM_IntrOpBase</*Dialect dialect=*/AME_Dialect,
                  /*string opName=*/"intr." # !subst("_", ".", intrinsicName),
                  /*string enumName=*/"riscv_ztt_" # intrinsicName,
                  /*list<int> overloadedResults=*/overloadedResults,
                  /*list<int> overloadedOperands=*/overloadedOperands,
                  /*list<Trait> traits=*/[],
                  /*int numResults=*/numResults>;

def AME_AmeAcquire_IntrOp :
    AME_IntrOpBase<"ame_acquire", [0], [], 1>,
    Arguments<(ins LLVM_Type:$rs1)>;

def AME_AmeRelease_IntrOp :
    AME_IntrOpBase<"ame_release", [], [], 0>,
    Arguments<(ins)>;

def AME_Agettyp_IntrOp :
    AME_IntrOpBase<"agettyp", [0], [0], 1>,
    Arguments<(ins LLVM_Type:$acc)>;

def AME_Asettyp_IntrOp :
    AME_IntrOpBase<"asettyp", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$acc_in, LLVM_Type:$rs1)>;

def AME_MabsEw_IntrOp :
    AME_IntrOpBase<"mabs_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MabsdiffEw_IntrOp :
    AME_IntrOpBase<"mabsdiff_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MabsdiffEwX_IntrOp :
    AME_IntrOpBase<"mabsdiff_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MaddEw_IntrOp :
    AME_IntrOpBase<"madd_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MaddEwX_IntrOp :
    AME_IntrOpBase<"madd_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MandEw_IntrOp :
    AME_IntrOpBase<"mand_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MandEwX_IntrOp :
    AME_IntrOpBase<"mand_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MandnotEw_IntrOp :
    AME_IntrOpBase<"mandnot_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MandnotEwX_IntrOp :
    AME_IntrOpBase<"mandnot_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_McmovgeEw_IntrOp :
    AME_IntrOpBase<"mcmovge_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_McmovltEw_IntrOp :
    AME_IntrOpBase<"mcmovlt_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_McmpgeEw_IntrOp :
    AME_IntrOpBase<"mcmpge_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_McmpgeEwX_IntrOp :
    AME_IntrOpBase<"mcmpge_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_McmpltEw_IntrOp :
    AME_IntrOpBase<"mcmplt_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_McmpltEwX_IntrOp :
    AME_IntrOpBase<"mcmplt_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_McolbcastEwX_IntrOp :
    AME_IntrOpBase<"mcolbcast_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_McolgatherEw_IntrOp :
    AME_IntrOpBase<"mcolgather_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_McolidEw_IntrOp :
    AME_IntrOpBase<"mcolid_ew", [0], [], 1>,
    Arguments<(ins LLVM_Type:$md_in)>;

def AME_McolshiftEwX_IntrOp :
    AME_IntrOpBase<"mcolshift_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_McolunzipEw_IntrOp :
    AME_IntrOpBase<"mcolunzip_ew", [0], [], 2>,
    Arguments<(ins LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_McolzipEw_IntrOp :
    AME_IntrOpBase<"mcolzip_ew", [0], [], 2>,
    Arguments<(ins LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MconvEw_IntrOp :
    AME_IntrOpBase<"mconv_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MbcastMX_IntrOp :
    AME_IntrOpBase<"mbcast_m_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$rs2)>;

def AME_McosEw_IntrOp :
    AME_IntrOpBase<"mcos_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_Mexp2Ew_IntrOp :
    AME_IntrOpBase<"mexp2_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MfrintmEw_IntrOp :
    AME_IntrOpBase<"mfrintm_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MfrintnEw_IntrOp :
    AME_IntrOpBase<"mfrintn_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MfrintpEw_IntrOp :
    AME_IntrOpBase<"mfrintp_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MfrintzEw_IntrOp :
    AME_IntrOpBase<"mfrintz_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_Mgettyp_IntrOp :
    AME_IntrOpBase<"mgettyp", [0], [0], 1>,
    Arguments<(ins LLVM_Type:$ms1)>;

def AME_MhdiffEw_IntrOp :
    AME_IntrOpBase<"mhdiff_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MhdiffEwX_IntrOp :
    AME_IntrOpBase<"mhdiff_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MldexpEw_IntrOp :
    AME_IntrOpBase<"mldexp_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MldexpEwX_IntrOp :
    AME_IntrOpBase<"mldexp_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MldexpaccEw_IntrOp :
    AME_IntrOpBase<"mldexpacc_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MldexpaccEwX_IntrOp :
    AME_IntrOpBase<"mldexpacc_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_Mlog2Ew_IntrOp :
    AME_IntrOpBase<"mlog2_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_Mlog2subEw_IntrOp :
    AME_IntrOpBase<"mlog2sub_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_Mlog2subEwX_IntrOp :
    AME_IntrOpBase<"mlog2sub_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_Mls1r_IntrOp :
    AME_IntrOpBase<"mls_1r", [0], [], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$base)>;

def AME_MlsCm_IntrOp :
    AME_IntrOpBase<"mls_cm", [0], [], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$base)>;

def AME_MlsRm_IntrOp :
    AME_IntrOpBase<"mls_rm", [0], [], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$base)>;

def AME_MlsSt_IntrOp :
    AME_IntrOpBase<"mls_st", [0], [2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$base, LLVM_Type:$rs1)>;

def AME_MlsTst_IntrOp :
    AME_IntrOpBase<"mls_tst", [0], [2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$base, LLVM_Type:$rs1)>;

def AME_MmaxEw_IntrOp :
    AME_IntrOpBase<"mmax_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmaxEwX_IntrOp :
    AME_IntrOpBase<"mmax_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MmeanEw_IntrOp :
    AME_IntrOpBase<"mmean_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmeanEwX_IntrOp :
    AME_IntrOpBase<"mmean_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MminEw_IntrOp :
    AME_IntrOpBase<"mmin_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MminEwX_IntrOp :
    AME_IntrOpBase<"mmin_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MmovMA_IntrOp :
    AME_IntrOpBase<"mmov_m_a", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$acc)>;

def AME_MmovAM_IntrOp :
    AME_IntrOpBase<"mmov_a_m", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$acc_in, LLVM_Type:$ms1)>;

def AME_MmovMM_IntrOp :
    AME_IntrOpBase<"mmov_m_m", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_Mmove8MX_IntrOp :
    AME_IntrOpBase<"mmove8_m_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$rs2)>;

def AME_Mmove16MX_IntrOp :
    AME_IntrOpBase<"mmove16_m_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$rs2)>;

def AME_Mmove32MX_IntrOp :
    AME_IntrOpBase<"mmove32_m_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$rs2)>;

def AME_Mmove64MX_IntrOp :
    AME_IntrOpBase<"mmove64_m_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$rs2)>;

def AME_Mmove8XM_IntrOp :
    AME_IntrOpBase<"mmove8_x_m", [0], [0], 1>,
    Arguments<(ins LLVM_Type:$ms1, LLVM_Type:$rs1)>;

def AME_Mmove16XM_IntrOp :
    AME_IntrOpBase<"mmove16_x_m", [0], [0], 1>,
    Arguments<(ins LLVM_Type:$ms1, LLVM_Type:$rs1)>;

def AME_Mmove32XM_IntrOp :
    AME_IntrOpBase<"mmove32_x_m", [0], [0], 1>,
    Arguments<(ins LLVM_Type:$ms1, LLVM_Type:$rs1)>;

def AME_Mmove64XM_IntrOp :
    AME_IntrOpBase<"mmove64_x_m", [0], [0], 1>,
    Arguments<(ins LLVM_Type:$ms1, LLVM_Type:$rs1)>;

def AME_MmulEw_IntrOp :
    AME_IntrOpBase<"mmul_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmulEwX_IntrOp :
    AME_IntrOpBase<"mmul_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_Mmulacc2d_IntrOp :
    AME_IntrOpBase<"mmulacc_2d", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$acc_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmulaccEw_IntrOp :
    AME_IntrOpBase<"mmulacc_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmulaccEwX_IntrOp :
    AME_IntrOpBase<"mmulacc_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_Mmulaccneg2d_IntrOp :
    AME_IntrOpBase<"mmulaccneg_2d", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$acc_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmulaccnegEw_IntrOp :
    AME_IntrOpBase<"mmulaccneg_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmulaccnegEwX_IntrOp :
    AME_IntrOpBase<"mmulaccneg_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MmuladdEw_IntrOp :
    AME_IntrOpBase<"mmuladd_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmuladdEwX_IntrOp :
    AME_IntrOpBase<"mmuladd_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_Mmulatacc2d_IntrOp :
    AME_IntrOpBase<"mmulatacc_2d", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$acc_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_Mmulataccneg2d_IntrOp :
    AME_IntrOpBase<"mmulataccneg_2d", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$acc_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_Mmulbtacc2d_IntrOp :
    AME_IntrOpBase<"mmulbtacc_2d", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$acc_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_Mmulbtaccneg2d_IntrOp :
    AME_IntrOpBase<"mmulbtaccneg_2d", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$acc_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmulnegEw_IntrOp :
    AME_IntrOpBase<"mmulneg_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmulnegEwX_IntrOp :
    AME_IntrOpBase<"mmulneg_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MmulsubEw_IntrOp :
    AME_IntrOpBase<"mmulsub_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmulsubEwX_IntrOp :
    AME_IntrOpBase<"mmulsub_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MorEw_IntrOp :
    AME_IntrOpBase<"mor_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MorEwX_IntrOp :
    AME_IntrOpBase<"mor_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MornotEw_IntrOp :
    AME_IntrOpBase<"mornot_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MornotEwX_IntrOp :
    AME_IntrOpBase<"mornot_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MpackEwX_IntrOp :
    AME_IntrOpBase<"mpack_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MprefixaddCol_IntrOp :
    AME_IntrOpBase<"mprefixadd_col", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MprefixaddRow_IntrOp :
    AME_IntrOpBase<"mprefixadd_row", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MprefixmaxCol_IntrOp :
    AME_IntrOpBase<"mprefixmax_col", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MprefixmaxRow_IntrOp :
    AME_IntrOpBase<"mprefixmax_row", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MrdexpEw_IntrOp :
    AME_IntrOpBase<"mrdexp_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MrdexpaccEw_IntrOp :
    AME_IntrOpBase<"mrdexpacc_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MrecEw_IntrOp :
    AME_IntrOpBase<"mrec_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MreduceaddCol_IntrOp :
    AME_IntrOpBase<"mreduceadd_col", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MreduceaddRow_IntrOp :
    AME_IntrOpBase<"mreduceadd_row", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MreducemaxCol_IntrOp :
    AME_IntrOpBase<"mreducemax_col", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MreducemaxRow_IntrOp :
    AME_IntrOpBase<"mreducemax_row", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MreduceminCol_IntrOp :
    AME_IntrOpBase<"mreducemin_col", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MreduceminRow_IntrOp :
    AME_IntrOpBase<"mreducemin_row", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MrowbcastEwX_IntrOp :
    AME_IntrOpBase<"mrowbcast_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MrowgatherEw_IntrOp :
    AME_IntrOpBase<"mrowgather_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MrowidEw_IntrOp :
    AME_IntrOpBase<"mrowid_ew", [0], [], 1>,
    Arguments<(ins LLVM_Type:$md_in)>;

def AME_MrowshiftEwX_IntrOp :
    AME_IntrOpBase<"mrowshift_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MrowunzipEw_IntrOp :
    AME_IntrOpBase<"mrowunzip_ew", [0], [], 2>,
    Arguments<(ins LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MrowzipEw_IntrOp :
    AME_IntrOpBase<"mrowzip_ew", [0], [], 2>,
    Arguments<(ins LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MrsqrtEw_IntrOp :
    AME_IntrOpBase<"mrsqrt_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MrowscataddEw_IntrOp :
    AME_IntrOpBase<"mrowscatadd_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_McolscataddEw_IntrOp :
    AME_IntrOpBase<"mcolscatadd_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MrowscatmaxEw_IntrOp :
    AME_IntrOpBase<"mrowscatmax_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_McolscatmaxEw_IntrOp :
    AME_IntrOpBase<"mcolscatmax_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MselgeEw_IntrOp :
    AME_IntrOpBase<"mselge_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MselltEw_IntrOp :
    AME_IntrOpBase<"msellt_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_Msettyp_IntrOp :
    AME_IntrOpBase<"msettyp", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1)>;

def AME_MsinEw_IntrOp :
    AME_IntrOpBase<"msin_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MsllEw_IntrOp :
    AME_IntrOpBase<"msll_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MsllEwX_IntrOp :
    AME_IntrOpBase<"msll_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MsqrtEw_IntrOp :
    AME_IntrOpBase<"msqrt_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MsraEw_IntrOp :
    AME_IntrOpBase<"msra_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MsraEwX_IntrOp :
    AME_IntrOpBase<"msra_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MsrlEw_IntrOp :
    AME_IntrOpBase<"msrl_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MsrlEwX_IntrOp :
    AME_IntrOpBase<"msrl_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_Mss1r_IntrOp :
    AME_IntrOpBase<"mss_1r", [], [0], 0>,
    Arguments<(ins LLVM_Type:$ms, LLVM_Type:$base)>;

def AME_MssCm_IntrOp :
    AME_IntrOpBase<"mss_cm", [], [0], 0>,
    Arguments<(ins LLVM_Type:$ms, LLVM_Type:$base)>;

def AME_MssRm_IntrOp :
    AME_IntrOpBase<"mss_rm", [], [0], 0>,
    Arguments<(ins LLVM_Type:$ms, LLVM_Type:$base)>;

def AME_MssSt_IntrOp :
    AME_IntrOpBase<"mss_st", [], [0, 2], 0>,
    Arguments<(ins LLVM_Type:$ms, LLVM_Type:$base, LLVM_Type:$rs1)>;

def AME_MssTst_IntrOp :
    AME_IntrOpBase<"mss_tst", [], [0, 2], 0>,
    Arguments<(ins LLVM_Type:$ms, LLVM_Type:$base, LLVM_Type:$rs1)>;

def AME_MsubEw_IntrOp :
    AME_IntrOpBase<"msub_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MsubEwX_IntrOp :
    AME_IntrOpBase<"msub_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_Msublog2Ew_IntrOp :
    AME_IntrOpBase<"msublog2_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_Msublog2EwX_IntrOp :
    AME_IntrOpBase<"msublog2_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MtanhEw_IntrOp :
    AME_IntrOpBase<"mtanh_ew", [0], [1], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1)>;

def AME_MunpackEwX_IntrOp :
    AME_IntrOpBase<"munpack_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_MxorEw_IntrOp :
    AME_IntrOpBase<"mxor_ew", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MxorEwX_IntrOp :
    AME_IntrOpBase<"mxor_ew_x", [0], [1, 2], 1>,
    Arguments<(ins LLVM_Type:$md_in, LLVM_Type:$rs1, LLVM_Type:$ms1)>;

def AME_Mzero2dAcc_IntrOp :
    AME_IntrOpBase<"mzero_2d_acc", [0], [], 1>,
    Arguments<(ins LLVM_Type:$acc_in)>;

def AME_Mzero2dM_IntrOp :
    AME_IntrOpBase<"mzero_2d_m", [0], [], 1>,
    Arguments<(ins LLVM_Type:$md_in)>;

#endif // AME_DIALECT
