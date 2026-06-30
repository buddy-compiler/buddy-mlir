//====--------- XTAME.td - XTAME dialect operation definitions ------------===//
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
// This file defines the XTAME dialect for RISC-V Matrix Extension.
//
//===----------------------------------------------------------------------===//

#ifndef XTAME_DIALECT
#define XTAME_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// XTAME Dialect Definition
//===----------------------------------------------------------------------===//

def XTAME_Dialect : Dialect {
  let name = "xt_ame";
  let cppNamespace = "::buddy::xtame";
  let description = [{
    The XTAME dialect provides operations for the RISC-V Matrix Extension
     (RVA23 profile).
    XTAME provides matrix multiply-accumulate instructions with
    tile-based computation,
    supporting various data types and widening modes for
     AI/ML workloads.

    Key features:
    - Tile-based matrix registers with configurable dimensions
    - Support for int4/int8/int16/int32/int64 integer types
    - Support for fp8/fp16/fp32/fp64 floating-point types
    - Widening matrix multiplication (2x, 4x, 8x output width)
    - Saturating arithmetic options
  }];
}

//===----------------------------------------------------------------------===//
// AME Operation Base Class
//===----------------------------------------------------------------------===//

class XTAME_Op<string mnemonic, list<Trait> traits = []> :
  Op<XTAME_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Matrix Configuration Operations
//===----------------------------------------------------------------------===//

def ThMcfgOp : XTAME_Op<"th.mcfg"> {
  let summary = "Set entire matrix size configuration";
  let description = [{
    Sets the matrix type configuration from rs1 register.
  }];
  let arguments = (ins I64:$mtype);
  let assemblyFormat = "$mtype attr-dict `:` type($mtype)";
}

def ThMcfgmOp : XTAME_Op<"th.mcfgm"> {
  let summary = "Set tile M size configuration";
  let description = [{
    Sets the tile M size (number of rows).
  }];
  let arguments = (ins I64:$tilem);
  let assemblyFormat = "$tilem attr-dict `:` type($tilem)";
}

def ThMcfgnOp : XTAME_Op<"th.mcfgn"> {
  let summary = "Set tile N size configuration";
  let description = [{
    Sets the tile N size (number of columns).
  }];
  let arguments = (ins I64:$tilen);
  let assemblyFormat = "$tilen attr-dict `:` type($tilen)";
}

def ThMcfgkOp : XTAME_Op<"th.mcfgk"> {
  let summary = "Set tile K size configuration";
  let description = [{
    Sets the tile K size (inner dimension for matrix multiply).
  }];
  let arguments = (ins I64:$tilek);
  let assemblyFormat = "$tilek attr-dict `:` type($tilek)";
}

//===----------------------------------------------------------------------===//
// Matrix Configuration Operations (with immediate)
//===----------------------------------------------------------------------===//

def ThMcfgmiOp : XTAME_Op<"th.mcfgmi"> {
  let summary = "Set tile M size with immediate";
  let description = [{
    Sets the tile M size (number of rows) from an immediate value.
    mtilem = min(tilem, TMMAX)
  }];
  let arguments = (ins I64Attr:$tilem);
  let assemblyFormat = "$tilem attr-dict";
}

def ThMcfgniOp : XTAME_Op<"th.mcfgni"> {
  let summary = "Set tile N size with immediate";
  let description = [{
    Sets the tile N size (number of columns) from an immediate value.
    mtilen = min(tilen, TNMAX)
  }];
  let arguments = (ins I64Attr:$tilen);
  let assemblyFormat = "$tilen attr-dict";
}

def ThMcfgkiOp : XTAME_Op<"th.mcfgki"> {
  let summary = "Set tile K size with immediate";
  let description = [{
    Sets the tile K size (inner dimension for matrix multiply) from
     an immediate value.
    mtilek = min(tilek, TKMAX)
  }];
  let arguments = (ins I64Attr:$tilek);
  let assemblyFormat = "$tilek attr-dict";
}

//===----------------------------------------------------------------------===//
// Matrix Zero Operations
//===----------------------------------------------------------------------===//

def ThMzeroOp : XTAME_Op<"th.mzero"> {
  let summary = "Zero one out matrix tile";
  let description = [{
    Sets all elements in the specified matrix tile to zero.
  }];
  let arguments = (ins I64Attr:$md);
  let assemblyFormat = "$md attr-dict";
}

def ThMzero2rOp : XTAME_Op<"th.mzero2r"> {
  let summary = "Zero two out matrix tile";
  let description = [{
    Sets all elements in the specified matrix tile to zero.
    This is a variant of mzero that can zero two tiles.
  }];
  let arguments = (ins I64Attr:$md);
  let assemblyFormat = "$md attr-dict";
}

def ThMzero4rOp : XTAME_Op<"th.mzero4r"> {
  let summary = "Zero four out matrix tile";
  let description = [{
    Sets all elements in the specified matrix tile to zero.
    This is a variant of mzero that can zero four tiles.
  }];
  let arguments = (ins I64Attr:$md);
  let assemblyFormat = "$md attr-dict";
}

def ThMzero8rOp : XTAME_Op<"th.mzero8r"> {
  let summary = "Zero eight out matrix tile";
  let description = [{
    Sets all elements in the specified matrix tile to zero.
    This is a variant of mzero that can zero eight tiles.
  }];
  let arguments = (ins I64Attr:$md);
  let assemblyFormat = "$md attr-dict";
}

//===----------------------------------------------------------------------===//
// Matrix MISC Instructions
//===----------------------------------------------------------------------===//

// Data Move Instructions between Matrix Registers
def ThMmovMmOp : XTAME_Op<"th.mmov.mm"> {
  let summary = "Move data between matrix tile registers";
  let description = [{
    Move the entire matrix data from source tile register ms1 to destination
    tile register md.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms1 attr-dict";
}

// Data Move Instructions between Integer and Matrix (Duplicate)
def ThMdupbMXOp : XTAME_Op<"th.mdupb.m.x"> {
  let summary = "Duplicate scalar to all elements in a matrix tile register";
  let description = [{
    Moves and Duplicates a scalar data to every element of the
    destination matrix register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$rs2
  );
  let assemblyFormat = "$md `,` $rs2 attr-dict";
}

def ThMduphMXOp : XTAME_Op<"th.mduph.m.x"> {
  let summary = "Duplicate scalar to all elements in a matrix tile register";
  let description = [{
    Moves and Duplicates a scalar data to every element of the
    destination matrix register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$rs2
  );
  let assemblyFormat = "$md `,` $rs2 attr-dict";
}

def ThMdupwMXOp : XTAME_Op<"th.mdupw.m.x"> {
  let summary = "Duplicate scalar to all elements in a matrix tile register";
  let description = [{
    Moves and Duplicates a scalar data to every element of the
    destination matrix register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$rs2
  );
  let assemblyFormat = "$md `,` $rs2 attr-dict";
}

def ThMdupdMXOp : XTAME_Op<"th.mdupd.m.x"> {
  let summary = "Duplicate scalar to all elements in a matrix tile register";
  let description = [{
    Moves and Duplicates a scalar data to every element of the
    destination matrix register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$rs2
  );
  let assemblyFormat = "$md `,` $rs2 attr-dict";
}

// Data Move Instructions between Integer and Matrix (Scalar to Matrix)
def ThMmovbMXOp : XTAME_Op<"th.mmovb.m.x"> {
  let summary = "Move from integer register to matrix tile register";
  let description = [{
    Moves a scalar data to an element of the destination matrix
    register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$rs2,
    I64:$rs1
  );
  let assemblyFormat = "$md `,` $rs2 `,` $rs1 attr-dict";
}

def ThMmovhMXOp : XTAME_Op<"th.mmovh.m.x"> {
  let summary = "Move from integer register to matrix tile register";
  let description = [{
    Moves a scalar data to an element of the destination matrix
    register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$rs2,
    I64:$rs1
  );
  let assemblyFormat = "$md `,` $rs2 `,` $rs1 attr-dict";
}

def ThMmovwMXOp : XTAME_Op<"th.mmovw.m.x"> {
  let summary = "Move from integer register to matrix tile register";
  let description = [{
    Moves a scalar data to an element of the destination matrix
    register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$rs2,
    I64:$rs1
  );
  let assemblyFormat = "$md `,` $rs2 `,` $rs1 attr-dict";
}

def ThMmovdMXOp : XTAME_Op<"th.mmovd.m.x"> {
  let summary = "Move from integer register to matrix tile register";
  let description = [{
    Moves a scalar data to an element of the destination matrix
    register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$rs2,
    I64:$rs1
  );
  let assemblyFormat = "$md `,` $rs2 `,` $rs1 attr-dict";
}

// Data Move Instructions between Integer and Matrix (Matrix to Scalar)
def ThMmovbXMOp : XTAME_Op<"th.mmovb.x.m"> {
  let summary = "Move from matrix tile register to integer register";
  let description = [{
    Moves a scalar data from a matrix register to a gerneral purpose
    register specified by rd.
  }];
  let arguments = (ins
    I64Attr:$ms2,
    I64:$rs1
  );
  let results = (outs I64:$result);
  let assemblyFormat = "$ms2 `,` $rs1 attr-dict `:` type($rs1) `->` type($result)";
}

def ThMmovhXMOp : XTAME_Op<"th.mmovh.x.m"> {
  let summary = "Move from matrix tile register to integer register";
  let description = [{
    Moves a scalar data from a matrix register to a gerneral purpose
    register specified by rd.
  }];
  let arguments = (ins
    I64Attr:$ms2,
    I64:$rs1
  );
  let results = (outs I64:$result);
  let assemblyFormat = "$ms2 `,` $rs1 attr-dict `:` type($rs1) `->` type($result)";
}

def ThMmovwXMOp : XTAME_Op<"th.mmovw.x.m"> {
  let summary = "Move from matrix tile register to integer register";
  let description = [{
    Moves a scalar data from a matrix register to a gerneral purpose
    register specified by rd.
  }];
  let arguments = (ins
    I64Attr:$ms2,
    I64:$rs1
  );
  let results = (outs I64:$result);
  let assemblyFormat = "$ms2 `,` $rs1 attr-dict `:` type($rs1) `->` type($result)";
}

def ThMmovdXMOp : XTAME_Op<"th.mmovd.x.m"> {
  let summary = "Move from matrix tile register to integer register";
  let description = [{
    Moves a scalar data from a matrix register to a gerneral purpose
    register specified by rd.
  }];
  let arguments = (ins
    I64Attr:$ms2,
    I64:$rs1
  );
  let results = (outs I64:$result);
  let assemblyFormat = "$ms2 `,` $rs1 attr-dict `:` type($rs1) `->` type($result)";
}

// Data Broadcast Instructions (row broadcast)
def ThMmovMvIOp : XTAME_Op<"th.mmov.mv.i"> {
  let summary = "matrix register row broadcast instructions";
  let description = [{
     Moves and duplicates a vector to every row of
     the destination matrix register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms1,
    I64Attr:$uimm3
  );
  let assemblyFormat = "$md `,` $ms1 `[` $uimm3 `]` attr-dict";
}

// Data Broadcast Instructions (column broadcast)
def ThMcmovbMvIOp : XTAME_Op<"th.mcmovb.mv.i"> {
  let summary = "matrix register column broadcast instructions";
  let description = [{
     Moves and duplicates a vector to every column of
     the destination matrix register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms1,
    I64Attr:$uimm3
  );
  let assemblyFormat = "$md `,` $ms1 `[` $uimm3 `]` attr-dict";
}

def ThMcmovhMvIOp : XTAME_Op<"th.mcmovh.mv.i"> {
  let summary = "matrix register column broadcast instructions";
  let description = [{
     Moves and duplicates a vector to every column of
     the destination matrix register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms1,
    I64Attr:$uimm3
  );
  let assemblyFormat = "$md `,` $ms1 `[` $uimm3 `]` attr-dict";
}

def ThMcmovwMvIOp : XTAME_Op<"th.mcmovw.mv.i"> {
  let summary = "matrix register column broadcast instructions";
  let description = [{
     Moves and duplicates a vector to every column of
     the destination matrix register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms1,
    I64Attr:$uimm3
  );
  let assemblyFormat = "$md `,` $ms1 `[` $uimm3 `]` attr-dict";
}

def ThMcmovdMvIOp : XTAME_Op<"th.mcmovd.mv.i"> {
  let summary = "matrix register column broadcast instructions";
  let description = [{
     Moves and duplicates a vector to every column of
     the destination matrix register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms1,
    I64Attr:$uimm3
  );
  let assemblyFormat = "$md `,` $ms1 `[` $uimm3 `]` attr-dict";
}

// Matrix Pack Instructions
def ThMpackMmOp : XTAME_Op<"th.mpack.mm"> {
  let summary = "Matrix tile pack operation";
  let description = [{
    Pack the low part of ms2 and low part of ms1 into md.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMpackhlMmOp : XTAME_Op<"th.mpackhl.mm"> {
  let summary = "Matrix tile pack operation";
  let description = [{
    Pack the high part of ms2 and low part of ms1 into md.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMpackhhMmOp : XTAME_Op<"th.mpackhh.mm"> {
  let summary = "Matrix tile pack operation";
  let description = [{
    Pack the high part of ms2 and high part of ms1 into md.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

//===----------------------------------------------------------------------===//
// Matrix Load Instructions
// Load matrix tiles from memory to tile registers
//===----------------------------------------------------------------------===//

def ThMlde8Op : XTAME_Op<"th.mlde8"> {
  let summary = "Load 8-bit matrix tile";
  let description = [{
    Load a matrix tile from memory to tile register.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMlde16Op : XTAME_Op<"th.mlde16"> {
  let summary = "Load 16-bit matrix tile";
  let description = [{
    Load a matrix tile from memory to tile register.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMlde32Op : XTAME_Op<"th.mlde32"> {
  let summary = "Load 32-bit matrix tile";
  let description = [{
    Load a matrix tile from memory to tile register.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMlde64Op : XTAME_Op<"th.mlde64"> {
  let summary = "Load 64-bit matrix tile";
  let description = [{
    Load a 64-bit matrix tile from memory to tile register.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Matrix transposed Load Instructions
// transposed load matrix tiles from memory to tile registers
//===----------------------------------------------------------------------===//

def ThMldte8Op : XTAME_Op<"th.mldte8"> {
  let summary = "Load and transpose 8-bit matrix tile";
  let description = [{
    Load a 8-bit matrix tile from memory to tile register with transposition.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMldte16Op : XTAME_Op<"th.mldte16"> {
  let summary = "Load and transpose 16-bit matrix tile";
  let description = [{
    Load a 16-bit matrix tile from memory to tile register with transposition.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMldte32Op : XTAME_Op<"th.mldte32"> {
  let summary = "Load and transpose 32-bit matrix tile";
  let description = [{
    Load a 32-bit matrix tile from memory to tile register with transposition.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMldte64Op : XTAME_Op<"th.mldte64"> {
  let summary = "Load and transpose 64-bit matrix tile";
  let description = [{
    Load a 64-bit matrix tile from memory to tile register with transposition.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Matrix stream Load Instructions
// stream load matrix tiles from memory to tile registers
//===----------------------------------------------------------------------===//

def ThMslde8Op : XTAME_Op<"th.mslde8"> {
  let summary = "Stream load 8-bit matrix tile";
  let description = [{
    Stream load a 8-bit matrix tile from memory to tile register.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMslde16Op : XTAME_Op<"th.mslde16"> {
  let summary = "Stream load 16-bit matrix tile";
  let description = [{
    Stream load a 16-bit matrix tile from memory to tile register.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMslde32Op : XTAME_Op<"th.mslde32"> {
  let summary = "Stream load 32-bit matrix tile";
  let description = [{
    Stream load a 32-bit matrix tile from memory to tile register.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMslde64Op : XTAME_Op<"th.mslde64"> {
  let summary = "Stream load 64-bit matrix tile";
  let description = [{
    Stream load a 64-bit matrix tile from memory to tile register.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Matrix transposed stream Load Instructions
// transposed stream load matrix tiles from memory to tile registers
//===----------------------------------------------------------------------===//

def ThMsldte8Op : XTAME_Op<"th.msldte8"> {
  let summary = "Stream load and transpose 8-bit matrix tile";
  let description = [{
    Stream load a 8-bit matrix tile from memory to tile register with transposition.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMsldte16Op : XTAME_Op<"th.msldte16"> {
  let summary = "Stream load and transpose 16-bit matrix tile";
  let description = [{
    Stream load a 16-bit matrix tile from memory to tile register with transposition.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMsldte32Op : XTAME_Op<"th.msldte32"> {
  let summary = "Stream load and transpose 32-bit matrix tile";
  let description = [{
    Stream load a 32-bit matrix tile from memory to tile register with transposition.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMsldte64Op : XTAME_Op<"th.msldte64"> {
  let summary = "Stream load and transpose 64-bit matrix tile";
  let description = [{
    Stream load a 64-bit matrix tile from memory to tile register with transposition.
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$md `,` $stride `,` $base attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Matrix Prefetch Load Instructions
// Prefetch load matrix tiles from memory to cache (not tile registers)
//===----------------------------------------------------------------------===//

def ThMplde8Op : XTAME_Op<"th.mplde8"> {
  let summary = "Prefetch 8-bit matrix tile";
  let description = [{
    Prefetch a 8-bit matrix tile from memory to cache.
    - stride: row byte stride
    - base: base address pointer
  }];
  let arguments = (ins
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$stride `,` $base attr-dict `:` type($base)";
}

def ThMplde16Op : XTAME_Op<"th.mplde16"> {
  let summary = "Prefetch 16-bit matrix tile";
  let description = [{
    Prefetch a 16-bit matrix tile from memory to cache.
    - stride: row byte stride
    - base: base address pointer
  }];
  let arguments = (ins
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$stride `,` $base attr-dict `:` type($base)";
}

def ThMplde32Op : XTAME_Op<"th.mplde32"> {
  let summary = "Prefetch 32-bit matrix tile";
  let description = [{
    Prefetch a 32-bit matrix tile from memory to cache.
    - stride: row byte stride
    - base: base address pointer
  }];
  let arguments = (ins
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$stride `,` $base attr-dict `:` type($base)";
}

def ThMplde64Op : XTAME_Op<"th.mplde64"> {
  let summary = "Prefetch 64-bit matrix tile";
  let description = [{
    Prefetch a 64-bit matrix tile from memory to cache.
    - stride: row byte stride
    - base: base address pointer
  }];
  let arguments = (ins
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$stride `,` $base attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Matrix Transposed Prefetch Load Instructions
// Transposed prefetch load matrix tiles from memory to cache (not tile registers)
//===----------------------------------------------------------------------===//

def ThMpldte8Op : XTAME_Op<"th.mpldte8"> {
  let summary = "Prefetch and transpose 8-bit matrix tile";
  let description = [{
    Prefetch a 8-bit matrix tile from memory to cache with transposition.
    - stride: row byte stride
    - base: base address pointer
  }];
  let arguments = (ins
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$stride `,` $base attr-dict `:` type($base)";
}

def ThMpldte16Op : XTAME_Op<"th.mpldte16"> {
  let summary = "Prefetch and transpose 16-bit matrix tile";
  let description = [{
    Prefetch a 16-bit matrix tile from memory to cache with transposition.
    - stride: row byte stride
    - base: base address pointer
  }];
  let arguments = (ins
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$stride `,` $base attr-dict `:` type($base)";
}

def ThMpldte32Op : XTAME_Op<"th.mpldte32"> {
  let summary = "Prefetch and transpose 32-bit matrix tile";
  let description = [{
    Prefetch a 32-bit matrix tile from memory to cache with transposition.
    - stride: row byte stride
    - base: base address pointer
  }];
  let arguments = (ins
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$stride `,` $base attr-dict `:` type($base)";
}

def ThMpldte64Op : XTAME_Op<"th.mpldte64"> {
  let summary = "Prefetch and transpose 64-bit matrix tile";
  let description = [{
    Prefetch a 64-bit matrix tile from memory to cache with transposition.
    - stride: row byte stride
    - base: base address pointer
  }];
  let arguments = (ins
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$stride `,` $base attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Matrix Store Instructions
// Store matrix tiles from tile registers to memory
//===----------------------------------------------------------------------===//

def ThMste8Op : XTAME_Op<"th.mste8"> {
  let summary = "Store 8-bit output matrix tile";
  let description = [{
    Store a 8-bit output matrix tile to memory.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMste16Op : XTAME_Op<"th.mste16"> {
  let summary = "Store 16-bit output matrix tile";
  let description = [{
    Store a 16-bit output matrix tile to memory.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMste32Op : XTAME_Op<"th.mste32"> {
  let summary = "Store 32-bit output matrix tile";
  let description = [{
    Store a 32-bit output matrix tile to memory.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMste64Op : XTAME_Op<"th.mste64"> {
  let summary = "Store 64-bit output matrix tile";
  let description = [{
    Store a 64-bit output matrix tile to memory.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Matrix transposed Store Instructions
// Transposed store matrix tiles from tile registers to memory
//===----------------------------------------------------------------------===//

def ThMstte8Op : XTAME_Op<"th.mstte8"> {
  let summary = "Store and transpose 8-bit output matrix tile";
  let description = [{
    Store a 8-bit output matrix tile to memory with transposition.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMstte16Op : XTAME_Op<"th.mstte16"> {
  let summary = "Store and transpose 16-bit output matrix tile";
  let description = [{
    Store a 16-bit output matrix tile to memory with transposition.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMstte32Op : XTAME_Op<"th.mstte32"> {
  let summary = "Store and transpose 32-bit output matrix tile";
  let description = [{
    Store a 32-bit output matrix tile to memory with transposition.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMstte64Op : XTAME_Op<"th.mstte64"> {
  let summary = "Store and transpose 64-bit output matrix tile";
  let description = [{
    Store a 64-bit output matrix tile to memory with transposition.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Matrix stream Store Instructions
// Stream store matrix tiles from tile registers to memory
//===----------------------------------------------------------------------===//

def ThMsste8Op : XTAME_Op<"th.msste8"> {
  let summary = "Stream store 8-bit output matrix tile";
  let description = [{
    Stream store a 8-bit output matrix tile to memory.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMsste16Op : XTAME_Op<"th.msste16"> {
  let summary = "Stream store 16-bit output matrix tile";
  let description = [{
    Stream store a 16-bit output matrix tile to memory.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMsste32Op : XTAME_Op<"th.msste32"> {
  let summary = "Stream store 32-bit output matrix tile";
  let description = [{
    Stream store a 32-bit output matrix tile to memory.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMsste64Op : XTAME_Op<"th.msste64"> {
  let summary = "Stream store 64-bit output matrix tile";
  let description = [{
    Stream store a 64-bit output matrix tile to memory.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Matrix transposed Stream Store Instructions
// Transposed stream store matrix tiles from tile registers to memory
//===----------------------------------------------------------------------===//

def ThMsstte8Op : XTAME_Op<"th.msstte8"> {
  let summary = "Stream store and transpose 8-bit output matrix tile";
  let description = [{
    Stream store a 8-bit output matrix tile to memory with transposition.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMsstte16Op : XTAME_Op<"th.msstte16"> {
  let summary = "Stream store and transpose 16-bit output matrix tile";
  let description = [{
    Stream store a 16-bit output matrix tile to memory with transposition.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMsstte32Op : XTAME_Op<"th.msstte32"> {
  let summary = "Stream store and transpose 32-bit output matrix tile";
  let description = [{
    Stream store a 32-bit output matrix tile to memory with transposition.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

def ThMsstte64Op : XTAME_Op<"th.msstte64"> {
  let summary = "Stream store and transpose 64-bit output matrix tile";
  let description = [{
    Stream store a 64-bit output matrix tile to memory with transposition.
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    I64:$stride,
    AnyMemRef:$base
  );
  let assemblyFormat = "$ms3 `,` $stride `,` $base attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Integer Matrix Multiplication Operations (Tile Register Version)
// These operate on tile registers, not memory
//===----------------------------------------------------------------------===//

def ThMmaccWBOp : XTAME_Op<"th.mmacc.w.b"> {
  let summary = "signed 8bit tile matrix multiply-accumulate";
  let description = [{
    Performs signed 8bit matrix multiply-accumulate on tile registers:
      md = md + ms1 × ms2
    output quad-widen

    - md: accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMmaccuWBOp : XTAME_Op<"th.mmaccu.w.b"> {
  let summary = "unsigned 8bit tile matrix multiply-accumulate";
  let description = [{
    Performs unsigned 8bit matrix multiply-accumulate on tile registers:
      md = md + ms1 × ms2
    output quad-widen

    - md: accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMmaccusWBOp : XTAME_Op<"th.mmaccus.w.b"> {
  let summary = "unsigned-signed 8bit tile matrix multiply-accumulate";
  let description = [{
    Performs unsigned-signed 8bit matrix multiply-accumulate on tile registers:
      md = md + ms1 × ms2
    output quad-widen
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMmaccsuWBOp : XTAME_Op<"th.mmaccsu.w.b"> {
  let summary = "signed-unsigned 8bit tile matrix multiply-accumulate";
  let description = [{
    Performs signed-unsigned 8bit matrix multiply-accumulate on tile registers:
      md = md + ms1 × ms2
    output quad-widen
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

//===----------------------------------------------------------------------===//
// Float point Matrix Multiplication Operations (Tile Register Version)
// These operate on tile registers, not memory
//===----------------------------------------------------------------------===//

// h: 16-bit float point (fp16) ; bf16: 16-bit float point(bf16);
// s: 32-bit float point; d: 64-bit float point
// e4m3 && e5m2: 8-bit float point with different configurations

def ThMfmaccHOp : XTAME_Op<"th.mfmacc.h"> {
  let summary = "16-bit float point matrix multiply-accumulate on tile registers";
  let description = [{
    Performs  16-bit float point matrix multiply-accumulate on tile registers:
      md = md + ms1 × ms2

    - md: accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccBf16Op : XTAME_Op<"th.mfmacc.bf16"> {
  let summary = "16-bit float point matrix multiply-accumulate on tile registers";
  let description = [{
    Performs 16-bit float point matrix multiply-accumulate on tile registers:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccSOp : XTAME_Op<"th.mfmacc.s"> {
  let summary = "32-bit float point matrix multiply-accumulate on tile registers";
  let description = [{
    Performs 32-bit float point matrix multiply-accumulate on tile registers:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccDOp : XTAME_Op<"th.mfmacc.d"> {
  let summary = "64-bit float point matrix multiply-accumulate on tile registers";
  let description = [{
    Performs 64-bit floating-point matrix multiply-accumulate on tile registers:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccHE4m3Op : XTAME_Op<"th.mfmacc.h.e4m3"> {
  let summary = "8-bit float point matrix multiply-accumulate with double-widen output";
  let description = [{
    Performs 8-bit float point matrix multiply-accumulate with double-widen output:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccHE5m2Op : XTAME_Op<"th.mfmacc.h.e5m2"> {
  let summary = "8-bit float point matrix multiply-accumulate with double-widen output";
  let description = [{
    Performs 8-bit float point matrix multiply-accumulate with double-widen output:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccBf16E4m3Op : XTAME_Op<"th.mfmacc.bf16.e4m3"> {
  let summary = "8bit Floating-point matrix multiply-accumulate with double-widen output";
  let description = [{
    Performs 8bit floating-point matrix multiply-accumulate with double-widen output:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccBf16E5m2Op : XTAME_Op<"th.mfmacc.bf16.e5m2"> {
  let summary = "8bit Floating-point matrix multiply-accumulate with double-widen output";
  let description = [{
    Performs 8bit floating-point matrix multiply-accumulate with double-widen output:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccSHOp : XTAME_Op<"th.mfmacc.s.h"> {
  let summary = "16-bit float point matrix multiply-accumulate with double-widen output";
  let description = [{
    Performs 16-bit float point matrix multiply-accumulate with double-widen output:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccSBf16Op : XTAME_Op<"th.mfmacc.s.bf16"> {
  let summary = "16-bit float point matrix multiply-accumulate with double-widen output";
  let description = [{
    Performs 16-bit float point matrix multiply-accumulate with double-widen output:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccDSOp : XTAME_Op<"th.mfmacc.d.s"> {
  let summary = "32-bit float point matrix multiply-accumulate with double-widen output";
  let description = [{
    Performs 32-bit float point matrix multiply-accumulate with double-widen output:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccSE4m3Op : XTAME_Op<"th.mfmacc.s.e4m3"> {
  let summary = "8-bit float point matrix multiply-accumulate with quad-widen output";
  let description = [{
    Performs 8-bit float point matrix multiply-accumulate with quad-widen output:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

def ThMfmaccSE5m2Op : XTAME_Op<"th.mfmacc.s.e5m2"> {
  let summary = "8-bit float point matrix multiply-accumulate with quad-widen output";
  let description = [{
    Performs 8-bit float point matrix multiply-accumulate with quad-widen output:
      md = md + ms1 × ms2

    - md: wide accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms2,
    I64Attr:$ms1
  );
  let assemblyFormat = "$md `,` $ms2 `,` $ms1 attr-dict";
}

//===----------------------------------------------------------------------===//
// AME Intrinsic Operation Definitions (for LLVM IR translation)
// These map directly to LLVM intrinsics defined in IntrinsicsRISCVExt.td
//===----------------------------------------------------------------------===//

// Base class for AME intrinsic ops
// Note: enumName must match the intrinsic name pattern: riscv_buddy_<name>
class XTAME_IntrOpBase<string mnemonic, int numResults = 0, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/XTAME_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/numResults>;

// Configuration intrinsics (with register)
def XTAME_ThMcfg_IntrOp : XTAME_IntrOpBase<"th.mcfg">,
  Arguments<(ins LLVM_Type:$mtype)>;
def XTAME_ThMcfgm_IntrOp : XTAME_IntrOpBase<"th.mcfgm">,
  Arguments<(ins LLVM_Type:$tilem)>;
def XTAME_ThMcfgn_IntrOp : XTAME_IntrOpBase<"th.mcfgn">,
  Arguments<(ins LLVM_Type:$tilen)>;
def XTAME_ThMcfgk_IntrOp : XTAME_IntrOpBase<"th.mcfgk">,
  Arguments<(ins LLVM_Type:$tilek)>;

// Configuration intrinsics (with immediate value)
def XTAME_ThMcfgmi_IntrOp : XTAME_IntrOpBase<"th.mcfgmi">,
  Arguments<(ins LLVM_Type:$tilem)>;
def XTAME_ThMcfgni_IntrOp : XTAME_IntrOpBase<"th.mcfgni">,
  Arguments<(ins LLVM_Type:$tilen)>;
def XTAME_ThMcfgki_IntrOp : XTAME_IntrOpBase<"th.mcfgki">,
  Arguments<(ins LLVM_Type:$tilek)>;

// Matrix zero intrinsic
def XTAME_ThMzero_IntrOp : XTAME_IntrOpBase<"th.mzero">,
  Arguments<(ins LLVM_Type:$md)>;
def XTAME_ThMzero2r_IntrOp : XTAME_IntrOpBase<"th.mzero2r">,
  Arguments<(ins LLVM_Type:$md)>;
def XTAME_ThMzero4r_IntrOp : XTAME_IntrOpBase<"th.mzero4r">,
  Arguments<(ins LLVM_Type:$md)>;
def XTAME_ThMzero8r_IntrOp : XTAME_IntrOpBase<"th.mzero8r">,
  Arguments<(ins LLVM_Type:$md)>;

// MISC Intrinsics (Data Move, Pack, Broadcast)
// Matrix to Matrix Move
def XTAME_ThMmovMm_IntrOp : XTAME_IntrOpBase<"th.mmov.mm">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms1)>;

// Duplicate Scalar to Matrix
def XTAME_ThMdupbMX_IntrOp : XTAME_IntrOpBase<"th.mdupb.m.x">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$rs2)>;
def XTAME_ThMduphMX_IntrOp : XTAME_IntrOpBase<"th.mduph.m.x">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$rs2)>;
def XTAME_ThMdupwMX_IntrOp : XTAME_IntrOpBase<"th.mdupw.m.x">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$rs2)>;
def XTAME_ThMdupdMX_IntrOp : XTAME_IntrOpBase<"th.mdupd.m.x">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$rs2)>;

// Scalar to Matrix Element
def XTAME_ThMmovbMX_IntrOp : XTAME_IntrOpBase<"th.mmovb.m.x">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$rs2, LLVM_Type:$rs1)>;
def XTAME_ThMmovhMX_IntrOp : XTAME_IntrOpBase<"th.mmovh.m.x">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$rs2, LLVM_Type:$rs1)>;
def XTAME_ThMmovwMX_IntrOp : XTAME_IntrOpBase<"th.mmovw.m.x">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$rs2, LLVM_Type:$rs1)>;
def XTAME_ThMmovdMX_IntrOp : XTAME_IntrOpBase<"th.mmovd.m.x">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$rs2, LLVM_Type:$rs1)>;

// Matrix to Scalar Element (Requires Return Value!)
def XTAME_ThMmovbXM_IntrOp : XTAME_IntrOpBase<"th.mmovb.x.m", 1>,
  Arguments<(ins LLVM_Type:$ms2, LLVM_Type:$rs1)>;
def XTAME_ThMmovhXM_IntrOp : XTAME_IntrOpBase<"th.mmovh.x.m", 1>,
  Arguments<(ins LLVM_Type:$ms2, LLVM_Type:$rs1)>;
def XTAME_ThMmovwXM_IntrOp : XTAME_IntrOpBase<"th.mmovw.x.m", 1>,
  Arguments<(ins LLVM_Type:$ms2, LLVM_Type:$rs1)>;
def XTAME_ThMmovdXM_IntrOp : XTAME_IntrOpBase<"th.mmovd.x.m", 1>,
  Arguments<(ins LLVM_Type:$ms2, LLVM_Type:$rs1)>;

// Broadcast Instructions (row broadcast)
def XTAME_ThMmovMvI_IntrOp : XTAME_IntrOpBase<"th.mmov.mv.i">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms1, LLVM_Type:$uimm3)>;

// Broadcast Instructions (column broadcast)
def XTAME_ThMcmovbMvI_IntrOp : XTAME_IntrOpBase<"th.mcmovb.mv.i">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms1, LLVM_Type:$uimm3)>;
def XTAME_ThMcmovhMvI_IntrOp : XTAME_IntrOpBase<"th.mcmovh.mv.i">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms1, LLVM_Type:$uimm3)>;
def XTAME_ThMcmovwMvI_IntrOp : XTAME_IntrOpBase<"th.mcmovw.mv.i">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms1, LLVM_Type:$uimm3)>;
def XTAME_ThMcmovdMvI_IntrOp : XTAME_IntrOpBase<"th.mcmovd.mv.i">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms1, LLVM_Type:$uimm3)>;

// Matrix Pack Instructions
def XTAME_ThMpackMm_IntrOp : XTAME_IntrOpBase<"th.mpack.mm">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMpackhlMm_IntrOp : XTAME_IntrOpBase<"th.mpackhl.mm">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMpackhhMm_IntrOp : XTAME_IntrOpBase<"th.mpackhh.mm">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;

// Matrix load intrinsics (to tile registers)
def XTAME_ThMlde8_IntrOp : XTAME_IntrOpBase<"th.mlde8">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMlde16_IntrOp : XTAME_IntrOpBase<"th.mlde16">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMlde32_IntrOp : XTAME_IntrOpBase<"th.mlde32">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMlde64_IntrOp : XTAME_IntrOpBase<"th.mlde64">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

// Matrix transpose load intrinsics (to tile registers)
def XTAME_ThMldte8_IntrOp : XTAME_IntrOpBase<"th.mldte8">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMldte16_IntrOp : XTAME_IntrOpBase<"th.mldte16">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMldte32_IntrOp : XTAME_IntrOpBase<"th.mldte32">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMldte64_IntrOp : XTAME_IntrOpBase<"th.mldte64">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

// Matrix stream load intrinsics (to tile registers)
def XTAME_ThMslde8_IntrOp : XTAME_IntrOpBase<"th.mslde8">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMslde16_IntrOp : XTAME_IntrOpBase<"th.mslde16">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMslde32_IntrOp : XTAME_IntrOpBase<"th.mslde32">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMslde64_IntrOp : XTAME_IntrOpBase<"th.mslde64">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

// Matrix transpose stream load intrinsics (to tile registers)
def XTAME_ThMsldte8_IntrOp : XTAME_IntrOpBase<"th.msldte8">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMsldte16_IntrOp : XTAME_IntrOpBase<"th.msldte16">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMsldte32_IntrOp : XTAME_IntrOpBase<"th.msldte32">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMsldte64_IntrOp : XTAME_IntrOpBase<"th.msldte64">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

// Prefetch Load Intrinsics
def XTAME_ThMplde8_IntrOp : XTAME_IntrOpBase<"th.mplde8">,
  Arguments<(ins LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMplde16_IntrOp : XTAME_IntrOpBase<"th.mplde16">,
  Arguments<(ins LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMplde32_IntrOp : XTAME_IntrOpBase<"th.mplde32">,
  Arguments<(ins LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMplde64_IntrOp : XTAME_IntrOpBase<"th.mplde64">,
  Arguments<(ins LLVM_Type:$stride, LLVM_Type:$base)>;

// Prefetch Transpose Load Intrinsics
def XTAME_ThMpldte8_IntrOp : XTAME_IntrOpBase<"th.mpldte8">,
  Arguments<(ins LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMpldte16_IntrOp : XTAME_IntrOpBase<"th.mpldte16">,
  Arguments<(ins LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMpldte32_IntrOp : XTAME_IntrOpBase<"th.mpldte32">,
  Arguments<(ins LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMpldte64_IntrOp : XTAME_IntrOpBase<"th.mpldte64">,
  Arguments<(ins LLVM_Type:$stride, LLVM_Type:$base)>;

// Matrix store intrinsics (from tile registers)
def XTAME_ThMste8_IntrOp : XTAME_IntrOpBase<"th.mste8">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMste16_IntrOp : XTAME_IntrOpBase<"th.mste16">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMste32_IntrOp : XTAME_IntrOpBase<"th.mste32">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMste64_IntrOp : XTAME_IntrOpBase<"th.mste64">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;

// Matrix transpose store intrinsics (from tile registers)
def XTAME_ThMstte8_IntrOp : XTAME_IntrOpBase<"th.mstte8">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMstte16_IntrOp : XTAME_IntrOpBase<"th.mstte16">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMstte32_IntrOp : XTAME_IntrOpBase<"th.mstte32">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMstte64_IntrOp : XTAME_IntrOpBase<"th.mstte64">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;

// Matrix stream store intrinsics (from tile registers)
def XTAME_ThMsste8_IntrOp : XTAME_IntrOpBase<"th.msste8">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMsste16_IntrOp : XTAME_IntrOpBase<"th.msste16">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMsste32_IntrOp : XTAME_IntrOpBase<"th.msste32">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMsste64_IntrOp : XTAME_IntrOpBase<"th.msste64">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;

// Matrix transpose stream store intrinsics (from tile registers)
def XTAME_ThMsstte8_IntrOp : XTAME_IntrOpBase<"th.msstte8">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMsstte16_IntrOp : XTAME_IntrOpBase<"th.msstte16">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMsstte32_IntrOp : XTAME_IntrOpBase<"th.msstte32">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;
def XTAME_ThMsstte64_IntrOp : XTAME_IntrOpBase<"th.msstte64">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;

// Tile register matrix multiplication intrinsics
def XTAME_ThMmaccWB_IntrOp : XTAME_IntrOpBase<"th.mmacc.w.b">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMmaccuWB_IntrOp : XTAME_IntrOpBase<"th.mmaccu.w.b">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMmaccusWB_IntrOp : XTAME_IntrOpBase<"th.mmaccus.w.b">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMmaccsuWB_IntrOp : XTAME_IntrOpBase<"th.mmaccsu.w.b">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;

// Tile register matrix multiply intrinsics for floating-point types
def XTAME_ThMfmaccH_IntrOp : XTAME_IntrOpBase<"th.mfmacc.h">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccBf16_IntrOp : XTAME_IntrOpBase<"th.mfmacc.bf16">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccS_IntrOp : XTAME_IntrOpBase<"th.mfmacc.s">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccD_IntrOp : XTAME_IntrOpBase<"th.mfmacc.d">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccHE4m3_IntrOp : XTAME_IntrOpBase<"th.mfmacc.h.e4m3">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccHE5m2_IntrOp : XTAME_IntrOpBase<"th.mfmacc.h.e5m2">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccBf16E4m3_IntrOp : XTAME_IntrOpBase<"th.mfmacc.bf16.e4m3">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccBf16E5m2_IntrOp : XTAME_IntrOpBase<"th.mfmacc.bf16.e5m2">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccSH_IntrOp : XTAME_IntrOpBase<"th.mfmacc.s.h">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccSBf16_IntrOp : XTAME_IntrOpBase<"th.mfmacc.s.bf16">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccDS_IntrOp : XTAME_IntrOpBase<"th.mfmacc.d.s">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccSE4m3_IntrOp : XTAME_IntrOpBase<"th.mfmacc.s.e4m3">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;
def XTAME_ThMfmaccSE5m2_IntrOp : XTAME_IntrOpBase<"th.mfmacc.s.e5m2">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms2, LLVM_Type:$ms1)>;

#endif // XTAME_DIALECT
