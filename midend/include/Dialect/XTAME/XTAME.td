//====------ XTAME.td - XTAME dialect operation definitions --- tablegen ------===//
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
  let name = "xtame";
  let cppNamespace = "::buddy::xtame";
  let description = [{
    The XTAME dialect provides operations for the RISC-V Matrix Extension (RVA23 profile).
    XTAME provides matrix multiply-accumulate instructions with tile-based computation,
    supporting various data types and widening modes for AI/ML workloads.

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
    Returns the previous mtype value.
  }];
  let arguments = (ins I64:$mtype);
  let results = (outs I64:$result);
  let assemblyFormat = "$mtype attr-dict";
}

def ThMcfgmOp : XTAME_Op<"th.mcfgm"> {
  let summary = "Set tile M size configuration";
  let description = [{
    Sets the tile M size (number of rows).
    Returns the actual mtilem value set.
  }];
  let arguments = (ins I64:$tilem);
  let results = (outs I64:$result);
  let assemblyFormat = "$tilem attr-dict";
}

def ThMcfgnOp : XTAME_Op<"th.mcfgn"> {
  let summary = "Set tile N size configuration";
  let description = [{
    Sets the tile N size (number of columns).
    Returns the actual mtilen value set.
  }];
  let arguments = (ins I64:$tilen);
  let results = (outs I64:$result);
  let assemblyFormat = "$tilen attr-dict";
}

def ThMcfgkOp : XTAME_Op<"th.mcfgk"> {
  let summary = "Set tile K size configuration";
  let description = [{
    Sets the tile K size (inner dimension for matrix multiply).
    Returns the actual mtilek value set.
  }];
  let arguments = (ins I64:$tilek);
  let results = (outs I64:$result);
  let assemblyFormat = "$tilek attr-dict";
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
    Sets the tile K size (inner dimension for matrix multiply) from an immediate value.
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
    Load a 64-bit left matrix tile from memory to tile register.
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
  let summary = "signed-unsigned 8bit tile matrix multiply-accumulate with double-widen output";
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

// h: 16-bit float point (fp16) ; bf16: 16-bit float point(bf16); s: 32-bit float point; d: 64-bit float point
// e4m3 && e5m2: 8-bit float point with different exponent/mantissa configurations

def ThMfmaccHOp : XTAME_Op<"th.mfmacc.h"> {
  let summary = "16-bit float point(fp16) matrix multiply-accumulate on tile registers";
  let description = [{
    Performs  16-bit float point(fp16) matrix multiply-accumulate on tile registers:
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
  let summary = "16-bit float point(bf16) matrix multiply-accumulate on tile registers";
  let description = [{
    Performs 16-bit float point(bf16) matrix multiply-accumulate on tile registers:
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
  let summary = "8-bit float point matrix multiply-accumulate with double-widen on tile registers";
  let description = [{
    Performs 8-bit float point matrix multiply-accumulate with double-widen output on tile registers:
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
  let summary = "8-bit float point matrix multiply-accumulate with double-widen on tile registers";
  let description = [{
    Performs 8-bit float point matrix multiply-accumulate with double-widen output on tile registers:
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
  let summary = "8bit Floating-point matrix multiply-accumulate with double-widen on tile registers";
  let description = [{
    Performs 8bit floating-point matrix multiply-accumulate with double-widen output on tile registers:
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
  let summary = "8bit Floating-point matrix multiply-accumulate with double-widen on tile registers";
  let description = [{
    Performs 8bit floating-point matrix multiply-accumulate with double-widen output on tile registers:
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
  let summary = "16-bit float point matrix multiply-accumulate with double-widen on tile registers";
  let description = [{
    Performs 16-bit float point matrix multiply-accumulate with double-widen output on tile registers:
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
  let summary = "16-bit float point matrix multiply-accumulate with double-widen on tile registers";
  let description = [{
    Performs 16-bit float point matrix multiply-accumulate with double-widen output on tile registers:
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
  let summary = "32-bit float point matrix multiply-accumulate with double-widen on tile registers";
  let description = [{
    Performs 32-bit float point matrix multiply-accumulate with double-widen output on tile registers:
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
  let summary = "8-bit float point matrix multiply-accumulate with quad-widen on tile registers";
  let description = [{
    Performs 8-bit float point matrix multiply-accumulate with quad-widen output on tile registers:
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
  let summary = "8-bit float point matrix multiply-accumulate with quad-widen on tile registers";
  let description = [{
    Performs 8-bit float point matrix multiply-accumulate with quad-widen output on tile registers:
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
// These map directly to LLVM intrinsics defined in IntrinsicsRISCVBuddyExt.td
//===----------------------------------------------------------------------===//

// Base class for AME intrinsic ops
// Note: enumName must match the intrinsic name pattern: riscv_buddy_<name>
class XTAME_IntrOpBase<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/XTAME_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_buddy_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/0>;

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

// Matrix load intrinsics (to tile registers)
def XTAME_ThMlde8_IntrOp : XTAME_IntrOpBase<"th.mlde8">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

def XTAME_ThMlde16m_IntrOp : XTAME_IntrOpBase<"th.mlde16">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

def XTAME_ThMlde32m_IntrOp : XTAME_IntrOpBase<"th.mlde32">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

def XTAME_ThMlde64m_IntrOp : XTAME_IntrOpBase<"th.mlde64">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

// Matrix transpose load intrinsics (to tile registers)
def XTAME_ThMldte8_IntrOp : XTAME_IntrOpBase<"th.mldte8">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

def XTAME_ThMldte16m_IntrOp : XTAME_IntrOpBase<"th.mldte16">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

def XTAME_ThMldte32m_IntrOp : XTAME_IntrOpBase<"th.mldte32">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

def XTAME_ThMldte64m_IntrOp : XTAME_IntrOpBase<"th.mldte64">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$stride, LLVM_Type:$base)>;

// Matrix store intrinsics (from tile registers)
def XTAME_ThMste8_IntrOp : XTAME_IntrOpBase<"th.mste8">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;

def XTAME_ThMste16m_IntrOp : XTAME_IntrOpBase<"th.mste16">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;

def XTAME_ThMste32m_IntrOp : XTAME_IntrOpBase<"th.mste32">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$stride, LLVM_Type:$base)>;

def XTAME_ThMste64m_IntrOp : XTAME_IntrOpBase<"th.mste64">,
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
