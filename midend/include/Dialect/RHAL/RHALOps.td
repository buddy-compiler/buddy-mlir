//===- RHALOps.td - RHAL Dialect Operations ----------------*- tablegen -*-===//
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

#ifndef BUDDY_MIDEND_RHAL_RHALOPS_TD
#define BUDDY_MIDEND_RHAL_RHALOPS_TD

include "RHAL/RHALDialect.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Base op class
//===----------------------------------------------------------------------===//

class RHAL_Op<string mnemonic, list<Trait> traits = []> :
    Op<RHAL_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// rhal.module — top-level model manifest container
//
// Syntax:
//   rhal.module @name attributes {version = "...", model_name = "...",
//                                  vocab_uri = "..."} {
//     ... rhal.constant, rhal.codeobj, rhal.buffer, rhal.func ops ...
//   }
//===----------------------------------------------------------------------===//

def RHAL_ModuleOp : RHAL_Op<"module", [
    AffineScope, IsolatedFromAbove, NoRegionArguments,
    SymbolTable, Symbol, NoTerminator, SingleBlock]> {
  let summary = "RHAL model manifest module.";
  let description = [{
    Top-level container op that declares all model resources and functions.
    Attributes carry module-level metadata (version, model_name, vocab_uri).
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$version,
    OptionalAttr<StrAttr>:$model_name,
    OptionalAttr<StrAttr>:$vocab_uri
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];
}

//===----------------------------------------------------------------------===//
// rhal.constant — external model weight declaration
//
// Syntax:
//   rhal.constant @name id = N : i32, storage = "external",
//                              type = tensor<...xf32>, uri = "file:..."
//===----------------------------------------------------------------------===//

def RHAL_ConstantOp : RHAL_Op<"constant", [Symbol]> {
  let summary = "External model constant (weight) declaration.";
  let description = [{
    Declares a model weight tensor stored outside the .rax file.
    - `storage`: "external" (file URI) or "inline" (embedded bytes)
    - `uri`:     "file:X" → X resolved relative to the .rax file
    - `type`:    shape and element type of the weight tensor
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I32Attr:$id,
    StrAttr:$storage,
    TypeAttr:$type,
    StrAttr:$uri
  );

  let assemblyFormat = [{
    $sym_name attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// rhal.codeobj — compiled kernel library declaration
//
// Syntax:
//   rhal.codeobj @name id = N : i32, kind = "host_shared_lib",
//                              backend = "cpu", uri = "file:..."
//===----------------------------------------------------------------------===//

def RHAL_CodeobjOp : RHAL_Op<"codeobj", [Symbol]> {
  let summary = "Compiled kernel code object declaration.";
  let description = [{
    Declares a compiled kernel library (e.g. a .so file) that provides
    the forward functions for this model.
    - `kind`:    "host_shared_lib" — dlopen-able shared library
    - `backend`: "cpu", "cuda", etc.
    - `uri`:     "file:X" → X resolved relative to the .rax file
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I32Attr:$id,
    StrAttr:$kind,
    StrAttr:$backend,
    StrAttr:$uri
  );

  let assemblyFormat = [{
    $sym_name attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// rhal.buffer — named buffer descriptor
//
// Syntax:
//   rhal.buffer @name space = "host"|"dram", type = tensor<...>
//===----------------------------------------------------------------------===//

def RHAL_BufferOp : RHAL_Op<"buffer", [Symbol]> {
  let summary = "Named buffer descriptor (input, output, or KV-cache state).";
  let description = [{
    Declares a named buffer used by the model functions.
    - `space`: "host" (CPU-visible), "dram" (device memory)
    - `type`:  shape and element type of the buffer
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$space,
    TypeAttr:$type
  );

  let assemblyFormat = [{
    $sym_name attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// rhal.func — function dispatch declaration
//
// Syntax:
//   rhal.func @name inputs = ["buf1", ...], outputs = ["buf2", ...],
//                   dispatch = "codeobj_name",
//                   args = ["buf1", "buf2", ...]
//===----------------------------------------------------------------------===//

def RHAL_FuncOp : RHAL_Op<"func", [Symbol]> {
  let summary = "Model function dispatch declaration.";
  let description = [{
    Declares a model entry point and its dispatch configuration:
    - `inputs`:   buffer names that are function inputs
    - `outputs`:  buffer names that are function outputs
    - `dispatch`: name of the code object that provides this function
    - `args`:     all buffer names passed as arguments to the kernel,
                  in ABI order (inputs + state + outputs)
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ArrayAttr:$inputs,
    ArrayAttr:$outputs,
    StrAttr:$dispatch,
    ArrayAttr:$args
  );

  let assemblyFormat = [{
    $sym_name attr-dict
  }];
}

#endif // BUDDY_MIDEND_RHAL_RHALOPS_TD
