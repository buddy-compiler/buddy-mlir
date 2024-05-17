#ifndef MEMREFEXP_MEMREFEXPDIALECT_TD
#define MEMREFEXP_MEMREFEXPDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Memref Experiment Dialect Definition.
//===----------------------------------------------------------------------===//

def MemrefExp_Dialect : Dialect {
  let name = "memref_exp";
  let summary = "The Memref Experiment Dialect.";
  let cppNamespace = "::buddy::memref_exp";
}

//===----------------------------------------------------------------------===//
// Base Memref Experiment Operation Definition.
//===----------------------------------------------------------------------===//

class MemrefExp_Op<string mnemonic, list<Trait> traits = []> :
    Op<MemrefExp_Dialect, mnemonic, traits>;

#endif // MEMREFEXP_MEMREFEXPDIALECT_TD