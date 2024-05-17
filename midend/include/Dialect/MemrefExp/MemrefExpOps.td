#ifndef MEMREFEXP_MEMREFEXPOPS_TD
#define MEMREFEXP_MEMREFEXPOPS_TD

include "MemrefExpDialect.td"

include "mlir/IR/AttrTypeBase.td"

def MemrefExp_NullOp : MemrefExp_Op<"null"> {
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

#endif
