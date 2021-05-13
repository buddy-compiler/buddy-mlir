//===- ImgOps.td - Image Dialect Ops -----------------------*- tablegen -*-===//
//
// This is the top level file for operations in the image dialect.
//
//===----------------------------------------------------------------------===//

#ifndef IMG_IMGOPS_TD
#define IMG_IMGOPS_TD

include "ImgDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Img_PrewittOp : Img_Op<"prewitt", [NoSideEffect, MemRefsNormalizable]> {
  let summary = "Prewitt Operation.";
  let description = [{
    The `img.prewitt` operation is to perform edge detection.

    This operation takes an memref argument and returns the result memref.

    Example:

    ```mlir
    %0 = img.prewitt : memref<?x?xf32>
    ```
  }];

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

#endif // IMG_IMGOPS_TD
