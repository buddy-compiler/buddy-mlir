//===- ImgDialect.td - Image Dialect Definition ------------*- tablegen -*-===//
//
// This is the top level file for the Img dialect.
//
//===----------------------------------------------------------------------===//

#ifndef IMG_IMGDIALECT_TD
#define IMG_IMGDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Img dialect definition.
//===----------------------------------------------------------------------===//

def Img_Dialect : Dialect {
  let name = "img";
  let summary = "The Img dialect.";
  let description = [{
    The `img` dialect is intended to present image processing abstraction.
  }];
  let cppNamespace = "::buddy::img";
}

//===----------------------------------------------------------------------===//
// Base Img operation definition.
//===----------------------------------------------------------------------===//

class Img_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Img_Dialect, mnemonic, traits>;

#endif // IMG_IMGDIALECT_TD
