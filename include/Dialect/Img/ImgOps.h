//===- ImgOps.h - Image Dialect Ops -----------------------------*- C++ -*-===//
//
// This is the header file for operations in image dialect. 
//
//===----------------------------------------------------------------------===//

#ifndef IMG_IMGOPS_H
#define IMG_IMGOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Img/ImgOps.h.inc"

#endif // IMG_IMGOPS_H
