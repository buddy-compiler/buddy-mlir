//===- ImgDialect.cpp - Image Dialect Definition-----------------*- C++ -*-===//
//
// This file defines image dialect.
//
//===----------------------------------------------------------------------===//

#include "Img/ImgDialect.h"
#include "Img/ImgOps.h"

using namespace mlir;
using namespace buddy::img;

//===----------------------------------------------------------------------===//
// Img dialect.
//===----------------------------------------------------------------------===//

void ImgDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Img/ImgOps.cpp.inc"
      >();
}
