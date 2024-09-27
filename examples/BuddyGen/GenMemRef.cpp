//===- GenMemRef.cpp ------------------------------------------------------===//
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

// $ export LLVM_DIR=$PWD/../../llvm/
// $ export LLVM_BUILD_DIR=$LLVM_DIR/build
// $ c++ GenMemRef.cpp \
       -I $LLVM_DIR/llvm/include/ -I $LLVM_BUILD_DIR/include/ \
       -I $LLVM_DIR/mlir/include/ -I $LLVM_BUILD_DIR/tools/mlir/include/ \
       -L$LLVM_BUILD_DIR/lib -lMLIRIR -lMLIRParser -lMLIRSupport -lLLVMCore \
       -lLLVMSupport -lncurses -ltinfo -lstdc++ -lLLVMDemangle \
       -o a.out
// $ ./a.out

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

int main() {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::Type eleType = builder.getF64Type();
  // Target memref type:
  // `memref<?xf64, strided<[1], offset: ?>>`
  mlir::MemRefType memrefType = mlir::MemRefType::get(
      {mlir::ShapedType::kDynamic}, eleType,
      mlir::StridedLayoutAttr::get(
          &context, /*offset=*/mlir::ShapedType::kDynamic, /*strides=*/{1}));
  memrefType.dump();
  return 0;
}
