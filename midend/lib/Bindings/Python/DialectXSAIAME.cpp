//===- DialectXSAIAME.cpp - Python bindings of the XSAIAME types ----------===//
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
// The `_buddyDialectsXSAIAME` extension of `buddy_mlir.dialects.xsai_ame`: the
// TileType and SyncType classes. TileType.get reports the C++ type verifier's
// diagnostics as ir.MLIRError, as the builtin checked types do.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

#include "buddy-mlir-c/Dialect/XSAIAME.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

using namespace nanobind::literals;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace xsaiame {

static nb::str toPyString(MlirStringRef ref) {
  return nb::str(ref.data, ref.length);
}

struct TileType : PyConcreteType<TileType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAXSAIAMETileType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirXSAIAMETileTypeGetTypeID;
  static constexpr const char *pyClassName = "TileType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &kind, std::vector<int64_t> shape,
           const std::string &element, DefaultingPyLocation loc) {
          PyMlirContext::ErrorCapture errors(loc->getContext());
          MlirType type = mlirXSAIAMETileTypeGetChecked(
              loc, mlirStringRefCreate(kind.data(), kind.size()), shape.size(),
              shape.data(),
              mlirStringRefCreate(element.data(), element.size()));
          if (mlirTypeIsNull(type))
            throw MLIRError("Invalid type", errors.take());
          return TileType(loc->getContext(), type);
        },
        "kind"_a, "shape"_a, "element"_a, nb::kw_only(), "loc"_a = nb::none(),
        "A tile type, e.g. !xsai_ame.tile<a, 128x32, fp16> for "
        "TileType.get(\"a\", [128, 32], \"fp16\"); dynamic dims are "
        "ShapedType.get_dynamic_size().");
    c.def_prop_ro("kind", [](const TileType &self) {
      return toPyString(mlirXSAIAMETileTypeGetKind(self));
    });
    c.def_prop_ro("shape", [](const TileType &self) {
      std::vector<int64_t> shape(mlirXSAIAMETileTypeGetRank(self));
      for (size_t dim = 0; dim < shape.size(); ++dim)
        shape[dim] = mlirXSAIAMETileTypeGetDimSize(self, dim);
      return shape;
    });
    c.def_prop_ro("element", [](const TileType &self) {
      return toPyString(mlirXSAIAMETileTypeGetElement(self));
    });
  }
};

struct SyncType : PyConcreteType<SyncType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAXSAIAMESyncType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirXSAIAMESyncTypeGetTypeID;
  static constexpr const char *pyClassName = "SyncType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return SyncType(context->getRef(),
                          mlirXSAIAMESyncTypeGet(context.get()->get()));
        },
        nb::kw_only(), "context"_a = nb::none(), "The !xsai_ame.sync type.");
  }
};

static void populateDialectXSAIAMESubmodule(nb::module_ &m) {
  TileType::bind(m);
  SyncType::bind(m);
}

} // namespace xsaiame
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_buddyDialectsXSAIAME, m) {
  m.doc() = "Buddy MLIR XSAIAME matrix tile dialect";

  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::xsaiame::
      populateDialectXSAIAMESubmodule(m);
}
