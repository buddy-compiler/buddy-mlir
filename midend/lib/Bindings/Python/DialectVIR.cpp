//===- DialectVIR.cpp - Python bindings of the VIR types and attributes ---===//
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
// The `_buddyDialectsVIR` extension of `buddy_mlir.dialects.vir`: the
// DynamicVectorType (a ShapedType, as VectorType) and ScalingFactorAttr
// classes, as the upstream dialect extensions (e.g. DialectQuant.cpp) bind
// theirs.
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <string>
#include <vector>

#include "buddy-mlir-c/Dialect/VIR.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/IRTypes.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

using namespace nanobind::literals;
using namespace mlir::python::nanobind_adaptors;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace vir {

struct ScalingFactorAttr : PyConcreteAttribute<ScalingFactorAttr> {
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsAVIRScalingFactorAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirVIRScalingFactorAttrGetTypeID;
  static constexpr const char *pyClassName = "ScalingFactorAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &ratio, int64_t value,
           DefaultingPyMlirContext context) {
          if (ratio != "m" && ratio != "f")
            throw nb::value_error("the ratio of a scaling factor is \"m\" "
                                  "(multiplier) or \"f\" (fraction)");
          if (value < 2 || value > 64 || (value & (value - 1)) != 0)
            throw nb::value_error(
                "the value of a scaling factor is 2, 4, 8, 16, 32 or 64");
          return ScalingFactorAttr(
              context->getRef(),
              mlirVIRScalingFactorAttrGet(
                  context.get()->get(),
                  mlirStringRefCreate(ratio.data(), ratio.size()), value));
        },
        "ratio"_a, "value"_a, nb::kw_only(), "context"_a = nb::none(),
        "A scaling factor, e.g. ScalingFactorAttr.get(\"m\", 2) for m2.");
    c.def_prop_ro("ratio", [](const ScalingFactorAttr &self) {
      MlirStringRef ref = mlirVIRScalingFactorAttrGetRatio(self);
      return nb::str(ref.data, ref.length);
    });
    c.def_prop_ro("value", [](const ScalingFactorAttr &self) {
      return mlirVIRScalingFactorAttrGetValue(self);
    });
  }
};

struct DynamicVectorType : PyConcreteType<DynamicVectorType, PyShapedType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAVIRDynamicVectorType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirVIRDynamicVectorTypeGetTypeID;
  static constexpr const char *pyClassName = "DynamicVectorType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, PyType &elementType,
           std::optional<PyAttribute> scalingFactor) {
          // as the parser: static dims, then the dynamic one
          int64_t dynamic = mlirShapedTypeGetDynamicSize();
          if (shape.empty() || shape.back() != dynamic)
            throw nb::value_error(
                "the last dim of a dynamic vector is the "
                "dynamic one (ShapedType.get_dynamic_size())");
          for (size_t i = 0; i + 1 < shape.size(); ++i)
            if (shape[i] == dynamic || shape[i] <= 0)
              throw nb::value_error("the dims before the last one of a "
                                    "dynamic vector are static and positive");
          if (!mlirTypeIsAInteger(elementType) &&
              !mlirTypeIsAIndex(elementType) && !mlirTypeIsAFloat(elementType))
            throw nb::value_error("the element type of a dynamic vector is an "
                                  "integer, index or float type");
          MlirAttribute factor = mlirAttributeGetNull();
          if (scalingFactor) {
            if (!mlirAttributeIsAVIRScalingFactorAttr(*scalingFactor))
              throw nb::value_error("expected a ScalingFactorAttr");
            factor = *scalingFactor;
          }
          return DynamicVectorType(
              elementType.getContext(),
              mlirVIRDynamicVectorTypeGet(shape.size(), shape.data(),
                                          elementType, factor));
        },
        "shape"_a, "element_type"_a, nb::kw_only(),
        "scaling_factor"_a = nb::none(),
        "A dynamic vector type, e.g. !vir.vec<2x?xf32> for "
        "DynamicVectorType.get([2, ShapedType.get_dynamic_size()], "
        "F32Type.get()).");
    c.def_prop_ro(
        "scaling_factor",
        [](const DynamicVectorType &self) -> std::optional<MlirAttribute> {
          MlirAttribute factor = mlirVIRDynamicVectorTypeGetScalingFactor(self);
          if (mlirAttributeIsNull(factor))
            return std::nullopt;
          return factor;
        },
        "The scaling factor (a ScalingFactorAttr), or None.");
  }
};

static void populateDialectVIRSubmodule(nb::module_ &m) {
  ScalingFactorAttr::bind(m);
  DynamicVectorType::bind(m);
}

} // namespace vir
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_buddyDialectsVIR, m) {
  m.doc() = "Buddy MLIR Dynamic Vector IR (VIR) dialect";

  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::vir::populateDialectVIRSubmodule(
      m);
}
