//===DialectVectorExp.cpp - Pybind module for VectorExp dialect API support===//
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

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "buddy-mlir-c/Dialects.h"
#include "buddy-mlir-c/Registration.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_vector_expDialects, m) {

  auto vector_expM = m.def_submodule("vector_exp");

  buddyMlirRegisterAllPasses();

  vector_expM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__vector_exp__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
