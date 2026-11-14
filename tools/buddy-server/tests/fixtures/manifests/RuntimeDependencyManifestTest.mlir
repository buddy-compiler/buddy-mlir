//===- RuntimeDependencyManifestTest.mlir - Dependency test manifest -----===//
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

rhal.module @runtime_dependency_test attributes {
    version = "0.1.0",
    model_name = "runtime_dependency_test",
    model_library = "file:service.so"} {
  rhal.codeobj @model {id = 1 : i32, kind = "host_shared_lib",
                       backend = "cpu", uri = "file:model.so"}
  rhal.codeobj @dependency {id = 2 : i32, kind = "host_shared_lib",
                            backend = "cpu", uri = "file:dependency.so"}
}
