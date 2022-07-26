//===- fir.mlir ------------------------------------------------------------===//
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
// This file provides the MLIR Fir (Conv 1d) function.
//
//===----------------------------------------------------------------------===//

func.func @conv1d_linalg(%in : memref<?xf32>, %filter : memref<?xf32>, %out : memref<?xf32>) -> () {
  linalg.conv_1d ins(%in, %filter : memref<?xf32>, memref<?xf32>)
                outs(%out : memref<?xf32>)
  return
}

// Produces the same result as the above function, but with the dap fir opeartion.
func.func @conv1d_buddy(%in : memref<?xf32>, %filter : memref<?xf32>, %out : memref<?xf32>) -> () {
  dap.fir %in, %filter, %out : memref<?xf32>, memref<?xf32>, memref<?xf32>
  return
}
