//===- DAP.mlir -----------------------------------------------------------===//
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
// This file provides DAP dialect functions.
//
//===----------------------------------------------------------------------===//

func.func @buddy_fir(%in : memref<?xf32>, %filter : memref<?xf32>, %out : memref<?xf32>) -> () {
  dap.fir %in, %filter, %out : memref<?xf32>, memref<?xf32>, memref<?xf32>
  return
}

func.func @buddy_iir(%in : memref<?xf32>, %filter : memref<?x?xf32>, %out : memref<?xf32>) -> () {
  dap.iir %in, %filter, %out : memref<?xf32>, memref<?x?xf32>, memref<?xf32>
  return
} 

func.func @buddy_biquad(%in : memref<?xf32>, %filter : memref<?xf32>, %out : memref<?xf32>) -> () {
  dap.biquad %in, %filter, %out : memref<?xf32>, memref<?xf32>, memref<?xf32>
  return
}
