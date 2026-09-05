//===- whisper_preprocess_lowering.mlir - Whisper preprocessing lowering -===//
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

// RUN: buddy-opt %s -extend-dap -one-shot-bufferize \
// RUN:   -convert-linalg-to-loops | FileCheck %s

func.func @whisper_preprocess(%input : memref<?xf64>)
    -> memref<1x80x3000xf32> {
  %result = dap.whisper_preprocess %input
      : memref<?xf64> to memref<1x80x3000xf32>
  return %result : memref<1x80x3000xf32>
}

// Bufferization evaluates tensor.pad regions across their full result tensors.
// Check that both reflection indices are guarded before they reach a load.

// CHECK-LABEL: func.func @whisper_preprocess
// CHECK: %[[LEFT_RESULT:.*]] = memref.alloc() {{.*}}memref<480200xf64>
// CHECK: scf.for %[[LEFT_IV:.*]] = {{.*}} to {{.*}} step {{.*}} {
// CHECK: %[[LEFT_REFLECTED:.*]] = arith.subi {{.*}}, %[[LEFT_IV]] : index
// CHECK: %[[IS_LEFT:.*]] = arith.cmpi slt, %[[LEFT_IV]], {{.*}} : index
// CHECK: %[[LEFT_SAFE:.*]] = arith.select %[[IS_LEFT]], %[[LEFT_REFLECTED]], {{.*}} : index
// CHECK: %[[LEFT_VALUE:.*]] = memref.load {{.*}}[%[[LEFT_SAFE]]] : memref<480000xf64>
// CHECK: memref.store %[[LEFT_VALUE]], %[[LEFT_RESULT]][%[[LEFT_IV]]] : memref<480200xf64>

// CHECK: %[[RIGHT_RESULT:.*]] = memref.alloc() {{.*}}memref<480400xf64>
// CHECK: scf.for %[[RIGHT_IV:.*]] = {{.*}} to {{.*}} step {{.*}} {
// CHECK: %[[RIGHT_DELTA:.*]] = arith.subi %[[RIGHT_IV]], {{.*}} : index
// CHECK: %[[RIGHT_REFLECTED:.*]] = arith.subi {{.*}}, %[[RIGHT_DELTA]] : index
// CHECK: %[[IS_RIGHT:.*]] = arith.cmpi sge, %[[RIGHT_IV]], {{.*}} : index
// CHECK: %[[RIGHT_SAFE:.*]] = arith.select %[[IS_RIGHT]], %[[RIGHT_REFLECTED]], {{.*}} : index
// CHECK: %[[RIGHT_VALUE:.*]] = memref.load {{.*}}[%[[RIGHT_SAFE]]] : memref<480200xf64>
// CHECK: memref.store %[[RIGHT_VALUE]], %[[RIGHT_RESULT]][%[[RIGHT_IV]]] : memref<480400xf64>
