//===- BuddyIIR.mlir ---------------------------------------------------===//
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
// This file provides the MLIR IIR function.
//
//===----------------------------------------------------------------------===//

func.func @MLIR_iir(%in : memref<?xf32>, %filter : memref<?x?xf32>, %out : memref<?xf32>){
  %c0 = arith.constant 0 : index
  %N = memref.dim %in, %c0 : memref<?xf32>
  %M = memref.dim %filter, %c0: memref<?x?xf32>

  affine.for %j = 0 to %M iter_args(%inpt = %in) -> (memref<?xf32>){
    %b0 = affine.load %filter[%j, 0] : memref<?x?xf32>
    %b1 = affine.load %filter[%j, 1] : memref<?x?xf32>
    %b2 = affine.load %filter[%j, 2] : memref<?x?xf32>
    %a0 = affine.load %filter[%j, 3] : memref<?x?xf32>
    %a1 = affine.load %filter[%j, 4] : memref<?x?xf32>
    %a2 = affine.load %filter[%j, 5] : memref<?x?xf32>
    %init_z1 = arith.constant 0.0 : f32
    %init_z2 = arith.constant 0.0 : f32
    %res:2 = affine.for %i = 0 to %N iter_args(%z1 = %init_z1, %z2 = %init_z2) -> (f32, f32) {
        %input = affine.load %inpt[%i] : memref<?xf32>
        %t0 = arith.mulf %b0, %input : f32
        %output = arith.addf %t0, %z1 : f32

        %t1 = arith.mulf %b1, %input : f32
        %t2 = arith.mulf %a1, %output : f32
        %t3 = arith.subf %t1, %t2 : f32
        %z1_next = arith.addf %z2, %t3 : f32

        %t4 = arith.mulf %b2, %input : f32
        %t5 = arith.mulf %a2, %output : f32
        %z2_next = arith.subf %t4, %t5 : f32
        
        affine.store %output, %out[%i] : memref<?xf32>
        affine.yield %z1_next, %z2_next : f32, f32
    }
    affine.yield %out : memref<?xf32>
  }
  return
}

func.func @buddy_iir(%in : memref<?xf32>, %filter : memref<?x?xf32>, %out : memref<?xf32>) -> () {
  dap.iir %in, %filter, %out : memref<?xf32>, memref<?x?xf32>, memref<?xf32>
  return
} 
