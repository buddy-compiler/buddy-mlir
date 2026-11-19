//===- tanh-approx.mlir - EmbeddingGemma GELU tanh approximation -----------===//
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %tanh = transform.structured.match ops{["math.tanh"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %parents = transform.get_parent_op %tanh <deduplicate>
      : (!transform.any_op) -> !transform.any_op
    transform.foreach %parents : !transform.any_op {
    ^bb0(%parent: !transform.any_op):
      %tiled, %forall = transform.structured.tile_using_forall %parent tile_sizes [1, 1, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.structured.vectorize %tiled vector_sizes [1, 1, 32] : !transform.any_op
      %parallel = transform.loop.forall_to_parallel %forall
        : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %parallel {
        transform.apply_patterns.math.polynomial_approximation
      } : !transform.any_op
      transform.yield
    }
    // Flatten contiguous binary elementwise loops before OpenMP collapsing:
    // otherwise B > 1 leaves expensive per-element coordinate reconstruction.
    %binary = transform.structured.match ops{["linalg.generic"]} attributes {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
    } in %root : (!transform.any_op) -> !transform.any_op
    %flat = transform.structured.flatten_elementwise %binary
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  // Run on tensors after scalar inlining, before tensor.extract becomes a
  // memref.load. Match multiplication only, keeping tanh's tiled path intact.
  transform.named_sequence @__flatten_scalar_muls(%root: !transform.any_op {transform.readonly}) {
    %unary = transform.structured.match ops{["linalg.generic"]} attributes {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
    } in %root : (!transform.any_op) -> !transform.any_op
    transform.foreach %unary : !transform.any_op {
    ^bb0(%op: !transform.any_op):
      %mul = transform.structured.match ops{["arith.mulf"]} in %op
        : (!transform.any_op) -> !transform.any_op
      %parent = transform.get_parent_op %mul <deduplicate>
        : (!transform.any_op) -> !transform.any_op
      %flat = transform.structured.flatten_elementwise %parent
        : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.yield
  }
}
