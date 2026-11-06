//===- Endpoint.h - AME memory endpoint provenance -------------===//
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
// Classify where a memref SSA value ultimately lives for BOSCAME planning and
// verification: external DDR entry arguments, private allocations, the bounded
// private DDR frame, or SPM banks. Used by DMA / tile verifiers and by the
// planner when deciding legal transfers.
//
// Roots are found by walking view-like ops (subview, reinterpret_cast,
// memref.view, etc.) back to an alloc, argument, or SPM base. Classification
// never invents a root from element type alone.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_DIALECT_BOSCAME_ENDPOINT_H
#define BUDDY_DIALECT_BOSCAME_ENDPOINT_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>

namespace buddy::boscame {

// Storage class of a memref endpoint after view-chain resolution.
enum class EndpointKind {
  Unknown,           // Cannot classify; treat as illegal for AME DMA.
  ExternalDDR,       // Module entry / ABI-covered DDR argument.
  PrivateAllocation, // Function-local alloc (not the dedicated DDR frame).
  PrivateDDRFrame,   // Bounded, aligned private DDR staging window.
  SPM                // Fixed SPM bank memory space.
};

// Hardware / ABI limits for the private DDR frame used by experimental paths.
inline constexpr int64_t kPrivateDDRFrameAlignment = 64;
inline constexpr int64_t kPrivateDDRFrameMaxBytes = 262016;

// Walk view-like ops to the root memref that owns storage.
mlir::Value getEndpointRoot(mlir::Value value);

// Classify `value` after resolving its root. Returns Unknown when provenance
// is incomplete or conflicting.
EndpointKind classifyEndpoint(mlir::Value value);

// Static byte size of the root storage when fully static; std::nullopt if any
// dynamic dimension or unsupported type prevents a closed size.
std::optional<int64_t> getStaticStorageBytes(mlir::Value value);

// Emit a diagnostic on `diagnostic` and fail unless `value` is a legal private
// allocation for BOSCAME use (alignment, escape, size constraints).
mlir::LogicalResult verifyPrivateAllocation(mlir::Operation *diagnostic,
                                            mlir::Value value);

// Fail unless `value` is a legal private endpoint (allocation or DDR frame)
// for transfers that must not touch external entry buffers directly.
mlir::LogicalResult verifyPrivateEndpoint(mlir::Operation *diagnostic,
                                          mlir::Value value);

} // namespace buddy::boscame

#endif // BUDDY_DIALECT_BOSCAME_ENDPOINT_H
