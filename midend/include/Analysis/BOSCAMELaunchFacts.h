//===- BOSCAMELaunchFacts.h - BOSCAME launch metadata facts ---------------===//
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
// Read-only snapshot of BOSCAME / GPU launch metadata already present on
// func.func and its arguments. This analysis never mutates IR and never
// invents roles from argument position or type.
//
// Recognized attributes (all optional unless noted by the verifier):
//   * argument `bosc_ame.program_role` = "num_programs" | "program_id"
//   * argument `bosc_ame.program_axis` = i64 in [0, 2] for num_programs
//   * function  `gpu.known_grid_size`  = dense<[Gx,Gy,Gz]> of positive i32
//
// knownGrid is a caller precondition used by planning / specialization. It
// does not prove that runtime program_id values lie inside that grid, and it
// does not authorize sharing private buffers across programs.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_ANALYSIS_BOSCAMELAUNCHFACTS_H
#define BUDDY_ANALYSIS_BOSCAMELAUNCHFACTS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace buddy::boscame {

// Role carried by an annotated program-grid argument.
enum class LaunchProgramRole { NumPrograms, ProgramId };

// One function argument that the frontend explicitly tagged as a launch
// parameter. `axis` is set only when `bosc_ame.program_axis` is present; a
// missing axis stays unknown and is never inferred from `argumentIndex`.
struct LaunchProgramArgumentFact {
  LaunchProgramRole role = LaunchProgramRole::NumPrograms;
  unsigned argumentIndex = 0;
  std::optional<int64_t> axis;
};

// Per-function launch fact detached from SSA / attribute handles so callers
// can keep the snapshot after the IR is rewritten. Empty `programArguments`
// means the function has no annotated program roles (not an error).
struct LaunchFunctionFact {
  std::string functionName;
  std::optional<std::array<int32_t, 3>> knownGrid;
  llvm::SmallVector<LaunchProgramArgumentFact, 6> programArguments;
};

// Walk every `func.func` in `module` and collect launch facts.
// Failure cases (diagnostics on the offending function):
//   * num_programs axis outside [0,2] or duplicated among num_programs args
//   * gpu.known_grid_size missing three positive i32 extents
// Unknown roles and absent axes are preserved, not rejected.
mlir::FailureOr<llvm::SmallVector<LaunchFunctionFact, 4>>
analyzeBOSCAMELaunchFacts(mlir::ModuleOp module);

} // namespace buddy::boscame

#endif // BUDDY_ANALYSIS_BOSCAMELAUNCHFACTS_H
