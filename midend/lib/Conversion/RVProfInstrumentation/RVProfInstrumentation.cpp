//====- RVProfInstrumentation.cpp - RVProf Instrumentation Pass -----------===//
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
// This file implements the RVProf instrumentation pass that automatically
// inserts profiling markers around linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"

#include "RVProf/RVProfDialect.h"
#include "RVProf/RVProfOps.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// RVProfInstrumentationPass
//===----------------------------------------------------------------------===//

namespace {
class RVProfInstrumentationPass
    : public PassWrapper<RVProfInstrumentationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RVProfInstrumentationPass)

  RVProfInstrumentationPass() = default;
  RVProfInstrumentationPass(const RVProfInstrumentationPass &) {}

  StringRef getArgument() const final { return "rvprof-instrument"; }
  StringRef getDescription() const final {
    return "Instrument MLIR code with RVProf profiling markers.";
  }

  Option<std::string> granularity{
      *this, "rvprof-granularity",
      llvm::cl::desc("Profiling granularity: linalg, gemmini, all"),
      llvm::cl::init("linalg")};

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<rvprof::RVProfDialect, linalg::LinalgDialect,
                    func::FuncDialect>();
  }

private:
  // Extract operation name from location attribute
  std::string extractNameFromLoc(Location loc);

  // Insert profiling markers around an operation
  void insertProfilingMarkers(Operation *op, StringRef name);

  // Counter for generating unique names
  int counter = 0;
};
} // end anonymous namespace

std::string RVProfInstrumentationPass::extractNameFromLoc(Location loc) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    return nameLoc.getName().str();
  }
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    return fileLoc.getFilename().str();
  }
  return "";
}

void RVProfInstrumentationPass::insertProfilingMarkers(Operation *op,
                                                       StringRef name) {
  OpBuilder builder(op);

  // Insert region_begin before the operation
  builder.create<rvprof::RegionBeginOp>(op->getLoc(),
                                        builder.getStringAttr(name));

  // Insert region_end after the operation
  builder.setInsertionPointAfter(op);
  builder.create<rvprof::RegionEndOp>(op->getLoc(),
                                      builder.getStringAttr(name));
}

void RVProfInstrumentationPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Instrument linalg operations
  if (granularity == "linalg" || granularity == "all") {
    module.walk([&](linalg::LinalgOp op) {
      // Try to extract name from location
      std::string name = extractNameFromLoc(op.getLoc());

      // If no name from location, generate one
      if (name.empty()) {
        name = op->getName().getStringRef().str() + "_" +
               std::to_string(counter++);
      }

      insertProfilingMarkers(op, name);
    });
  }

  // Note: gemmini instrumentation can be added here
  // when needed, following the same pattern
}

namespace mlir {
namespace buddy {
void registerRVProfInstrumentationPass() {
  PassRegistration<RVProfInstrumentationPass>();
}
} // namespace buddy
} // namespace mlir
