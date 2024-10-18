#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Bud/BudDialect.h"
#include "Bud/BudOps.h"
#include "DAP/DAPDialect.h"
#include "DAP/DAPOps.h"
#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "RVV/RVVDialect.h"
#include "Sche/ScheDialect.h"
#include "Sche/ScheOps.h"
#include "Trace/TraceDialect.h"
#include "Trace/TraceOps.h"
#include "VectorExp/VectorExpDialect.h"
#include "VectorExp/VectorExpOps.h"

#include "TimeManager.h"
#include <cstdint>
#include <dlfcn.h> // for dlopen, dlsym, dlclose
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace buddy {
namespace runtime {

class Profiler {

public:
  Profiler(std::filesystem::path mlirFilePath) : mlirFilePath(mlirFilePath) {
    /* init MLIRContext */
    mlir::DialectRegistry registry;
    // Register all MLIR core dialects.
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    // Register dialects in buddy-mlir project.
    // clang-format off
    registry.insert<bud::BudDialect,
                    trace::TraceDialect,
                    dip::DIPDialect,
                    dap::DAPDialect,
                    rvv::RVVDialect,
                    vector_exp::VectorExpDialect,
                    gemmini::GemminiDialect,
                    sche::ScheDialect>();
    // clang-format on
    context.appendDialectRegistry(registry);

    /* init TimeManager */
    if (!timeManager) {
      timeManager = std::make_unique<TimeManager>();
    }
  }

  ~Profiler() = default;

  std::string instrument(const std::string &targetDialect);

  void compile(const std::string &target);

  void loadLib(const std::string &);

  mlir::OwningOpRef<mlir::ModuleOp> parseMLIRsrcfile(std::string mlirFilePath);

  static TimeManager &getTimeManager() { return *timeManager; }

  void makeTarget(const std::string &target);

private:
  mlir::MLIRContext context;

  std::filesystem::path mlirFilePath;

  static std::unique_ptr<TimeManager> timeManager;
};
} // namespace runtime
} // namespace buddy
