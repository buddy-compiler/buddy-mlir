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
#include <filesystem>
#include <iostream>
#include <string>

namespace mlir {
namespace buddy {
void registerConvVectorizationPass();
void registerPointwiseConvToGemmPass();
void registerPoolingVectorizationPass();
void registerLowerBudPass();
void registerLowerTracePass();
void registerLowerDIPPass();
void registerLowerDAPPass();
void registerExtendDAPPass();
void registerDAPVectorizePass();
void registerLowerRVVPass();
void registerBatchMatMulOptimizePass();
void registerMatMulOptimizePass();
void registerMatMulVectorizationPass();
void registerMatMulParallelVectorizationPass();
void registerTransposeOptimizationPass();
void registerConvOptimizePass();
void registerLowerVectorExpPass();
void registerLowerGemminiPass();
void registerLowerLinalgToGemminiPass();
void registerDeviceSchedulePass();
void registerLowerSchePass();
void registerFuncBufferizeDynamicOffsetPass();
} // namespace buddy
} // namespace mlir

int main(int argc, char **argv) {

  // Register all MLIR passes.
  mlir::registerAllPasses();
  mlir::buddy::registerPointwiseConvToGemmPass();
  // Register Vectorization of Convolution.
  mlir::buddy::registerConvVectorizationPass();
  // Register Vectorization of Pooling.
  mlir::buddy::registerPoolingVectorizationPass();
  mlir::buddy::registerLowerBudPass();
  mlir::buddy::registerLowerTracePass();
  mlir::buddy::registerLowerDIPPass();
  mlir::buddy::registerLowerDAPPass();
  mlir::buddy::registerExtendDAPPass();
  // Register Vectorization of DAP Dialect.
  mlir::buddy::registerDAPVectorizePass();
  mlir::buddy::registerLowerRVVPass();
  mlir::buddy::registerLowerVectorExpPass();
  mlir::buddy::registerLowerGemminiPass();
  mlir::buddy::registerLowerLinalgToGemminiPass();

  // Register Several Optimize Pass.
  mlir::buddy::registerMatMulOptimizePass();
  mlir::buddy::registerMatMulVectorizationPass();
  mlir::buddy::registerMatMulParallelVectorizationPass();
  mlir::buddy::registerBatchMatMulOptimizePass();
  mlir::buddy::registerTransposeOptimizationPass();
  mlir::buddy::registerConvOptimizePass();
  mlir::buddy::registerDeviceSchedulePass();
  mlir::buddy::registerLowerSchePass();
  mlir::buddy::registerFuncBufferizeDynamicOffsetPass();

  mlir::DialectRegistry registry;
  // Register all MLIR core dialects.
  registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  // Register dialects in buddy-mlir project.
  // clang-format off
  registry.insert<buddy::bud::BudDialect,
                  buddy::trace::TraceDialect,
                  buddy::dip::DIPDialect,
                  buddy::dap::DAPDialect,
                  buddy::rvv::RVVDialect,
                  buddy::vector_exp::VectorExpDialect,
                  buddy::gemmini::GemminiDialect,
                  buddy::sche::ScheDialect>();
  // clang-format on

  mlir::MLIRContext context(registry);

  // 获取输入文件和输出文件路径
  const char *inputFilePath = argv[1];

  // 使用 LLVM 的 SourceMgr 管理输入文件
  llvm::SourceMgr sourceMgr;
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilePath);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Error reading file: " << ec.message() << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  mlir::ParserConfig config(&context);

  // 解析文件中的 ModuleOp
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, config);

  if (!module) {
    llvm::errs() << "Failed to parse MLIR file!\n";
    return 1;
  }

  // pass

  mlir::OpBuilder builder(&context);
  auto moduleOp = module.get();
  // 在module的开头插入rtclock函数声明
  builder.setInsertionPointToStart(moduleOp.getBody());

  auto rtFuncType = builder.getFunctionType({}, {});

  // 声明 func.func private @timingStart()
  auto timingStartFunc = builder.create<mlir::func::FuncOp>(
      moduleOp.getLoc(), "timingStart", rtFuncType);
  timingStartFunc.setPrivate();

  // 声明 func.func private @timingEnd()
  auto timingEndFunc = builder.create<mlir::func::FuncOp>(
      moduleOp.getLoc(), "timingEnd", rtFuncType);
  timingEndFunc.setPrivate();

  std::vector<mlir::Operation *> ops;

  // 遍历
  moduleOp.walk([&](mlir::func::FuncOp funcOp) {
    // llvm::outs() << "Function name: " << funcOp.getName() << "\n";
    if (funcOp == timingStartFunc || funcOp == timingEndFunc) {
      // llvm::outs() << "return\n";
      return;
    }

    // 匹配所有xxx dialect 的op
    for (mlir::Operation &op : funcOp.getBody().front().getOperations()) {
      // llvm::outs << op->getName().getStringRef() << "\n";
      // 获取操作的方言名称
      mlir::StringRef dialect = op.getName().getDialectNamespace();

      // 检查操作是否是我们感兴趣的特定类型
      if (dialect == "linalg") {
        ops.push_back(&op);
      }
    }
  });

  if (ops.empty()) {
    std::cout << " No op " << std::endl;
    return 1;
  }

  // 插桩
  for (auto op : ops) {

    // 在op的前面添加call start函数
    builder.setInsertionPoint(op);
    builder.create<mlir::func::CallOp>(op->getLoc(), timingStartFunc,
                                       mlir::ValueRange{});

    // 在op的后面添加call end 函数
    builder.setInsertionPointAfter(op);
    builder.create<mlir::func::CallOp>(op->getLoc(), timingEndFunc,
                                       mlir::ValueRange{});
  }

  // 初始化runtime::profiler

  // 打印解析好的 ModuleOp
  moduleOp->print(llvm::outs());

  // 准备文件输出流
  std::filesystem::path filePath = inputFilePath;
  std::string fileName = filePath.filename();
  std::error_code errorCode;
  llvm::ToolOutputFile outputFile("./tmp.mlir", errorCode,
                                  llvm::sys::fs::OpenFlags(1)); // OF_Text

  // 检查是否能够成功打开输出文件
  if (errorCode) {
    llvm::errs() << "Error opening output file: " << errorCode.message()
                 << "\n";
    return 1;
  }

  moduleOp->print(outputFile.os());
  outputFile.keep();

  // 输出mlir文件，开子线程，编译成.so

  // load .so库进来

  // 输入数据预处理，并调用.so库

  // 生成json文件。

  return 0;
}