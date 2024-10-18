#include "Profiler.h"

namespace buddy {
namespace runtime {
std::unique_ptr<TimeManager> Profiler::timeManager = nullptr;

std::string Profiler::instrument(const std::string &targetDialect) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseMLIRsrcfile(mlirFilePath);

  // pass

  mlir::OpBuilder builder(&context);
  auto moduleOp = module.get();
  // 在module的开头插入rtclock函数声明
  builder.setInsertionPointToStart(moduleOp.getBody());

  auto i64Type = builder.getIntegerType(64);
  auto rtFuncType = builder.getFunctionType({i64Type}, {});

  // 声明 func.func private @timingStart()
  auto timingStartFunc = builder.create<mlir::func::FuncOp>(
      moduleOp.getLoc(), "timingStart", rtFuncType);
  timingStartFunc.setPrivate();

  // 声明 func.func private @timingEnd()
  auto timingEndFunc = builder.create<mlir::func::FuncOp>(
      moduleOp.getLoc(), "timingEnd", rtFuncType);
  timingEndFunc.setPrivate();

  TimeManager &timeManager = getTimeManager();

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
      mlir::StringRef opDialectName = op.getName().getDialectNamespace();

      // 检查操作是否是我们感兴趣的特定类型
      if (opDialectName == targetDialect) {

        // add TimeEvent and instrumentation
        TimeEvent event(&op);
        timeManager.addEvent(std::move(event));

        TimeEvent *e = timeManager.eventsBack();

        // 在op的前面添加call start函数
        builder.setInsertionPoint(&op);
        auto constantOp = builder.create<mlir::arith::ConstantOp>(
            op.getLoc(), i64Type, builder.getIntegerAttr(i64Type, uint64_t(e)));

        builder.create<mlir::func::CallOp>(op.getLoc(), timingStartFunc,
                                           mlir::ValueRange{constantOp});

        // 在op的后面添加call end 函数
        builder.setInsertionPointAfter(&op);
        builder.create<mlir::func::CallOp>(op.getLoc(), timingEndFunc,
                                           mlir::ValueRange{constantOp});
      }
    }
  });

  // 打印解析好的 ModuleOp
  // moduleOp->print(llvm::outs());

  // 准备文件输出流
  // std::filesystem::path filePath = inputFilePath;
  std::string fileName = mlirFilePath.stem();
  std::filesystem::path parent_directory = mlirFilePath.parent_path();
  std::filesystem::path new_file_path =
      parent_directory / (fileName + "_instrumented.mlir");
  std::error_code errorCode;
  llvm::ToolOutputFile outputFile(new_file_path.string(), errorCode,
                                  llvm::sys::fs::OpenFlags(1)); // OF_Text

  // 检查是否能够成功打开输出文件
  if (errorCode) {
    llvm::errs() << "Error opening output file: " << errorCode.message()
                 << "\n";
  }

  moduleOp->print(outputFile.os());
  outputFile.keep();
  return new_file_path.string();
}

void Profiler::compile(const std::string &target) {
  std::thread t(&buddy::runtime::Profiler::makeTarget, this, target);
  t.join();
}

void Profiler::loadLib(const std::string &lib) {}

void Profiler::makeTarget(const std::string &target) {
  std::string targetFullName = "profiling-" + target;
  std::string makeCommand = "make " + targetFullName;

  int result = std::system(makeCommand.c_str());

  if (result == 0) {
    std::cout << "Shared " + target + " library compiled successfully."
              << std::endl;
  } else {
    std::cerr << "Failed command: `" + makeCommand +
                     "` to compile shared library."
              << std::endl;
  }
}

mlir::OwningOpRef<mlir::ModuleOp>
Profiler::parseMLIRsrcfile(std::string mlirFilePath) {
  this->mlirFilePath = mlirFilePath;
  // 使用 LLVM 的 SourceMgr 管理输入文件
  llvm::SourceMgr sourceMgr;
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(mlirFilePath);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Error reading file: " << ec.message() << "\n";
    return nullptr;
  }

  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  mlir::ParserConfig config(&context);

  // Parse the ModuleOp in the file
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, config);

  if (!module) {
    llvm::errs() << "Failed to parse MLIR file!\n";
    return nullptr;
  }
  return module;
}

} // namespace runtime
} // namespace buddy