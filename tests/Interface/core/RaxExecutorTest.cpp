// RUN: buddy-rax-executor-test %t.rax

#include "buddy/runtime/core/RaxExecutor.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/rax/RAX.h"

#include "flatbuffers/flatbuffers.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rhal::rax;

static void writeTestRax(const std::string &path) {
  flatbuffers::FlatBufferBuilder builder;
  const std::string uri = "file:" RAX_EXECUTOR_TEST_LIBRARY;

  auto codeObjectA = CreateCodeObject(
      builder, 10, builder.CreateString("stage_a"),
      CodeObjectKind_HostSharedLib, 0, builder.CreateString(uri),
      builder.CreateString("kernel_a"), builder.CreateString("cpu"), 0);
  auto codeObjectB = CreateCodeObject(
      builder, 11, builder.CreateString("stage_b"),
      CodeObjectKind_HostSharedLib, 0, builder.CreateString(uri),
      builder.CreateString("kernel_b"), builder.CreateString("cpu"), 0);

  std::vector<flatbuffers::Offset<Arg>> firstArgs = {
      CreateArg(builder, 1, 0, 0, 0), CreateArg(builder, 0, 7, 0, 0)};
  auto firstDispatch =
      CreateDispatchOp(builder, 10, builder.CreateVector(firstArgs), 0, 0);
  std::vector<flatbuffers::Offset<Arg>> secondArgs = {
      CreateArg(builder, 0, 8, 0, 0), CreateArg(builder, 1, 0, 0, 0)};
  auto secondDispatch =
      CreateDispatchOp(builder, 11, builder.CreateVector(secondArgs), 0, 0);

  std::vector<flatbuffers::Offset<Op>> ops = {
      CreateOp(builder, OpKind_Dispatch, firstDispatch, 0, 0, 0, 0, 0),
      CreateOp(builder, OpKind_Barrier, 0, 0, 0, CreateBarrierOp(builder, 0), 0,
               0),
      CreateOp(builder, OpKind_Dispatch, secondDispatch, 0, 0, 0, 0, 0)};
  std::vector<uint32_t> inputs = {1};
  std::vector<uint32_t> outputs = {1};
  std::vector<uint32_t> temps = {2};
  auto function = CreateFunction(
      builder, builder.CreateString("pipeline"), builder.CreateVector(inputs),
      builder.CreateVector(outputs), builder.CreateVector(temps),
      builder.CreateVector(ops), 0);

  std::vector<flatbuffers::Offset<CodeObject>> codeObjects = {codeObjectA,
                                                              codeObjectB};
  std::vector<flatbuffers::Offset<Function>> functions = {function};
  auto module = CreateModule(builder, builder.CreateString("RAX"),
                             CreateVersion(builder, 0, 1, 0), Endianness_Little,
                             0, 0, 0, 0, builder.CreateVector(codeObjects),
                             builder.CreateVector(functions));
  builder.Finish(module, "RAX0");

  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(builder.GetBufferPointer()),
               builder.GetSize());
  if (!output)
    throw std::runtime_error("failed to write test RAX file");
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: buddy-rax-executor-test <output.rax>\n";
    return 1;
  }

  try {
    writeTestRax(argv[1]);
    auto manifest = buddy::runtime::ModelManifest::loadFromRax(argv[1]);
    if (manifest.functions.size() != 1 ||
        manifest.functions[0].inputs != std::vector<uint32_t>{1} ||
        manifest.functions[0].outputs != std::vector<uint32_t>{1} ||
        manifest.functions[0].temps != std::vector<uint32_t>{2} ||
        manifest.functions[0].ops.size() != 3)
      throw std::runtime_error("function metadata was not preserved");

    int value = 3;
    int increment = 2;
    int factor = 4;
    buddy::runtime::RaxExecutor executor(manifest);
    executor.bindBuffer(1, &value);
    executor.bindConstant(7, &increment);
    executor.bindConstant(8, &factor);
    executor.execute("pipeline");
    if (value != 20)
      throw std::runtime_error("dispatches did not execute in RAX order");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
