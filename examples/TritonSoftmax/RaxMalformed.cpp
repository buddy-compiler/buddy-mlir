// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Board-side fixture generator: all fixtures use the actual RAX schema.
#include "SoftmaxTarget.h"
#include "buddy/runtime/rax/RAX.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace rhal::rax;

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  for (const std::string mode :
       {"valid",          "unknown-code",     "unknown-buffer",
        "missing-scalar", "extra-buffer",     "missing-buffer",
        "interleaved",    "constant",         "dual-arg",
        "float-scalar",   "string-scalar",    "multi-function",
        "multi-dispatch", "non-dispatch",     "bad-grid",
        "bad-block",      "duplicate-buffer", "duplicate-code",
        "barrier-before", "barrier-repeat",   "barrier-attrs"}) {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<KV>> attrs;
    for (auto [key, value] : std::vector<std::pair<std::string, std::string>>{
             {"optional", "true"},
             {"kernel_abi", "buddy.external.v1"},
             {"operation", "softmax"},
             {"dtype", "f32"},
             {"layout", "row_major"},
             {"shape", "2,4"},
             {"strides", "4,1"},
             {"axis", "1"},
             {"target", "riscv64-unknown-linux-gnu"},
             {"required_isa", TRITON_SOFTMAX_REQUIRED_FEATURES},
             {"profile", "softmax.f32.2x4.rvv1.vlen256.v1"},
             {"required_rvv_version", "1.0"},
             {"rvv_vlen_bits", "256"},
             {"required_matrix_features", "none"}})
      attrs.push_back(CreateKV(b, b.CreateString(key), b.CreateString(value)));
    auto kernel =
        CreateCodeObject(b, mode == "duplicate-code" ? 1 : 2,
                         b.CreateString("kernel"), CodeObjectKind_HostSharedLib,
                         0, b.CreateString("file:libtriton_softmax.so"),
                         b.CreateString("buddy_external_kernel_query_v1"),
                         b.CreateString("triton_cpu"), b.CreateVector(attrs));
    auto primary = CreateCodeObject(
        b, 1, b.CreateString("computation"), CodeObjectKind_HostSharedLib, 0,
        b.CreateString("file:softmax_runner.so"), 0, b.CreateString("cpu"));
    std::vector<flatbuffers::Offset<CodeObject>> codes{kernel, primary};
    std::vector<int64_t> shape{2, 4}, strides{4, 1};
    auto type =
        CreateTensorType(b, DType_F32, CreateShape(b, b.CreateVector(shape)),
                         Layout_RowMajor, b.CreateVector(strides));
    auto in = CreateBufferBinding(b, 1, b.CreateString("input"), type,
                                  MemorySpace_Host);
    auto out =
        CreateBufferBinding(b, mode == "duplicate-buffer" ? 1 : 2,
                            b.CreateString("output"), type, MemorySpace_Host);
    std::vector<flatbuffers::Offset<BufferBinding>> buffers{in, out};
    auto arg0 = CreateArg(b, mode == "unknown-buffer" ? 999 : 1,
                          mode == "dual-arg" ? 1 : 0);
    auto arg1 = CreateArg(b, 2);
    auto scalarValue = CreateScalar(
        b, mode == "float-scalar" ? 0 : -1, mode == "float-scalar" ? -1.0 : 0.0,
        mode == "string-scalar" ? b.CreateString("axis") : 0);
    auto scalar = CreateArg(b, 0, 0, scalarValue);
    std::vector<flatbuffers::Offset<Arg>> args{arg0, arg1, scalar};
    if (mode == "missing-scalar")
      args.pop_back();
    if (mode == "missing-buffer")
      args = {arg0, scalar};
    if (mode == "extra-buffer")
      args = {arg0, arg1, arg0, scalar};
    if (mode == "interleaved")
      args = {arg0, scalar, arg1};
    if (mode == "constant")
      args = {CreateArg(b, 0, 1), arg1, scalar};
    std::vector<uint32_t> grid{1, 1, 1}, block{1, 1, 1};
    if (mode == "bad-grid")
      grid[0] = 2;
    if (mode == "bad-block")
      block[0] = 2;
    auto dims =
        CreateLaunchDims(b, b.CreateVector(grid), b.CreateVector(block));
    auto dispatch = CreateDispatchOp(b, mode == "unknown-code" ? 999 : 2,
                                     b.CreateVector(args), dims);
    auto op = CreateOp(
        b, mode == "non-dispatch" ? OpKind_Barrier : OpKind_Dispatch, dispatch);
    std::vector<flatbuffers::Offset<Op>> ops{op};
    if (mode.rfind("barrier-", 0) == 0) {
      std::vector<flatbuffers::Offset<KV>> barrierAttrs;
      if (mode == "barrier-attrs")
        barrierAttrs.push_back(
            CreateKV(b, b.CreateString("scope"), b.CreateString("device")));
      auto barrier = CreateOp(
          b, OpKind_Barrier, 0, 0, 0,
          CreateBarrierOp(
              b, barrierAttrs.empty() ? 0 : b.CreateVector(barrierAttrs)));
      ops = mode == "barrier-before"
                ? std::vector<flatbuffers::Offset<Op>>{barrier, op}
                : std::vector<flatbuffers::Offset<Op>>{op, barrier};
      if (mode == "barrier-repeat")
        ops.push_back(barrier);
    }
    if (mode == "multi-dispatch")
      ops.push_back(op);
    std::vector<uint32_t> inputs{1}, outputs{2};
    auto fn =
        CreateFunction(b, b.CreateString("softmax"), b.CreateVector(inputs),
                       b.CreateVector(outputs), 0, b.CreateVector(ops));
    std::vector<flatbuffers::Offset<Function>> functions{fn};
    if (mode == "multi-function")
      functions.push_back(fn);
    std::vector<flatbuffers::Offset<KV>> moduleAttrs{
        CreateKV(b, b.CreateString("model_name"),
                 b.CreateString("softmax_contract")),
        CreateKV(b, b.CreateString("runner_library"),
                 b.CreateString("file:softmax_runner.so"))};
    auto module =
        CreateModule(b, b.CreateString("RAX"), 0, Endianness_Little,
                     b.CreateVector(moduleAttrs), 0, b.CreateVector(buffers), 0,
                     b.CreateVector(codes), b.CreateVector(functions));
    b.Finish(module, "RAX0");
    std::ofstream file(std::filesystem::path(argv[1]) /
                           ("raw-" + mode + ".rax"),
                       std::ios::binary);
    file.write(reinterpret_cast<const char *>(b.GetBufferPointer()),
               b.GetSize());
    if (!file)
      return 2;
  }
}
