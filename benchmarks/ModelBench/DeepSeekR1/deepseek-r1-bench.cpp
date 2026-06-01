//===- deepseek-r1-bench.cpp ----------------------------------------------===//
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

#ifndef DEEPSEEKR1_EXAMPLE_PATH
#define DEEPSEEKR1_EXAMPLE_PATH ""
#endif
#ifndef DEEPSEEKR1_EXAMPLE_BUILD_PATH
#define DEEPSEEKR1_EXAMPLE_BUILD_PATH ""
#endif

#define main deepseek_r1_interactive_main
#include "../../../examples/BuddyDeepSeekR1/buddy-deepseek-r1-main.cpp"
#undef main

static PrefillReturns makePrefillReturns(MemRef<float, 3> &logits) {
  auto makeKV = []() {
    return MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.0f);
  };
  return {makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(),
          makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(),
          makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(),
          makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(),
          makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(),
          makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(),
          makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(),
          makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(), makeKV(),
          logits};
}

static DecodeReturns makeDecodeReturns(MemRef<float, 3> &logits) {
  auto makeKV = []() {
    return MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.0f);
  };
  auto makeDummy = []() { return MemRef<long long, 1>({1}, 0LL); };
  return {makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          makeDummy(), makeKV(), makeKV(), makeDummy(), makeKV(), makeKV(),
          logits};
}

static void setDecodeDummies(DecodeReturns &ret, long long cachePosition) {
  ret.ret_dummy0.getData()[0] = cachePosition;
  ret.ret_dummy1.getData()[0] = cachePosition;
  ret.ret_dummy2.getData()[0] = cachePosition;
  ret.ret_dummy3.getData()[0] = cachePosition;
  ret.ret_dummy4.getData()[0] = cachePosition;
  ret.ret_dummy5.getData()[0] = cachePosition;
  ret.ret_dummy6.getData()[0] = cachePosition;
  ret.ret_dummy7.getData()[0] = cachePosition;
  ret.ret_dummy8.getData()[0] = cachePosition;
  ret.ret_dummy9.getData()[0] = cachePosition;
  ret.ret_dummy10.getData()[0] = cachePosition;
  ret.ret_dummy11.getData()[0] = cachePosition;
  ret.ret_dummy12.getData()[0] = cachePosition;
  ret.ret_dummy13.getData()[0] = cachePosition;
  ret.ret_dummy14.getData()[0] = cachePosition;
  ret.ret_dummy15.getData()[0] = cachePosition;
  ret.ret_dummy16.getData()[0] = cachePosition;
  ret.ret_dummy17.getData()[0] = cachePosition;
  ret.ret_dummy18.getData()[0] = cachePosition;
  ret.ret_dummy19.getData()[0] = cachePosition;
  ret.ret_dummy20.getData()[0] = cachePosition;
  ret.ret_dummy21.getData()[0] = cachePosition;
  ret.ret_dummy22.getData()[0] = cachePosition;
  ret.ret_dummy23.getData()[0] = cachePosition;
  ret.ret_dummy24.getData()[0] = cachePosition;
  ret.ret_dummy25.getData()[0] = cachePosition;
  ret.ret_dummy26.getData()[0] = cachePosition;
}

static void runDecode(DecodeReturns &ret, MemRef<float, 1> &params,
                      MemRef<long long, 2> &input,
                      MemRef<long long, 1> &cachePosition) {
  _mlir_ciface_forward_decode(
      &ret, &params, &input, &cachePosition, &ret.kv0, &ret.kv1,
      &ret.ret_dummy0, &ret.kv2, &ret.kv3, &ret.ret_dummy1, &ret.kv4, &ret.kv5,
      &ret.ret_dummy2, &ret.kv6, &ret.kv7, &ret.ret_dummy3, &ret.kv8, &ret.kv9,
      &ret.ret_dummy4, &ret.kv10, &ret.kv11, &ret.ret_dummy5, &ret.kv12,
      &ret.kv13, &ret.ret_dummy6, &ret.kv14, &ret.kv15, &ret.ret_dummy7,
      &ret.kv16, &ret.kv17, &ret.ret_dummy8, &ret.kv18, &ret.kv19,
      &ret.ret_dummy9, &ret.kv20, &ret.kv21, &ret.ret_dummy10, &ret.kv22,
      &ret.kv23, &ret.ret_dummy11, &ret.kv24, &ret.kv25, &ret.ret_dummy12,
      &ret.kv26, &ret.kv27, &ret.ret_dummy13, &ret.kv28, &ret.kv29,
      &ret.ret_dummy14, &ret.kv30, &ret.kv31, &ret.ret_dummy15, &ret.kv32,
      &ret.kv33, &ret.ret_dummy16, &ret.kv34, &ret.kv35, &ret.ret_dummy17,
      &ret.kv36, &ret.kv37, &ret.ret_dummy18, &ret.kv38, &ret.kv39,
      &ret.ret_dummy19, &ret.kv40, &ret.kv41, &ret.ret_dummy20, &ret.kv42,
      &ret.kv43, &ret.ret_dummy21, &ret.kv44, &ret.kv45, &ret.ret_dummy22,
      &ret.kv46, &ret.kv47, &ret.ret_dummy23, &ret.kv48, &ret.kv49,
      &ret.ret_dummy24, &ret.kv50, &ret.kv51, &ret.ret_dummy25, &ret.kv52,
      &ret.kv53, &ret.ret_dummy26, &ret.kv54, &ret.kv55);
}

int main() {
  MemRef<float, 1> params({ParamsSize}, MemRefAllocationKind::MMap);
  MemRef<size_t, 2> prefillInput({1, MaxTokenLength}, static_cast<size_t>(0));
  MemRef<long long, 2> decodeInput({1, 1}, 0LL);
  MemRef<long long, 1> cachePosition({1}, 200LL);
  MemRef<float, 3> prefillLogits({1, MaxTokenLength, MaxVocabSize});
  MemRef<float, 3> decodeLogits({1, 1, MaxVocabSize});

  PrefillReturns prefillRet = makePrefillReturns(prefillLogits);
  DecodeReturns decodeRet = makeDecodeReturns(decodeLogits);
  setDecodeDummies(decodeRet, cachePosition.getData()[0]);

  const auto prefillStart = std::chrono::steady_clock::now();
  _mlir_ciface_forward_prefill(
      &prefillRet, &params, reinterpret_cast<Text<size_t, 2> *>(&prefillInput));
  const auto prefillEnd = std::chrono::steady_clock::now();

  constexpr int DecodeIterations = 5;
  double decodeSeconds = 0.0;
  for (int i = 0; i < DecodeIterations; ++i) {
    setDecodeDummies(decodeRet, cachePosition.getData()[0]);
    const auto decodeStart = std::chrono::steady_clock::now();
    runDecode(decodeRet, params, decodeInput, cachePosition);
    const auto decodeEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<double> decodeTime = decodeEnd - decodeStart;
    decodeSeconds += decodeTime.count();
  }

  const std::chrono::duration<double> prefillTime = prefillEnd - prefillStart;
  const double decodeAverageSeconds =
      decodeSeconds / static_cast<double>(DecodeIterations);
  volatile float sink =
      prefillRet.logits.getData()[0] + decodeRet.logits.getData()[0];
  (void)sink;

  std::cout << "model,precision,prefill_s,decode_s,decode_iters\n"
            << "deepseek-r1,f32," << prefillTime.count() << ","
            << decodeAverageSeconds << "," << DecodeIterations << "\n";
  return 0;
}
