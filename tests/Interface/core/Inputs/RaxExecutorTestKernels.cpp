#include <cstdio>

extern "C" void kernel_a(void **args) {
  auto *value = static_cast<int *>(args[0]);
  auto *increment = static_cast<int *>(args[1]);
  *value += *increment;
}

extern "C" void kernel_b(void **args) {
  auto *factor = static_cast<int *>(args[0]);
  auto *value = static_cast<int *>(args[1]);
  *value *= *factor;
}

extern "C" void collective_producer(void **args) {
  auto *data = static_cast<float *>(args[0]);
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;
}

extern "C" void collective_consumer(void **args) {
  auto *data = static_cast<float *>(args[0]);
  data[0] += data[3];
}

extern "C" void deepseek_session_kernel(void **args) {
  auto *parameters = static_cast<int *>(args[0]);
  auto *input = static_cast<int *>(args[1]);
  auto *kvCache = static_cast<int *>(args[2]);
  auto *output = static_cast<int *>(args[3]);
  *output = *parameters + *input + *kvCache;
}

extern "C" void stage5d_rank0_prefill(void **args) {
  *static_cast<float *>(args[1]) = 1.0f;
  *static_cast<float *>(args[2]) = 10.0f;
  auto *logits = static_cast<float *>(args[3]);
  for (int i = 0; i < 2048; ++i)
    logits[i] = 0.0f;
  logits[5 * 2] = 10.0f;
}

extern "C" void stage5d_rank1_prefill(void **args) {
  *static_cast<float *>(args[1]) = 2.0f;
  *static_cast<float *>(args[2]) = 20.0f;
  auto *logits = static_cast<float *>(args[3]);
  for (int i = 0; i < 2048; ++i)
    logits[i] = 0.0f;
  logits[5 * 2] = 10.0f;
}

extern "C" void stage5d_rank0_decode(void **args) {
  const auto token = *static_cast<long long *>(args[0]);
  const auto position = *static_cast<long long *>(args[1]);
  const auto key = *static_cast<float *>(args[2]);
  *static_cast<long long *>(args[4]) = position;
  *static_cast<float *>(args[5]) = key;
  *static_cast<float *>(args[6]) = *static_cast<float *>(args[3]);
  static_cast<float *>(args[7])[1] = 1.0f;
  if (key == 1.0f)
    std::puts("Stage 5D.1 rank0 prefill/decode passed");
  if (key == 1.0f && token == 0 && position == 6)
    std::puts("Stage 5D.2 rank0 generation passed");
}

extern "C" void stage5d_rank1_decode(void **args) {
  const auto token = *static_cast<long long *>(args[0]);
  const auto position = *static_cast<long long *>(args[1]);
  const auto key = *static_cast<float *>(args[2]);
  *static_cast<long long *>(args[4]) = position;
  *static_cast<float *>(args[5]) = key;
  *static_cast<float *>(args[6]) = *static_cast<float *>(args[3]);
  static_cast<float *>(args[7])[1] = 1.0f;
  if (key == 2.0f)
    std::puts("Stage 5D.1 rank1 prefill/decode passed");
  if (key == 2.0f && token == 0 && position == 6)
    std::puts("Stage 5D.2 rank1 generation passed");
}

extern "C" void stage5d2_generation_prefill(void **args) {
  auto *tokens = static_cast<long long *>(args[0]);
  auto *key = static_cast<int *>(args[1]);
  auto *value = static_cast<int *>(args[2]);
  auto *logits = static_cast<float *>(args[3]);
  *key = static_cast<int>(tokens[0]);
  *value = static_cast<int>(tokens[1]);
  for (int i = 0; i < 40; ++i)
    logits[i] = 0.0f;
  logits[5 * 4 + 2] = 10.0f;
}

extern "C" void stage5d2_generation_decode(void **args) {
  const auto token = *static_cast<long long *>(args[0]);
  const auto position = *static_cast<long long *>(args[1]);
  const auto key = *static_cast<int *>(args[2]);
  const auto value = *static_cast<int *>(args[3]);
  auto *positionOut = static_cast<long long *>(args[4]);
  auto *keyOut = static_cast<int *>(args[5]);
  auto *valueOut = static_cast<int *>(args[6]);
  auto *logits = static_cast<float *>(args[7]);
  *positionOut = position;
  *keyOut = key + static_cast<int>(token);
  *valueOut = value;
  for (int i = 0; i < 4; ++i)
    logits[i] = 0.0f;
  if (position == 6 && token == 2 && key == 151646)
    logits[1] = 10.0f;
  else if (position == 7 && token == 1 && key == 151648)
    logits[3] = 10.0f;
}
