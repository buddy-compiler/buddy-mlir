//===- buddy-stable-diffusion-main.cpp-------------------------------------===//
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
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImgContainer.h>
#include <buddy/DIP/ImageContainer.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace buddy;

// Capture input message.
void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease input your prompt:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

void getInferenceSteps(int &input) {
  std::cout << "Please enter the number of inference steps:" << std::endl;
  std::cout << ">>> ";
  std::cin >> input;
  std::cout << std::endl;
}

void getFileName(std::string &input) {
  std::cout << "Please enter the file name of the generated image:"
            << std::endl;
  std::cout << ">>> ";
  std::cin >> input;
  std::cout << std::endl;
}

void printIterInfo(size_t iterIdx, double time) {
  std::cout << "\033[32;1m[Denoising steps " << iterIdx << "] \033[0m";
  std::cout << "Time: " << time << "s" << std::endl;
}

// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

// Load parameters into data container.
void loadParametersInt64(const std::string &int64ParamPath,
                         MemRef<long long, 1> &int64Param) {

  std::ifstream int64ParamFile(int64ParamPath, std::ios::in | std::ios::binary);
  if (!int64ParamFile.is_open()) {
    std::string errMsg = "Failed to open int64 param file: " +
                         std::filesystem::canonical(int64ParamPath).string();
    throw std::runtime_error(errMsg);
  }
  int64ParamFile.read(reinterpret_cast<char *>(int64Param.getData()),
                      int64Param.getSize() * sizeof(long long));
  if (int64ParamFile.fail()) {
    throw std::runtime_error("Failed to read int64 param file");
  }
  int64ParamFile.close();
}

// Load parameters into data container.
void loadParametersFloat(const std::string &floatParamPath,
                         MemRef<float, 1> &floatParam) {
  std::ifstream floatParamFile(floatParamPath, std::ios::in | std::ios::binary);
  if (!floatParamFile.is_open()) {
    std::string errMsg = "Failed to open float param file: " +
                         std::filesystem::canonical(floatParamPath).string();
    throw std::runtime_error(errMsg);
  }
  floatParamFile.read(reinterpret_cast<char *>(floatParam.getData()),
                      floatParam.getSize() * sizeof(float));
  if (floatParamFile.fail()) {
    throw std::runtime_error("Failed to read float param file");
  }
  floatParamFile.close();
}

void fill_random_normal(MemRef<float, 4> &input, size_t size, unsigned seed) {
  std::mt19937 generator(seed);
  std::normal_distribution<float> distribution(0.0f, 1.0f);
  for (size_t i = 0; i < size; i++) {
    input.getData()[i] = distribution(generator);
  }
}

// SchedulerConfig structure, which contains the necessary configuration
// information
struct SchedulerConfig {
  std::string prediction_type; // Prediction type: 'epsilon' or 'v_prediction'
  MemRef<float, 1> alphas_cumprod; // Store alpha t
  MemRef<float, 4> cur_sample;
  float final_alpha_cumprod; // Final alpha

  SchedulerConfig(const MemRef<float, 1> &alphas,
                  const MemRef<float, 4> &sample)
      : alphas_cumprod(alphas), cur_sample(sample) {}
};

MemRef<float, 1> generate_betas(float beta_start, float beta_end,
                                size_t num_train_timesteps) {
  MemRef<float, 1> betas({num_train_timesteps});

  //  Calculate the square root range
  float start_sqrt = std::sqrt(beta_start);
  float end_sqrt = std::sqrt(beta_end);
  for (size_t i = 0; i < num_train_timesteps; i++) {
    float t = static_cast<float>(i) /
              (num_train_timesteps - 1); // Calculate the scale
    float value =
        start_sqrt + t * (end_sqrt - start_sqrt); // Linear interpolation
    betas[i] = value * value;                     // square
  }
  return betas;
}

// Auxiliary function: scalar multiplication of multidimensional arrays
MemRef<float, 4> memref_mul_scalar(const MemRef<float, 4> &memref,
                                   float scalar) {
  MemRef<float, 4> result({1, 4, 64, 64});
  for (int i = 0; i < 1 * 4 * 64 * 64; ++i) {
    result[i] = memref[i] * scalar;
  }
  return result;
}

// Auxiliary function: Addition of multi-dimensional arrays
MemRef<float, 4> memref_add(const MemRef<float, 4> &a,
                            const MemRef<float, 4> &b) {
  MemRef<float, 4> result({1, 4, 64, 64});
  for (int i = 0; i < 1 * 4 * 64 * 64; ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

// get_prev_sample function implementation
MemRef<float, 4> get_prev_sample(const MemRef<float, 4> &sample, int timestep,
                                 int prev_timestep,
                                 const MemRef<float, 4> &model_output,
                                 SchedulerConfig &config) {
  float alpha_prod_t = config.alphas_cumprod.getData()[timestep];
  float alpha_prod_t_prev = (prev_timestep >= 0)
                                ? config.alphas_cumprod.getData()[prev_timestep]
                                : config.final_alpha_cumprod;
  float beta_prod_t = 1.0f - alpha_prod_t;
  float beta_prod_t_prev = 1.0f - alpha_prod_t_prev;
  MemRef<float, 4> prev_sample({1, 4, 64, 64});

  // Processing prediction type
  if (config.prediction_type == "v_prediction") {
    // v_prediction formula
    for (int i = 0; i < 1 * 4 * 64 * 64; ++i) {
      prev_sample[i] = std::sqrt(alpha_prod_t) * model_output[i] +
                       std::sqrt(beta_prod_t) * sample[i];
    }
  } else if (config.prediction_type != "epsilon") {
    throw std::invalid_argument(
        "prediction_type must be one of `epsilon` or `v_prediction`");
  }

  // Calculate sample_coeff
  float sample_coeff = std::sqrt(alpha_prod_t_prev / alpha_prod_t);

  // Calculate model_output_denom_coeff
  float model_output_denom_coeff =
      alpha_prod_t * std::sqrt(beta_prod_t_prev) +
      std::sqrt(alpha_prod_t * beta_prod_t * alpha_prod_t_prev);
  // Apply formula (9) to calculate prev_sample
  for (int i = 0; i < 1 * 4 * 64 * 64; ++i) {
    prev_sample[i] = sample_coeff * sample[i] -
                     (alpha_prod_t_prev - alpha_prod_t) * model_output[i] /
                         model_output_denom_coeff;
  }

  return prev_sample;
}

// The core function step_plms performs the inference step
MemRef<float, 4> step_plms(const MemRef<float, 4> model_output, int timestep,
                           MemRef<float, 4> sample, int num_inference_steps,
                           SchedulerConfig &config, int &counter,
                           std::vector<MemRef<float, 4>> &ets) {
  int prev_timestep = timestep - 1000 / num_inference_steps;
  if (counter != 1) {
    if (ets.size() > 3)
      ets.erase(ets.begin(), ets.begin() + ets.size() - 3);
    ets.push_back(model_output);
  } else {
    prev_timestep = timestep;
    timestep += 1000 / num_inference_steps;
  }

  MemRef<float, 4> updated_model_output({1, 4, 64, 64});

  if (ets.size() == 1 && counter == 0) {
    updated_model_output = model_output;
    config.cur_sample = sample;
  } else if (ets.size() == 1 && counter == 1) {
    updated_model_output =
        memref_mul_scalar(memref_add(model_output, ets.back()), 0.5);
    sample = config.cur_sample;
  } else if (ets.size() == 2) {
    updated_model_output = memref_mul_scalar(
        memref_add(memref_mul_scalar(ets.back(), 3.0),
                   memref_mul_scalar(ets[ets.size() - 2], -1.0)),
        0.5);
  } else if (ets.size() == 3) {
    updated_model_output = memref_mul_scalar(
        memref_add(memref_add(memref_mul_scalar(ets.back(), 23.0),
                              memref_mul_scalar(ets[ets.size() - 2], -16.0)),
                   memref_mul_scalar(ets[ets.size() - 3], 5.0)),
        1.0 / 12.0);
  } else {
    updated_model_output = memref_mul_scalar(
        memref_add(memref_add(memref_add(memref_mul_scalar(ets.back(), 55.0),
                                         memref_mul_scalar(ets[ets.size() - 2],
                                                           -59.0)),
                              memref_mul_scalar(ets[ets.size() - 3], 37.0)),
                   memref_mul_scalar(ets[ets.size() - 4], -9.0)),
        1.0 / 24.0);
  }

  MemRef<float, 4> prev_sample = get_prev_sample(
      sample, timestep, prev_timestep, updated_model_output, config);

  return prev_sample;
}

std::vector<int> set_timesteps(int num_inference_steps) {
  std::vector<int> timesteps;
  std::vector<int> prk_timesteps;
  std::vector<int> plms_timesteps;
  int step_ratio = 1000 / num_inference_steps;
  timesteps.resize(num_inference_steps);
  for (int i = 0; i < num_inference_steps; ++i) {
    timesteps[i] = static_cast<int>(round(i * step_ratio)) + 1;
  }

  prk_timesteps.clear();
  plms_timesteps.resize(timesteps.size() - 1 + 2);
  std::copy(timesteps.begin(), timesteps.end() - 1, plms_timesteps.begin());
  if (num_inference_steps > 1)
    plms_timesteps[plms_timesteps.size() - 2] = timesteps[timesteps.size() - 2];
  plms_timesteps[plms_timesteps.size() - 1] = timesteps.back();
  std::reverse(plms_timesteps.begin(), plms_timesteps.end());

  timesteps = prk_timesteps; // Adjust as needed
  timesteps.insert(timesteps.end(), plms_timesteps.begin(),
                   plms_timesteps.end());
  if (num_inference_steps == 1)
    timesteps.resize(1);
  return timesteps;
}

struct MemRefContainer {
  MemRef<float, 3> memRef3D;
  MemRef<float, 2> memRef2D;

  MemRefContainer(MemRef<float, 3> m1, MemRef<float, 2> m2)
      : memRef3D(m1), memRef2D(m2) {}
};

extern "C" void _mlir_ciface_forward_text_encoder(MemRefContainer *result,
                                                  MemRef<float, 1> *arg0,
                                                  MemRef<long long, 1> *arg1,
                                                  MemRef<long long, 2> *arg2);

extern "C" void _mlir_ciface_forward_unet(MemRef<float, 4> *result,
                                          MemRef<float, 1> *arg0,
                                          MemRef<float, 4> *arg1,
                                          MemRef<float, 1> *arg2,
                                          MemRef<float, 3> *arg3);

extern "C" void _mlir_ciface_forward_vae(MemRef<float, 4> *result,
                                         MemRef<float, 1> *arg0,
                                         MemRef<float, 4> *arg1);

int main() {
  const std::string title =
      "Stable Diffusion Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  // Define directories of vacabulary and parameter file.
  std::string stableDiffusionDir = STABLE_DIFFUSION_EXAMPLE_PATH;
  std::string stableDiffusionBuildDir = STABLE_DIFFUSION_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = stableDiffusionDir + "/vocab.txt";
  const std::string TextEncoderParamsDir1 =
      stableDiffusionBuildDir + "/arg0_text_encoder.data";
  const std::string TextEncoderParamsDir2 =
      stableDiffusionBuildDir + "/arg1_text_encoder.data";
  const std::string UnetParamsDir =
      stableDiffusionBuildDir + "/arg0_unet.data";
  const std::string VaeParamsDir =
      stableDiffusionBuildDir + "/arg0_vae.data";

  // Get user message.
  std::string inputStr;
  std::string image_name;
  int InferenceSteps;
  getUserInput(inputStr);
  getInferenceSteps(InferenceSteps);
  getFileName(image_name);
  // Define the text_encoder parameter
  MemRef<float, 3> myMemRef1({1, 77, 1024});
  MemRef<float, 2> myMemRef2({1, 1024});
  MemRefContainer resultTextEncoderPos(myMemRef1, myMemRef2);
  MemRefContainer resultTextEncoderNeg(myMemRef1, myMemRef2);
  MemRef<float, 3> TextEncoderOut({2, 77, 1024});
  MemRefContainer *ptrPos = &resultTextEncoderPos;
  MemRefContainer *ptrNeg = &resultTextEncoderNeg;
  MemRef<float, 1> arg0_text_encoder({340387840});
  MemRef<long long, 1> arg1_text_encoder({77});
  Text<long long, 2> TextEncoderInputIDsPos(inputStr);
  TextEncoderInputIDsPos.tokenizeStableDiffusion(vocabDir, 77);
  Text<long long, 2> TextEncoderInputIDsNeg("");
  TextEncoderInputIDsNeg.tokenizeStableDiffusion(vocabDir, 77);
  // Define unet parameters
  MemRef<float, 4> resultUnet({1, 4, 64, 64});
  MemRef<float, 1> arg0_unet({865910724});
  MemRef<float, 4> latents({1, 4, 64, 64});
  MemRef<float, 1> timestep({1});
  // Define vae parameters
  MemRef<float, 4> resultVae({1, 3, 512, 512});
  MemRef<float, 1> arg0_vae({49490199});

  // Output directory information
  printLogLabel();
  std::cout << "Vocab file: " << std::filesystem::canonical(vocabDir)
            << std::endl;
  printLogLabel();
  std::cout << "Params file: " << std::endl
            << std::filesystem::canonical(TextEncoderParamsDir1) << std::endl
            << std::filesystem::canonical(TextEncoderParamsDir2) << std::endl
            << std::filesystem::canonical(UnetParamsDir) << std::endl
            << std::filesystem::canonical(VaeParamsDir) << std::endl;

  // Loading model parameters
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  const auto loadStart = std::chrono::high_resolution_clock::now();
  loadParametersFloat(TextEncoderParamsDir1, arg0_text_encoder);
  loadParametersInt64(TextEncoderParamsDir2, arg1_text_encoder);
  loadParametersFloat(UnetParamsDir, arg0_unet);
  loadParametersFloat(VaeParamsDir, arg0_vae);
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;

  // Encode prompt
  printLogLabel();
  std::cout << "Encoding prompt..." << std::endl;
  const auto encodeStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_text_encoder(
      ptrPos, &arg0_text_encoder, &arg1_text_encoder, &TextEncoderInputIDsPos);
  _mlir_ciface_forward_text_encoder(
      ptrNeg, &arg0_text_encoder, &arg1_text_encoder, &TextEncoderInputIDsNeg);
  const auto encodeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> encodeTime =
      encodeEnd - encodeStart;
  printLogLabel();
  std::cout << "Prompt encode time: " << (double)(encodeTime.count()) / 1000
            << "s\n"
            << std::endl;
  // Concatenation of Positive and Negative embeddings
  auto TextEncoderOutPos = ptrPos->memRef3D;
  auto TextEncoderOutNeg = ptrNeg->memRef3D;
  for (int i = 0; i < 2 * 77 * 1024; i++) {
    if (i < 1 * 77 * 1024)
      TextEncoderOut.getData()[i] = TextEncoderOutNeg.getData()[i];
    else
      TextEncoderOut.getData()[i] =
          TextEncoderOutPos.getData()[i % (1 * 77 * 1024)];
  }
  // Generate initial noise
  fill_random_normal(latents, 1 * 4 * 64 * 64, 42);

  printLogLabel();
  std::cout << "Start denoising..." << std::endl;
  // Set config
  MemRef<float, 1> alphas_cumprod({1000});
  MemRef<float, 4> cur_sample({1, 4, 64, 64});
  SchedulerConfig config(alphas_cumprod, cur_sample);
  alphas_cumprod = generate_betas(0.00085, 0.012, 1000);
  for (int i = 0; i < 1000; i++) {
    alphas_cumprod.getData()[i] = 1.0 - alphas_cumprod.getData()[i];
    if (i >= 1)
      alphas_cumprod.getData()[i] =
          alphas_cumprod.getData()[i] * alphas_cumprod.getData()[i - 1];
    config.alphas_cumprod.getData()[i] = alphas_cumprod.getData()[i];
  }
  config.final_alpha_cumprod = config.alphas_cumprod.getData()[0];
  config.prediction_type = "epsilon";
  std::vector<MemRef<float, 4>> ets;
  auto timesteps = set_timesteps(InferenceSteps);

  // Denoising loop
  const auto inferenceTotalStart = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < (int)timesteps.size(); i++) {
    MemRef<float, 4> noise({2, 4, 64, 64});
    for (int j = 0; j < 2 * 4 * 64 * 64; j++)
      noise.getData()[j] = latents.getData()[j % (1 * 4 * 64 * 64)];

    timestep.getData()[0] = timesteps[i];
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    _mlir_ciface_forward_unet(&resultUnet, &arg0_unet, &noise, &timestep,
                              &TextEncoderOut);
    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    printIterInfo(i, inferenceTime.count() / 1000);
    MemRef<float, 4> pred_noise({2, 4, 64, 64});
    for (int j = 0; j < 1 * 4 * 64 * 64; j++) {
      pred_noise.getData()[j] =
          resultUnet.getData()[j] +
          7.5 * (resultUnet.getData()[j + 1 * 4 * 64 * 64] -
                 resultUnet.getData()[j]);
    }
    latents = step_plms(pred_noise, timesteps[i], latents, InferenceSteps,
                        config, i, ets);
  }
  const auto inferenceTotalEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTotalTime =
      inferenceTotalEnd - inferenceTotalStart;
  printLogLabel();
  std::cout << "Denoising complete." << std::endl;
  printLogLabel();
  std::cout << "Total time spent on denoising: "
            << (double)(inferenceTotalTime.count()) / 1000 << "s" << std::endl;

  for (int i = 0; i < 1 * 4 * 64 * 64; i++) {
    latents.getData()[i] = latents.getData()[i] / 0.18215;
  }

  // Decode
  std::cout << std::endl;
  printLogLabel();
  std::cout << "Start decoding..." << std::endl;
  const auto decodeStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_vae(&resultVae, &arg0_vae, &latents);
  const auto decodeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> decodeTime =
      decodeEnd - decodeStart;
  printLogLabel();
  std::cout << "Decode time: " << (double)(decodeTime.count()) / 1000 << "s\n"
            << std::endl;

  for (int i = 0; i < 1 * 3 * 512 * 512; i++) {
    resultVae.getData()[i] = (resultVae.getData()[i] + 1) / 2;
    // clamp(0, 1)
    if (resultVae.getData()[i] < 0)
      resultVae.getData()[i] = 0;
    if (resultVae.getData()[i] > 1)
      resultVae.getData()[i] = 1;
    resultVae.getData()[i] = resultVae.getData()[i] * 255;
  }
  intptr_t sizes[4] = {1, 3, 512, 512};
  Image<float, 4> img(resultVae.getData(), sizes);

  const std::string Imgfilename =
      stableDiffusionBuildDir + "/" + image_name + ".bmp";
  // Call the imageWrite function
  imageWrite(Imgfilename, img);

  printLogLabel();
  std::cout << "The prompt used to generate the image:" << inputStr
            << std::endl;

  printLogLabel();
  std::cout << "Image saved successfully to "
            << std::filesystem::canonical(Imgfilename) << std::endl;

  return 0;
}

