#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace buddy;

/// Capture input message.
void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease input your prompt:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

void getTimeSteps(float &input) {
  std::cout << "\nPlease input timesteps:" << std::endl;
  std::cout << ">>> ";
  std::cin >> input;
  std::cout << std::endl;
}

void printIterInfo(size_t iterIdx, double time) {
  std::cout << "\033[32;1m[Denoising steps " << iterIdx << "] \033[0m";
  std::cout << "Time: " << time << "s" << std::endl;
}

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }


/// Load parameters into data container.
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

/// Load parameters into data container.
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

void fill_random_normal(MemRef<float, 4> &input,  size_t size, unsigned seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0 ; i < size ; i ++ ) {
        input.getData()[i] = distribution(generator);
    }
}

struct MemRefContainer {
    MemRef<float, 3> memRef3D;
    MemRef<float, 2> memRef2D;

  MemRefContainer(MemRef<float, 3> m1, MemRef<float, 2> m2)
        : memRef3D(m1), memRef2D(m2) {}
};

extern "C" void
_mlir_ciface_forward_text_encoder(MemRefContainer *result,
                     MemRef<float, 1> *arg0, MemRef<long long, 1> *arg1, 
                     MemRef<long long, 2> *arg2, MemRef<long long, 2> *arg3);

extern "C" void
_mlir_ciface_forward_unet(MemRef<float, 4> *result1, 
                     MemRef<float, 1> *arg0, MemRef<float, 4> *arg1, 
                     MemRef<float, 1> *arg2, MemRef<float, 3> *arg3);

extern "C" void
_mlir_ciface_forward_vae(MemRef<float, 4> *result1, 
                         MemRef<float, 1> *arg0,
                         MemRef<float, 4> *arg1);

int main() {
  const std::string title = "StableDiffusion Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vacabulary and parameter file.
  const std::string vocabDir = "../../examples/BuddyStableDiffusion/vocab_sd.txt";
  const std::string TextEncoderParamsDir1 = "../../examples/BuddyStableDiffusion/arg0_text_encoder.data";
  const std::string TextEncoderParamsDir2 = "../../examples/BuddyStableDiffusion/arg1_text_encoder.data";
  const std::string UnetParamsDir = "../../examples/BuddyStableDiffusion/arg0_unet.data";
  const std::string VaeParamsDir = "../../examples/BuddyStableDiffusion/arg0_vae.data";

  /// Get user message.
  std::string inputStr;
  float timesteps;
  getUserInput(inputStr);
  getTimeSteps(timesteps);

  
  MemRef<float, 3> myMemRef1({1, 77, 1024});
  MemRef<float, 2> myMemRef2({1, 1024});
  MemRefContainer resultTextEncoderPos(myMemRef1, myMemRef2);
  MemRefContainer resultTextEncoderNeg(myMemRef1, myMemRef2);
  MemRef<float, 3> TextEncoderOut({2, 77, 1024});
  MemRefContainer* ptrPos = &resultTextEncoderPos;
  MemRefContainer* ptrNeg = &resultTextEncoderNeg;
  MemRef<float, 1> arg0_text_encoder({340387840});
  MemRef<long long, 1> arg1_text_encoder({77});
  Text<long long, 2> TextEncoderInputIDsPos(inputStr);
  TextEncoderInputIDsPos.tokenizeStableDiffusion(vocabDir, 77);
  Text<long long, 2> TextEncoderInputIDsNeg("");
  TextEncoderInputIDsNeg.tokenizeStableDiffusion(vocabDir, 77);
  MemRef<long long, 2> attention_mask_pos({1, 77}, 0LL);
  MemRef<long long, 2> attention_mask_neg({1, 77}, 0LL);
  for (int i = 0; i < 77; i++) {
    attention_mask_pos.getData()[i] = 1;
    if (TextEncoderInputIDsPos.getData()[i] == 49407) break;
  }
  for (int i = 0; i < 77; i++) {
    attention_mask_neg.getData()[i] = 1;
    if (TextEncoderInputIDsNeg.getData()[i] == 49407) break;
  }

  MemRef<float, 4> resultUnet({1, 4, 96, 96});
  MemRef<float, 1> arg0_unet({865910724});
  MemRef<float, 4> latents({1, 4, 96, 96});
  MemRef<float, 1> timestep({999});

  MemRef<float, 4> resultVae({1, 3, 768, 768});
  MemRef<float, 1> arg0_vae({49490179});


  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  const auto loadStart = std::chrono::high_resolution_clock::now();
  loadParametersFloat(TextEncoderParamsDir1, arg0_text_encoder);
  loadParametersInt64(TextEncoderParamsDir2, arg1_text_encoder);
  loadParametersFloat(UnetParamsDir, arg0_unet);
  loadParametersFloat(VaeParamsDir, arg0_vae);
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime = loadEnd - loadStart;
  printLogLabel();
  std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;
  

  printLogLabel();
  std::cout << "Encoding prompt..." << std::endl;
  const auto encodeStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_text_encoder(ptrPos, &arg0_text_encoder, &arg1_text_encoder, &TextEncoderInputIDsPos, &attention_mask_pos);
  _mlir_ciface_forward_text_encoder(ptrNeg, &arg0_text_encoder, &arg1_text_encoder, &TextEncoderInputIDsNeg, &attention_mask_neg);
  const auto encodeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> encodeTime = encodeEnd - encodeStart;
  printLogLabel();
  std::cout << "Encode prompt time: " << (double)(encodeTime.count()) / 1000
            << "s\n"
            << std::endl;
  auto TextEncoderOutPos = ptrPos->memRef3D;
  auto TextEncoderOutNeg = ptrNeg->memRef3D;
  for (int i = 0; i < 2 * 77 * 1024 ; i ++ ){
    if (i < 1 * 77 * 1024)
      TextEncoderOut.getData()[i] = TextEncoderOutPos.getData()[i];
    else
      TextEncoderOut.getData()[i] = TextEncoderOutNeg.getData()[i % (1 * 77 * 1024)];
  }
  std::ofstream outFileTextEncoder("../../examples/BuddyStableDiffusion/output_text_encoder.txt");
  if (outFileTextEncoder.is_open()) {
      for (int i = 0 ; i < 2 * 77 * 1024 ; i ++ ) {
          outFileTextEncoder << TextEncoderOut.getData()[i] << std::endl;
      }
      outFileTextEncoder.close();
  } else {
      std::cerr << "Unable to open file" << std::endl;
  }

  fill_random_normal(latents, 1 * 4 * 96 * 96, 12345);

  printLogLabel();
  std::cout << "Start denoising..." << std::endl;

  for (int i = 1; i <= timesteps ; i ++){
    MemRef<float, 4> noise({2, 4, 96, 96});
    for (int j = 0 ; j < 2 * 4 * 96 * 96 ; j ++ )
      noise.getData()[j] = latents.getData()[j % (1 * 4 * 96 * 96)];

    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    _mlir_ciface_forward_unet(&resultUnet, &arg0_unet, &noise, &timestep, &TextEncoderOut);
    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime = inferenceEnd - inferenceStart;
    printIterInfo(i, inferenceTime.count() / 1000);


    std::ofstream outFileUnet("../../examples/BuddyStableDiffusion/output_unet.txt");
    if (outFileUnet.is_open()) {
        for (int j = 0 ; j < 1 * 4 * 96 * 96 ; j ++ ) {
            outFileUnet << resultUnet.getData()[j] << std::endl;
        }
        outFileUnet.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }

    MemRef<float, 4> pred_noise({1, 4, 96, 96});
    for (int j = 0 ; j < 1 * 4 * 96 * 96; j ++){
      pred_noise.getData()[i] = resultUnet.getData()[i] + 7.5 * (resultUnet.getData()[i + 1 * 4 * 96 * 96] - resultUnet.getData()[i]);
    }

    //There is a scheduler that still needs to be implemented.
    // latents = schedulerStep(pred_noise, timestep, latents)

  }

  for (int i = 0 ; i < 1 * 4 * 96 * 96 ; i ++){
    latents.getData()[i] = latents.getData()[i] / 0.18215;
  }

  
  std::cout << std::endl;
  printLogLabel();
  std::cout << "Start decoding..." << std::endl;
  const auto decodeStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_vae(&resultVae, &arg0_vae, &latents);
  const auto decodeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> decodeTime = decodeEnd - decodeStart;
  printLogLabel();
  std::cout << "Decode time: " << (double)(decodeTime.count()) / 1000
            << "s\n"
            << std::endl;

  std::ofstream outFileVae("../../examples/BuddyStableDiffusion/output_vae.txt");
  if (outFileVae.is_open()) {
      for (int i = 0 ; i < 1 * 3 * 768 * 768 ; i ++ ) {
          outFileVae << resultVae.getData()[i] << std::endl;
      }
      outFileVae.close();
  } else {
      std::cerr << "Unable to open file" << std::endl;
  }

  for (int i = 0 ; i < 1 * 3 * 768 * 768 ; i ++ ){
    resultVae.getData()[i] = (resultVae.getData()[i]  + 1) / 2;
    //clamp(0, 1)
    if (resultVae.getData()[i] < 0)
      resultVae.getData()[i] = 0;
    if (resultVae.getData()[i] > 1)
      resultVae.getData()[i] = 1;
    resultVae.getData()[i] = resultVae.getData()[i] * 255;
  }

  // The conversion of data to the image part still needs to be implemented.

  return 0;
}