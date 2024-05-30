# Buddy Compiler WHISPER Example

## Introduction
This example shows how to use Buddy Compiler to compile a WHISPER model to MLIR code then run it.  The [model](openai/whisper-base) is a pre-trained model for automatic speech recognition (ASR) and speech translation.


## How to run
1. Ensure that LLVM, Buddy Compiler and the Buddy Compiler python packages are installed properly. You can refer to [here](https://github.com/buddy-compiler/buddy-mlir) for more information and do a double check.

2. Set the `PYTHONPATH` environment variable.

```bash
$ export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}
```

3. Set model and dataset environment variable.

```bash
$ export WHISPER_MODEL_PATH=/path-to-whisper-model/
$ export AUDIO_DATASET_PATH=/path-to-audio-dataset/

// For example:
$ export WHISPER_MODEL_PATH=/home/xxx/whisper-base
$ export AUDIO_DATASET_PATH=/home/xxx/librispeech_asr_dummy
```

4. Build and run the WHISPER example

```bash
$ cmake -G Ninja .. -DBUDDY_WHISPER_EXAMPLES=ON
$ ninja buddy-whisper-run
$ cd bin
$ ./buddy-whisper-run
```

4. Enjoy it!
