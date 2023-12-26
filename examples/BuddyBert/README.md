# Buddy Compiler BERT Emotion Classification Example

## Introduction
This example shows how to use Buddy Compiler to compile a BERT model to MLIR code then run it.  The [model](bhadresh-savani/bert-base-uncased-emotion) is trained to classify the emotion of a sentence into one of the following classes: sadness, joy,  love, anger, fear, and surprise.


## How to run
1. Ensure that LLVM, Buddy Compiler and the Buddy Compiler python packages are installed properly. You can refer to [here](https://github.com/buddy-compiler/buddy-mlir) for more information and do a double check.

2. Set the `PYTHONPATH` environment variable.
```bash
$ export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}
```

3. Build and run the BERT example
```bash
$ cmake -G Ninja .. -DBUDDY_BERT_EXAMPLES=ON
$ ninja buddy-bert-run
$ cd bin
$ ./buddy-bert-run
```

4. Enjoy it!
