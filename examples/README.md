# Examples

The purpose of the examples is to give users a better understanding of how to use the passes and the interfaces in buddy-mlir. Currently, we provide three types of examples.

- IR level conversion and transformation examples.
- Domain-specific application level examples.
- Testing and demonstrating examples.

## IR Level Examples

The IR level examples show how to use the passes in upstream MLIR and buddy-mlir, some of these examples come from the MLIR integration test. Most cases can be run directly with the MLIR JIT engine `mlir-cpu-runner`. The lowering pipeline and the toolchain configuration are specified in makefile target. Before you start trying the IR level examples, please make sure you have completed the [get started part](../README.md). 

Then you can find a Dialect that you are interested in, and you can go to the corresponding directory to find the target you want to run. The naming convention for target is `<Dialect Name>-<Operation Name>-<Target Type>`. For most examples, we provide the 
following targets:

- Lower Target (`<Dialect Name>-<Operation Name>-lower`): the lower target is designed to show the lowering pipeline. You can also remove some of these passes to see the different generated MLIR code. The name of the output file is `log.mlir`.

- Translate Target (`<Dialect Name>-<Operation Name>-translate`): the translate target is designed to show the LLVM IR that generate from MLIR. The name of the output file is `log.ll`.

- Executable Target (`<Dialect Name>-<Operation Name>-run`): the executable target uses MLIR JIT engine to showe the result of the cases.

You can run these targets with the `make` command. Take an example with the `memref.dim` operation, you can run the following command:

```
$ cd buddy-mlir/examples/MLIRMemRef
$ make memref-dim-lower
$ make memref-dim-translate
$ make memref-dim-run
```

Feel free to send PR when you write a new example.

## Domain-specific Application Level Examples

The domain-specific application level examples show how to use MLIR in domain applications and how to use the interfaces we provide. Currently, we provide the following examples. Before you start trying the application level examples, please make sure you have completed the [get started part](../README.md).

### Convolution Vectorization Examples

So far, we provide the 2D convolution vectorization pass `conv-vectorization`. The pass implements the Coefficients Broadcasting algorithm with Strip Mining strategy, and the strip mining size is configurable. Take the size of 256 as an example, you can use the tool with the following configuration.

```
$ buddy-opt <input> -conv-vectorization="strip-mining=256"
```

- Conversion example

We provide a function with `linalg.conv_2d` operation. You can use the following commands to print the conversion result.

```
$ cd buddy-mlir/build/bin
$ ./buddy-opt ../../examples/ConvOpt/conv2d.mlir -conv-vectorization="strip-mining=256"
```

- Edge detection example

We also provide an edge detection example to show the optimization. 
The `conv-vectorization` pass is responsible for lowering the `linalg.conv_2d` with our algorithm. And then we use `mlir-translate` and `llc` tools to generate the object file. At last, we call the MLIR convolution function in a C++ program.

Please make sure OpenCV is installed to play around.

This example can also show the "magic" of AutoConfig mechanism that can help you specify the `strip mining size`, `ISA SIMD/Vector extension`, and `target triple`. You only need to enable the `BUDDY_EXAMPLES` option and don't worry about the toolchain configuration.

```
$ cd buddy-mlir/build
$ cmake -G Ninja .. -DBUDDY_EXAMPLES=ON -DBUDDY_ENABLE_OPENCV=ON
$ ninja edge-detection
```

Of course, you can also use your own configuration assigning values `-DBUDDY_CONV_OPT_STRIP_MINING` (e.g. 64) and `-DBUDDY_OPT_ATTR` (e.g. avx2).

We provide an image at `buddy-mlir/examples/ConvOpt/images/YuTu.png`, which is the robotic lunar rover that formed part of the Chinese Chang'e 3 mission.

You can detect the edge of the image with `edge-detection`.

```
$ cd bin
$ ./edge-detection ../../examples/ConvOpt/images/YuTu.png result.png
```

We also provide the performance comparison between our `buddy-opt` tool and other state-of-the-art approaches. 
For more details, please see [the benchamrk in the buddy-benchmark repo](https://github.com/buddy-compiler/buddy-benchmark#image-processing-benchmark).


### Digital Image Processing Examples


```
$ buddy-opt <input> -lower-dip="DIP-strip-mining=${BUDDY_DIP_OPT_STRIP_MINING}"
```

- Conversion example:

```
$ cd buddy-mlir/build/bin
$ ./buddy-opt ../../examples/DIPDialect/corr2d.mlir --lower-dip="DIP-strip-mining=${BUDDY_DIP_OPT_STRIP_MINING}"
```

- Edge detection example:

Build and run the example.

*Note: Please make sure OpenCV is installed to play around.*

This example can also show the "magic" of AutoConfig mechanism that can help you specify the `strip mining size`, `ISA SIMD/Vector extension`, and `target triple`. You only need to enable the `BUDDY_EXAMPLES` option and don't worry about the toolchain configuration.

```
$ cd buddy-mlir/build
$ cmake -G Ninja .. -DBUDDY_EXAMPLES=ON -DBUDDY_ENABLE_OPENCV=ON
$ ninja correlation2D
$ cd bin
$ ./correlation2D ../../examples/ConvOpt/images/YuTu.png result-dip-corr2d-replicate-padding.png result-dip-corr2d-constant-padding.png
```

Of course, you can also use your own configuration assigning values `-DBUDDY_DIP_OPT_STRIP_MINING` (e.g. 64) and `-DBUDDY_OPT_ATTR` (e.g. avx2).

*Note: Maximum allowed value of `BUDDY_DIP_OPT_STRIP_MINING` for producing correct result is equal to image width.*

 - Rotation example:
```
$ cd buddy-mlir/build
$ cmake -G Ninja .. -DBUDDY_EXAMPLES=ON -DBUDDY_ENABLE_OPENCV=ON
$ ninja rotation2D
$ cd bin
$ ./rotation2D ../../examples/ConvOpt/images/YuTu.png result-dip-rotate.png
```

 - Resize example:
```
$ cd buddy-mlir/build
$ cmake -G Ninja .. -DBUDDY_EXAMPLES=ON -DBUDDY_ENABLE_OPENCV=ON
$ ninja resize2D
$ cd bin
$ ./resize2D ../../examples/ConvOpt/images/YuTu.png result-dip-resize.png
```

We also provide the performance comparison between our `buddy-opt` tool and other state-of-the-art approaches. 
For more details, please see [the benchamrk in the buddy-benchmark repo](https://github.com/buddy-compiler/buddy-benchmark#image-processing-benchmark).

### Digital Audio Processing Examples

- Fir Lowpass example:

Build and run the example.

*Note: No external library required.*

This example shows how FIR is acheived using our library and MLIR-based convolution method. It uses basic lowering pipelines so performance is poor by default. Different windows could be applied and cutoff frequency could be altered. The result could be saved to any specified destination available, or saved to current working directory by default. Notice that you must specify input file first than the output destination could be specified.

```
$ cd buddy-mlir/build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja firLowpass
$ cd bin
$ ./firLowpass [input_file] [output_dest]
```
Specify nothing to process default NASA audio.

## Testing and Demonstrating Examples

```
$ buddy-opt <input> -lower-bud
```

Example:

```
$ cd buddy-mlir/build/bin
$ ./buddy-opt ../../examples/BudDialect/TestConstant.mlir --lower-bud
```

## DSL Examples

We use Antlr as the frontend framework.

1. Build LLVM/MLIR with RTTI and EH enabled.

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON
$ ninja check-mlir
```

2. Build buddy-mlir with `BUDDY_DSL_EXAMPLES` enabled.

```
$ cd buddy-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_DSL_EXAMPLES=ON
$ ninja
```

3. Run the example.

For example, you can run the example of constant printing through makefile target `buddy-toy-constant-run`:

```
$ cd buddy-mlir/examples/ToyDSL
$ make buddy-toy-constant-run
```

All Toy DSL executable targets can be found in the `buddy-mlir/examples/ToyDSL/makefile`.
