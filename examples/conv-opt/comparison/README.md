# Convolution Comparison

## Build Comparison Target

```
$ cd buddy-mlir/build
$ cmake -G Ninja .. \
    -DBUDDY_CONV_OPT_EXAMPLES=ON \
    -DBoost_INCLUDE_DIR=</path/to/boost>
$ ninja opencv-filter2D
$ ninja boost_gil_conv2d
```

## Run Comparison

Please make sure OpenCV, TensorFlow, PyTorch are installed in your environment.

- OpenCV

```
$ cd buddy-mlir/build/bin
$ ./opencv-filter2D ../../examples/conv-opt/images/YuTu.png opencv-comparison.png
```

- TensorFlow

```
$ cd buddy-mlir/examples/conv-opt/comparison/
$ python3 tf-conv2d.py
```

- PyTorch

```
$ cd buddy-mlir/examples/conv-opt/comparison/
$ python3 pytorch-conv2d.py
```

- TVM
> **_NOTE:_**  In order to run tvm-conv2d.py, one has to manually build tvm with their system. For building reference please check TVM official installation guide: https://tvm.apache.org/docs/install/from_source.html.

```
$ cd buddy-mlir/examples/conv-opt/comparison/
$ python3 tvm-conv2d.py --size 3 --target 'llvm -mcpu=skylake-avx512'
$ # n kernel size, default 3, alternative options: 3, 5, 7, 9 
$ # target, default 'llvm -mcpu=skylake-avx512', alternative options: 'cuda'
```

- ONNX Runtime

```
$ cd buddy-mlir/examples/conv-opt/comparison/
$ python3 gen-conv-models.py
$ python3 onnxruntime-conv2d.py
```

- Boost GIL
> **_NOTE:_**  In order to run boost_gil_conv2d.cpp, one has to install the Boost distribution and 
> its related dependencies. Instructions regarding Boost installation can be found here : https://www.boost.org/doc/libs/1_77_0/more/getting_started/unix-variants.html
> More information regarding external dependencies can be found here : https://github.com/boostorg/gil/blob/develop/CONTRIBUTING.md#install-dependencies

```
$ cd buddy-mlir/build/bin
$ ./boost_gil_conv2d ../../examples/conv-opt/images/gil_sample.png GIL-comparison.png
```

## Difference and Performance Comparison

We proposed a coefficients broadcasting algorithm with strip mining strategy (CB-SM) to accelerate 2D convolutions, 
which is the key operation in deep learning, image/video processing, and computer vision.

The differences compared with state-of-the-art approaches are as follows:

- OpenCV

OpenCV uses the DFT-based algorithm for the filter2D function.
Unlike our method using FMA, the DFT-based algorithm only needs multiplication, 
but the cost is to perform a forward or inverse DFT.

- DL frameworks

TensorFlow and PyTorch compute Conv2D operation with GEMM. 
Thus, input and output transformation are necessary, which causes additional overhead. 
Our approach vectorizes the convolution directly without any transformation.
