# Convolution Comparison

## Build Comparison Target

```
$ cd buddy-mlir/build
$ cmake -G Ninja .. -DBUDDY_CONV_OPT_EXAMPLES=ON
$ ninja opencv-filter2D
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

- ONNX Runtime

```
$ cd buddy-mlir/examples/conv-opt/comparison/
$ python3 gen-conv-models.py
$ python3 onnxruntime-conv2d.py
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
