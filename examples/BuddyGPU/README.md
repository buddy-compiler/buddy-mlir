# Buddy GPU Example
This example demonstrates how to use the Buddy GPU to run a simple single-kernel program.

## Matmul
The example program is a simple matrix multiplication kernel. The linalg definition is in the `matmul.mlir` file. 
A transform sequence is in `transform.mlir` to optimize this kernel and prepare it for execution on the GPU.
The `matmul-cubin.mlir` provides a lowered file, in case the pipeline is not working.

Run the following command to compile and run the program:
```
  make bud-matmul-gpu-lower
  python3 run-module-gpu.py
```

The result should be:
```
 [[517.8539  504.74646 529.67615 ... 525.0187  518.1935  512.0135 ]
 [508.46912 501.6754  517.8974  ... 532.3624  512.7449  514.65265]
 [501.0289  496.60242 515.3227  ... 520.70935 505.95404 507.82297]
 ...
 [506.9494  495.7268  515.83734 ... 518.41675 511.71024 502.028  ]
 [513.69556 507.59366 520.084   ... 526.7201  512.178   508.53937]
 [510.5107  498.9726  513.9182  ... 521.1717  520.8544  504.67075]]
Skipping numpy comparison
 [[518.2012  505.08414 530.03485 ... 525.36487 518.5387  512.3538 ]
 [508.80725 502.01288 518.2523  ... 532.7181  513.0874  514.9979 ]
 [501.36856 496.94287 515.67505 ... 521.05884 506.29358 508.16623]
 ...
 [507.29694 496.06473 516.19684 ... 518.7683  512.06036 502.3666 ]
 [514.04254 507.93488 520.4351  ... 527.0661  512.5204  508.8747 ]
 [510.85178 499.30905 514.269   ... 521.5194  521.20215 505.01044]]
MLIR equal to PyTorch? True
```

As the tensorcore doesn't support fp32 computation, the operands are converted to tf32, hence the result is not exactly the same as the PyTorch result. 

### Profiling
You need to install nsight compute first.
```
ncu -o profile-result --set full python run-module-gpu.py
```