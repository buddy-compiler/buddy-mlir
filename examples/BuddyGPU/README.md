# Buddy GPU Example
This example demonstrates how to use the Buddy GPU to run a simple single-kernel program.

## Matmul
The example program is a simple matrix multiplication kernel. The linalg definition is in the `matmul.mlir` file. 
A transform sequence is in `transform.mlir` to optimize this kernel and prepare it for execution on the GPU.
The `matmul-cubin.mlir` provides a lowered file, in case the pipeline is not working.

Run the following command to compile and run the program:
```
  make buddy-gpu-matmul
  python run-module-gpu.py --source matmul.mlir --target matmul-cubin.mlir --llvm_dir ../../llvm
```

The result should be:
```
[[502.9141  499.7761  511.35623 ... 500.9083  505.25574 511.03818]
 [499.57034 494.8066  506.427   ... 492.7868  497.22513 509.95612]
 [511.2017  516.017   513.631   ... 515.5991  515.6389  521.8318 ]
 ...
 [496.2721  496.3155  506.08054 ... 502.36798 505.94202 516.3577 ]
 [512.06866 505.80127 518.81934 ... 510.64966 510.10333 531.85364]
 [501.23514 500.17123 505.71808 ... 496.4447  500.5735  514.4204 ]]
[[503.26013 500.11093 511.70193 ... 501.24622 505.60373 511.38376]
 [499.89877 495.13043 506.762   ... 493.1151  497.5555  510.29483]
 [511.54883 516.35547 513.9717  ... 515.944   515.9865  522.1828 ]
 ...
 [496.59937 496.63785 506.41483 ... 502.70337 506.27927 516.6994 ]
 [512.4154  506.1411  519.17175 ... 510.9929  510.45322 532.2152 ]
 [501.57388 500.5093  506.06213 ... 496.7807  500.91638 514.77124]]
MLIR equal to NumPy? True
```

As the tensorcore doesn't support fp32 computation, the operands are converted to tf32, hence the result is not exactly the same as the PyTorch result. 

### Profiling
You need to install nsight compute first.
```
ncu -o profile-result --set full python run-module-gpu.py --source matmul.mlir --target matmul-cubin.mlir --llvm_dir ../../llvm
```