# Benchmark
To help researchers test convolution operations more efficiently, we provide a benchmark to test performance of MLIR convolution operations. It can calculate the running time of the program and print the GFLOPS value. 
## Run test
```
cd benchmark
make
```
This will execute the default test, which means output size from 32 to 4096 as geometric progression and convolution kernel size from 3 to 11 with the stride of 2. The size of each vector slice is 32 by default.

And the result should be displayed as follows example:
```
conv-3-32
2.751225 GFLOPS
0.617487 GFLOPS

conv-3-64
5.286114 GFLOPS
0.616933 GFLOPS
...

```
The first line of each group of data indicates the size of the convolution kernel and the size of the output figure shown in `conv-<kernel size>-<output size>`. The second line is the running speed of our CB-SM approach, and the third line is the running speed of default scalar approach(nested loops).

You can generate and execute different tests by custom parameters, for example:

```
make STRIP=64 OUTPUT_min=128 FILTER_max=5
```

the available parameter settings are as follows:
```
STRIP: the size of each vector slice.

FILTER_min: the MINIMUM size of the convolution kernel.

FILTER_max: the MAXIMUM size of the convolution kernel.

FILTER_step: the step size of the convolution kernel size change.

OUTPUT_min: the MINIMUM size of the output matrix.

OUTPUT_max: the MAXIMUM size of the output matrix.

OUTPUT_step: the step size of the output matrix size change.
```

## Generate figure
In order to display the experimental results more intuitively, we provide a python program to execute the test and generate the corresponding figure. You can use the tool with the following configuration with python3 and pip3 installed.

```
pip3 install -r requirements.txt
python3 figure.py
```
This will use the default parameters to perform the test, and the three test results will be averaged for drawing(also save in `benchmark/data_<your parameter>.csv`). The generated figure is saved in `benchmark/figure_<your parameter>.png`.

Similarly, you can still use custom parameters. 

```
python3 figure.py -PASS 5 -STRIP 64 -OUTPUT_min 128 -FILTER_max 5
```

In addition to the parameters in the makefile, the following parameters can also be used:
```
PASS: the number of test passes used to average the results.

CSV: the path to a csv file. If this parameter is set, the figure will be generated directly using the given csv file, without running the test.

CONV_OPT: the path of the conv-opt (MLIR convolution optimizer with CB-SM approach).

MLIR_OPT: the path of the mlir-opt (MLIR modular optimizer driver).
```
