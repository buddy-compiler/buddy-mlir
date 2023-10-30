# BUDDY CODESIGN

## Getting Started

### Prerequisites

#### LLVM/MLIR Dependencies

Before building, please make sure [the dependencies](https://llvm.org/docs/GettingStarted.html#requirements) are available on your machine.

Please follow steps here, https://github.com/buddy-compiler/buddy-mlir

#### GEMMINI Dependencies

Before beginning, install the [Chipyard dependencies](https://chipyard.readthedocs.io/en/latest/Chipyard-Basics/Initial-Repo-Setup.html#default-requirements-installation).

Please follow steps here, https://github.com/ucb-bar/gemmini

### Installing

```shell
git clone git@github.com:buddy-compiler/buddy-codesign.git
```

### Examples

`tune_gemmini.py` , we represent an example of tuning gemmini programs.

```python
task = create(
    "./test/conv_2d_nhwc_hwcf_f32.mlir", 
    "gemmini",
    {
        "convert-linalg-to-gemmini": {
            "acc_t": ["f32"],
        },
        "convert-linalg-to-loops": {},
        "lower-gemmini": {
            "dim": [4, 8, 16],
            "acc_t": ["f32"],
            "elem_t": ["f32"],
        }
    }
)

print(task.config_space)
```

We will get output like below.

```shell
ConfigSpace (len=1, range_length=1, space_map=
   0 gemmini passes: Gemmini PASS Space: (
     ['-convert-linalg-to-gemmini="acc_t=f32"', '-convert-linalg-to-loops', '-lower-gemmini="dim=4 acc_t=f32 elem_t=f32"']
     ['-convert-linalg-to-gemmini="acc_t=f32"', '-convert-linalg-to-loops', '-lower-gemmini="dim=8 acc_t=f32 elem_t=f32"']
     ['-convert-linalg-to-gemmini="acc_t=f32"', '-convert-linalg-to-loops', '-lower-gemmini="dim=16 acc_t=f32 elem_t=f32"']
   )
)
```

when we use Random tuner, we will get output file 'matmul.json'

```python
tuner = RandomTuner(task)
tuner.tune(
    n_trial=10,
    measure_option=measure_option(builder="default", runner="spike"),
    callbacks=[log_to_file("matmul.json")]
)
```

