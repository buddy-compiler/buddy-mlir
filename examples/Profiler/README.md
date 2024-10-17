# Buddy Compiler MobileNetV3 Example

## MobileNetV3 Model Inference

0. Activate your python environment.

1. Build buddy-mlir

2. Same config as your example

3.Set BUDDY_PROFILE_EXAMPLES=ON

4.Build profiler
```bash
$ cmake -G Ninja .. -DBUDDY_PROFILE_EXAMPLES=ON
$ ninja  buddy-profile
$ cd bin
$ cp buddy-profile path/to/your/profiler_dir
$ cd path/to/your/profiler_dir
```

5. Set the `your example path` environment variable.

```bash
$ export MOBILENETV3_EXAMPLE_PATH= path/to/your/example
```

```bash
$ ./buddy-profile
```

