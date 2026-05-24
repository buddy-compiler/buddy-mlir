# ModelBench

`ModelBench` contains end-to-end model benchmarks.  The first benchmark is the
F32 DeepSeek-R1 path with one prefill call and one decode call.

The benchmark uses fake inputs and zero-backed fake parameters.  It does not
tokenize text, load real weights, copy KV cache from prefill to decode, or run a
generation loop.  It only measures one execution of each model entry point.

## Run

Activate the Python environment first:

```bash
conda activate <your env name>
```

Then build `buddy-mlir` by following the root `README.md`, including the
LLVM/MLIR build and the `buddy-mlir` build with Python packages enabled.  From
the `build/` directory, export the Python package path used by the importer:

```bash
export BUDDY_MLIR_BUILD_DIR=$PWD
export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

Then return to the repository root and run ModelBench:

```bash
cd ..
python3 benchmarks/ModelBench/run_model_bench.py deepseek-r1
```

The Python driver imports the model, compiles the generated MLIR to object
files, links the benchmark runner, and executes it.  Generated files are written
to `build/benchmarks/ModelBench/` by default.

Useful options:

```bash
# Reuse generated MLIR.
python3 benchmarks/ModelBench/run_model_bench.py deepseek-r1 --skip-import

# Reuse object files and only relink/run.
python3 benchmarks/ModelBench/run_model_bench.py deepseek-r1 --skip-import --skip-compile

# Run an already-built benchmark binary.
python3 benchmarks/ModelBench/run_model_bench.py deepseek-r1 --run-only
```

The importer follows `examples/BuddyDeepSeekR1/import-deepseek-r1.py`, so
`DEEPSEEKR1_MODEL_PATH` may point at a local HuggingFace model directory.  If it
is not set, the importer uses `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

The command line output separates build, run, and result sections:

```text
== ModelBench Result ==
model            precision         prefill    decode(avg) decode iters
------------------------------------------------------------------------
deepseek-r1      f32              10.775 s      40.698 ms            5

csv: build/benchmarks/ModelBench/deepseek-r1/results.csv
```

Use `--verbose-commands` to print full compiler commands during the build.
Set `NO_ANIMATION=1` to disable the interactive spinner.
