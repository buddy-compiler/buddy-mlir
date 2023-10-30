import sys
import logging

from task.task import create
from measure import measure_option
from tuner import RandomTuner, XGBTuner, log_to_file

task = create(
    "./test/linalg_matmul.mlir",
    "linalg",
    {
        "matmul-optimize": {
            "vec-size": [32, 64, 128, 256, 512, 1024],
            "kernel-m": [2, 4, 8, 16, 32],
            "kernel-n": [2, 4, 8, 16, 32],
        },
        "convert-linalg-to-loops": {},
        "expand-strided-metadata": {},
        "lower-affine": {},
        "convert-scf-to-cf": {},
        "convert-vector-to-llvm": {},
        "finalize-memref-to-llvm": {},
        "convert-arith-to-llvm": {},
        "convert-func-to-llvm": {},
        "reconcile-unrealized-casts": {},
    },
)


print(task.config_space)

# logging config (for printing tuning log to the screen)
logging.getLogger("autotuner").setLevel(logging.DEBUG)
logging.getLogger("autotuner").addHandler(logging.StreamHandler(sys.stdout))

# model_based_tuner
tuner = XGBTuner(task)
tuner.tune(
    n_trial=64,
    measure_option=measure_option(builder="linalg", runner="cpu"),
    callbacks=[log_to_file("matmul.json")],
)
