import sys
import logging

from task.task import create
from measure import measure_option
from tuner import RandomTuner, XGBTuner, log_to_file

# 支持自己自定义需要搜索的 pass, 根据提供的参数生成搜索空间
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

# 如果搜索空间非常小（小于 1000)， 选择 index_based_tuner 即可。
# 如搜索空间在 10^9 级别，选择 model_based_tuner 可以更有效地探索并找到更好的配置。
tuner = XGBTuner(task)
tuner.tune(
    n_trial=64,
    measure_option=measure_option(builder="linalg", runner="cpu"),
    callbacks=[log_to_file("matmul.json")]
)
