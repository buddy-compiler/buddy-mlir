import sys
import logging

from task.task import create
from measure import measure_option
from tuner import RandomTuner, XGBTuner, log_to_file

# 支持自己自定义需要搜索的 pass, 根据提供的参数生成搜索空间
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

# 默认搜索所有可能的配置
# task = create(
#     "./test/matmul-os.mlir", 
#     "gemmini",
# )

print(task.config_space)
# print(task.hardware_space)

# logging config (for printing tuning log to the screen)
logging.getLogger("autotuner").setLevel(logging.DEBUG)
logging.getLogger("autotuner").addHandler(logging.StreamHandler(sys.stdout))

# 如果搜索空间非常小（小于 1000)， 选择 index_based_tuner 即可。
# 如搜索空间在 10^9 级别，选择 model_based_tuner 可以更有效地探索并找到更好的配置。
tuner = RandomTuner(task)
tuner.tune(
    n_trial=10,
    measure_option=measure_option(builder="default", runner="spike"),
    callbacks=[log_to_file("matmul.json")]
)
