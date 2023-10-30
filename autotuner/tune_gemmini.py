import sys
import logging

from task.task import create
from measure import measure_option
from tuner import RandomTuner, XGBTuner, log_to_file

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
        },
    },
)

# We search all the config by default.
# task = create(
#     "./test/matmul-os.mlir",
#     "gemmini",
# )

print(task.config_space)
# print(task.hardware_space)

# logging config (for printing tuning log to the screen)
logging.getLogger("autotuner").setLevel(logging.DEBUG)
logging.getLogger("autotuner").addHandler(logging.StreamHandler(sys.stdout))

# randome tuner
tuner = RandomTuner(task)
tuner.tune(
    n_trial=10,
    measure_option=measure_option(builder="default", runner="spike"),
    callbacks=[log_to_file("matmul.json")],
)
