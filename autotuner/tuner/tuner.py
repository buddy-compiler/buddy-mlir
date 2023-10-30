# ===- tuner.py -------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Base class of tuner.
#
# ===---------------------------------------------------------------------------

import logging
import tempfile
import os
import numpy as np

from measure import MeasureInput, create_measure_batch
from task import Task

logger = logging.getLogger("autotuner")


class Tuner(object):
    """Base class for tuners"""

    def __init__(self, task: Task, **kwargs):
        self.param = kwargs
        self.recorder = None

        self.task = task
        self.hardware_space = self.task.hardware_space
        self.space = self.task.config_space

        # keep the current best
        self.best_config = None
        self.best_cost = 60 * 60 * 24
        self.best_measure_pair = None
        self.best_iter = 0
        self.error_ct_threshold = 150

        # time to leave
        self.ttl = None
        self.n_trial = None
        self.early_stopping = None
        self.output_dir = None

    def has_next(self):
        """Whether has next untried config in the space"""
        raise NotImplementedError()

    def next_batch(self, batch_size):
        """get the next batch of configs to be measure on real hardware"""
        raise NotImplementedError()

    def update(self, inputs, results):
        """Update parameters of the tuner according to measurement results"""

    def tune(
        self, n_trial, measure_option, early_stopping=None, callbacks=(), si_prefix="G"
    ):
        """Begin tuning"""
        # 指定输出目录
        output_dir = os.path.join(os.getcwd(), "results")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        measure_batch = create_measure_batch(self.task, measure_option, output_dir)
        n_parallel = getattr(measure_batch, "n_parallel", 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping
        old_level = logger.level
        i = error_ct = 0
        errors = []
        while i < n_trial:
            if not self.has_next():
                break

            configs = self.next_batch(min(n_parallel, n_trial - i))

            inputs = [
                MeasureInput(self.task.target, self.task, config) for config in configs
            ]
            results = measure_batch(inputs)

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    # TODO: 现在是根据 build 和 run 的 总时间开销 来作为评估标准
                    time_cost = res.all_cost
                    error_ct = 0
                    result_msg = res
                else:
                    time_cost = 60 * 60 * 24
                    error_ct += 1
                    tb, error = res.costs
                    if isinstance(error, str):
                        errors.append(tb + "\n" + error)
                    else:
                        errors.append(tb + "\n" + str(error))
                    result_msg = errors[-1]

                if time_cost < self.best_cost:
                    self.best_cost = time_cost
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

                logger.debug(
                    "No: %d\ttime cost: %2f\tresult: %s\t%s",
                    i + k + 1,
                    time_cost,
                    result_msg,
                    config,
                )

            i += len(results)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - i

            self.update(inputs, results)
            for callback in callbacks:
                callback(self, inputs, results)

            if i >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

            if error_ct > self.error_ct_threshold:
                logging.basicConfig()
                logger.warning(
                    "Too many errors happen in the tuning. Switching to debug mode."
                )
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

        if error_ct == i:
            _, f = tempfile.mkstemp(prefix="tuning_errors_", suffix=".log", text=True)
            with open(f, "w") as file:
                file.write("\n".join(errors))
            logging.warning(
                "Could not find any valid schedule for task %s. "
                "A file containing the errors has been written to %s.",
                self.task,
                f,
            )

        # log the best
        logger.debug(
            "\nBest No: %d\ttime cost: %2f\tbest config: %s",
            self.best_iter,
            self.best_cost,
            self.best_config,
        )

        del measure_batch

    def reset(self):
        """reset the status of tuner"""
        self.best_config = None
        self.best_cost = 60 * 60 * 24
        self.best_measure_pair = None

    def load_history(self, data_set, min_seed_records=500):
        """load history data for transfer learning"""
        raise NotImplementedError()

    def set_error_threshold(self, threshold):
        """Modify error counter threshold, which controls switch to debug mode"""
        self.error_ct_threshold = threshold
