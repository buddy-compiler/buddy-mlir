# ===- task.py -------------------------------------------------------------
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
# Definition of task function.
#
# Task can be constructed from tuple of func, args, and kwargs.
# func is a state-less function, or a string that
# registers the standard task.
#
# ===---------------------------------------------------------------------------

import functools

from .dispatcher import DispatchContext, ApplyConfig
from .space import ConfigSpace


def create(task_name, target, args: dict = None):
    """Create a tuning task and initialize its search space"""
    ret = Task(task_name, args, target)
    # init config space
    ret.config_space = ConfigSpace()
    ctx = ApplyConfig(ret.config_space)
    with ctx:
        if target == "gemmini":
            # TODO: The key here is determined by build function in measure.py.
            ret.config_space.define_gemmini("gemmini passes", args)
            # TODO: Consider hardware information.
            # ret.config_space.define_gemmini_hardware("gemmini HW")
        elif target == "linalg":
            ret.config_space.define_linalg("linalg passes", args)

    ret.target = target

    return ret


def get_config(target):
    """Get current config object

    Returns
    -------
    cfg: ConfigSpace or ConfigEntity
        The current config
    """
    return DispatchContext.current.query(target, None)


def path_to_name(path):
    return path.split("/")[-1].split(".")[0]


class Task(object):
    """A Tunable Task"""

    def __init__(self, input, args, target):
        self.input_path = input
        self.name = path_to_name(input)
        self.args = args
        self.flop = 100.0  # we can assume the flop of a task is 100.0, without considering the actual value.
        # self.args = parse_args(args)
        self.config_space = None
        self.hardware_space = None
        self.target = target

    def __getstate__(self):
        return {
            "input_path": self.input_path,
            "name": self.name,
            "args": self.args,
            "config_space": self.config_space,
            "hardware_space": self.hardware_space,
            "target": self.target,
        }

    def __setstate__(self, state):
        self.input_path = state["input_path"]
        self.name = state["name"]
        self.args = state["args"]
        self.config_space = state["config_space"]
        self.hardware_space = state["hardware_space"]
        self.target = state["target"]

    def __repr__(self):
        return "Task(input_path=%s, args=%s, target=%s)" % (
            self.input_path,
            self.args,
            self.target,
        )
