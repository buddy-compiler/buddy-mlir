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
            # TODO: 这里的 key 和 measure 的 build 函数对应
            ret.config_space.define_gemmini("gemmini passes", args)
            # TODO: 暂时不考虑硬件信息, 主要是 apply 函数还未实现
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
    return path.split('/')[-1].split('.')[0]


class Task(object):
    """A Tunable Task"""

    def __init__(self, input, args, target):
        self.input_path = input
        self.name = path_to_name(input)
        self.args = args
        self.flop = 100.0 # we can assume the flop of a task is 100.0, without considering the actual value.
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