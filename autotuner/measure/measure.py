"""Specifying how to measure the generated code"""
import enum
import logging
import multiprocessing
from collections import namedtuple


class MeasureInput(namedtuple("MeasureInput", ["target", "task", "config"])):
    """Stores all the necessary inputs for a measurement."""


class MeasureResult(namedtuple("MeasureResult", ["costs", "error_no", "all_cost", "timestamp"])):
    """Stores all the results of a measurement"""

    def __repr__(self):
        error_no_str = (
            str(MeasureErrorNo(self.error_no))
            if isinstance(self.error_no, (MeasureErrorNo, int))
            else str(self.error_no)
        )
        return (
            f"{self.__class__.__name__}(costs={self.costs!r}, error_no={error_no_str}, "
            f"all_cost={self.all_cost}, timestamp={self.timestamp!r})"
        )


class MeasureErrorNo(enum.IntEnum):
    """Error type for MeasureResult"""

    NO_ERROR = 0  # no error
    INSTANTIATION_ERROR = 1  # actively detected error in instantiating a template with a config
    COMPILE_ERROR = 3  # error when compiling code on device (e.g. OpenCL JIT on the device)
    RUNTIME_ERROR = 4  # error when run program on device
    WRONG_ANSWER = 5  # answer is wrong when compared to a golden output
    BUILD_TIMEOUT = 6  # timeout during compilation
    RUN_TIMEOUT = 7  # timeout during run
    UNKNOWN_ERROR = 8  # unknown error


class Builder(object):
    """Builder that builds programs in tuning"""

    def __init__(self, timeout=10, n_parallel=None, build_kwargs=None):
        self.timeout = timeout
        self.n_parallel = n_parallel or multiprocessing.cpu_count()
        self.user_build_kwargs = build_kwargs if build_kwargs is not None else {}
        self.runner_build_kwargs = None
        self.task = None

    def set_task(self, task, build_kwargs=None):
        """Initialize for a new tuning task"""
        self.task = task
        self.build_kwargs = dict(build_kwargs.items()) if build_kwargs is not None else {}
        if any(k in self.build_kwargs for k in self.user_build_kwargs):
            logging.warn(
                "Overriding these runner-supplied kwargs with user-supplied:\n%s",
                "\n".join(
                    f" * {k}: from {build_kwargs[k]!r} to {self.user_build_kwargs[k]!r}"
                    for k in sorted([k for k in build_kwargs if k in self.user_build_kwargs])
                ),
            )
        for k, v in self.user_build_kwargs.items():
            self.build_kwargs[k] = v

    def build(self, measure_inputs, output_dir):
        """Build programs"""
        raise NotImplementedError()


class Runner(object):
    """Runner that runs and measures the time cost of a generated program in tuning"""

    def __init__(self, timeout=5, n_parallel=None):
        self.timeout = timeout
        self.n_parallel = n_parallel or multiprocessing.cpu_count()
        self.task = None

    def set_task(self, task):
        """Initialize for a new tuning task"""
        self.task = task

    def get_build_kwargs(self):
        """Get device specific build arguments (e.g. maximum shared memory size)"""
        raise NotImplementedError()

    def run(self, measure_inputs, build_results):
        """Run amd measure built programs"""
        raise NotImplementedError()


def measure_option(builder, runner):
    """
    Set options for measure. To measure a config, we will build it and run it.
    So we have to set options for these two steps.
    They have their own options on timeout, parallel, etc.

    Note
    ----
    To make measurement results accurate, you should pick the correct value for the argument
    `number` and `repeat` in Runner(). Some devices need a certain minimum running time to
    "warm up," such as GPUs that need time to reach a performance power state.
    Using `min_repeat_ms` can dynamically adjusts `number`, so it is recommended.
    The typical value for NVIDIA GPU is 150 ms.
    
    Here we will use Spike for Gemmini.
    """
    from .measure_methods import BuddyMLIRBuilder, LocalRunner
    
    if isinstance(builder, str):
        if builder == "default" or builder == "gemmini" or builder == "linalg":
            builder = BuddyMLIRBuilder(build_func=builder)
        else:
            raise ValueError("Invalid builder: " + builder)
        
    if isinstance(runner, str):
        if runner == "spike" or runner == "cpu":
            runner = LocalRunner(run_func=runner, timeout=30)
        else:
            raise ValueError("Invalid runner: " + runner)

    opt = {
        "builder": builder,
        "runner": runner,
    }

    return opt


def create_measure_batch(task, option, output_dir):
    """Get a standard measure_batch function.

    Returns
    -------
    measure_batch: callable
        a callback function to measure a batch of configs
    """
    builder = option["builder"]
    runner = option["runner"]

    runner.set_task(task)

    # feed device related information from runner to builder
    # (e.g. max shared memory for validity checking)
    builder.set_task(task)

    def measure_batch(measure_inputs):
        build_results = builder.build(measure_inputs, output_dir)
        results = runner.run(build_results)
        return results

    measure_batch.n_parallel = builder.n_parallel
    return measure_batch
