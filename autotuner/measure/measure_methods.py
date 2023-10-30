# ===- measure_methods.py ------------------------------------------------------
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
# Functions that run on executor for measurement.
# Responsible for building, recording the running time costs, and checking the
# correctness of the output.
#
# ===---------------------------------------------------------------------------

import os
import time
import tempfile
import logging
import traceback
import subprocess
import warnings
from collections import namedtuple
from random import getrandbits

from .measure import Builder, MeasureErrorNo, MeasureInput, MeasureResult, Runner
from task.space import InstantiationError
from contrib.popen_pool import PopenPoolExecutor

logger = logging.getLogger("autotuner")


class BuildResult(namedtuple("BuildResult", ("filename", "error", "time_cost"))):
    """
    Stores all the necessary inputs for a measurement.

    Parameters
    ----------
    filename : str
        The filename of generated file
    error : Exception
        The error happens during compilation.
    time_cost : float
        The time cost of building
    """


def gemmini_build(measure_input: MeasureInput, output_path: str):
    """# example: build cmd
    passes = ["-lower-gemmini"]
    opt_cmd = 'buddy-opt ' + filepath + ' {} | \\'.format(' '.join(passes))
    translate_cmd = f'buddy-translate --buddy-to-llvmir | \\'
    llc_cmd = f'buddy-llc -filetype=obj -mtriple=riscv64 \
            -mattr=+buddyext,+D -float-abi=hard \
            -o log.o | \\'
    riscv_cmd = f'riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out'
    """
    _, task, config = measure_input
    input_path = task.input_path
    gemmini_pass_entity = config._entity_map.get("gemmini passes")
    # TODO: we need running environment same as "buddy-mlir", also we need run `conda activate CHIPYARD_CONDA_ENV_NAME` to support extension for RISCV now.
    activate_cmd = "source ~/.zshrc && conda activate gemmini && "
    opt_cmd = (
        "buddy-opt "
        + input_path
        + " {} | ".format(" ".join(gemmini_pass_entity.pass_config))
    )
    translate_cmd = f"buddy-translate --buddy-to-llvmir | "
    llc_cmd = f"buddy-llc -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D -float-abi=hard -o log.o && "
    riscv_cmd = f"riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o " + output_path
    build_cmd = activate_cmd + opt_cmd + translate_cmd + llc_cmd + riscv_cmd
    proc = subprocess.Popen(
        build_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        executable=os.environ["SHELL"],
    )
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "buddy build error:\n"
        py_str = lambda x: x.decode("utf-8")
        msg += py_str(out)
        raise RuntimeError(msg)


# specify the output format for every `build` function.
gemmini_build.output_format = "out"


def linalg_build(measure_input: MeasureInput, output_path: str):
    _, task, config = measure_input
    input_path = task.input_path
    linalg_pass_entity = config._entity_map.get("linalg passes")
    # TODO: we need running environment same as "buddy-mlir", also we need run `conda activate CHIPYARD_CONDA_ENV_NAME` to support extension for RISCV now.
    opt_cmd = (
        "buddy-opt "
        + input_path
        + " {}".format(" ".join(linalg_pass_entity.pass_config))
    )
    build_cmd = opt_cmd + " -o " + output_path
    proc = subprocess.Popen(
        build_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        executable=os.environ["SHELL"],
    )
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "linalg build error:\n"
        py_str = lambda x: x.decode("utf-8")
        msg += py_str(out)
        raise RuntimeError(msg)


linalg_build.output_format = "mlir"


class BuddyMLIRBuilder(Builder):
    def __init__(
        self,
        timeout=10,
        n_parallel=None,
        build_kwargs=None,
        build_func="default",
        do_fork=False,
    ):
        super(BuddyMLIRBuilder, self).__init__(timeout, n_parallel, build_kwargs)

        if isinstance(build_func, str):
            if build_func == "default" or build_func == "gemmini":
                build_func = gemmini_build
            elif build_func == "linalg":
                build_func = linalg_build
            else:
                raise ValueError("Invalid build_func" + build_func)
        self.build_func = _WrappedBuildFunc(build_func)
        if not do_fork:
            assert n_parallel in (
                None,
                1,
            ), f"if do_fork=False, need n_parallel=None or 1; got {n_parallel}"
        self.executor = PopenPoolExecutor(
            max_workers=16,
            timeout=timeout,
        )

    def build(self, measure_inputs, output_dir):
        results = []

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for inp in measure_inputs[i : i + self.n_parallel]:
                # TODO: according to task.py/create()
                ret = self.executor.submit(
                    self.build_func, inp, output_dir, **self.build_kwargs
                )
                futures.append(ret)

            for future in futures:
                try:
                    res = future.result()
                    if res.error is not None:
                        assert len(res.error) == 2, (
                            f"BuildResult errors should be a 2-tuple, but it is a {len(res.error)}"
                            "-tuple. This should not happen!"
                        )
                        tb, exception = res.error
                        # instantiation error
                        if isinstance(exception, InstantiationError):
                            res = MeasureResult(
                                (tb, exception),
                                MeasureErrorNo.INSTANTIATION_ERROR,
                                res.time_cost,
                                time.time(),
                            )
                        else:
                            if "InstantiationError" in str(exception):
                                msg = str(exception)
                                try:
                                    msg = msg.split("\n")[-2].split(": ")[1]
                                except Exception:  # pylint: disable=broad-except
                                    pass
                                res = MeasureResult(
                                    (tb, InstantiationError(msg)),
                                    MeasureErrorNo.INSTANTIATION_ERROR,
                                    res.time_cost,
                                    time.time(),
                                )

                            else:
                                res = MeasureResult(
                                    (tb, res.error),
                                    MeasureErrorNo.COMPILE_HOST,
                                    res.time_cost,
                                    time.time(),
                                )
                except TimeoutError as ex:
                    tb = traceback.format_exc()
                    res = MeasureResult(
                        (tb, ex),
                        MeasureErrorNo.BUILD_TIMEOUT,
                        self.timeout,
                        time.time(),
                    )
                except ChildProcessError as ex:
                    tb = traceback.format_exc()
                    res = MeasureResult(
                        (tb, ex),
                        MeasureErrorNo.RUNTIME_DEVICE,
                        self.timeout,
                        time.time(),
                    )

                results.append(res)
        return results


def spike_run(build_res, repeat=1):
    """example:
    spike_cmd = f'spike --extension=gemmini pk a.out'
    """
    if isinstance(build_res, MeasureResult):
        return build_res

    tic = time.time()
    errno = MeasureErrorNo.NO_ERROR

    activate_cmd = "source ~/.zshrc && conda activate gemmini && "
    spike_cmd = f"spike --extension=gemmini pk " + build_res.filename
    run_cmd = activate_cmd + spike_cmd
    proc = subprocess.Popen(
        run_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        executable=os.environ["SHELL"],
    )
    (out, _) = proc.communicate()

    tstamp = time.time()
    run_cost = tstamp - tic

    if proc.returncode != 0:
        # TODO: If we get error, should we throw exception or return error_code.
        # msg = "spike run error:\n"
        # py_str = lambda x: x.decode("utf-8")
        # msg += py_str(out)
        # raise RuntimeError(msg)
        return MeasureResult(
            run_cost,
            MeasureErrorNo.RUNTIME_ERROR,
            run_cost + build_res.time_cost,
            tstamp,
        )
    return MeasureResult(run_cost, errno, run_cost + build_res.time_cost, tstamp)


def cpu_run(build_res, repeat=1):
    if isinstance(build_res, MeasureResult):
        return build_res

    tic = time.time()
    errno = MeasureErrorNo.NO_ERROR

    PROJECT_DIR = os.getcwd()
    # TODO: local path
    MLIR_RUNNER_UTILS = PROJECT_DIR + "/lib/libmlir_runner_utils.so"
    MLIR_C_RUNNER_UTILS = PROJECT_DIR + "/lib/libmlir_c_runner_utils.so"

    # TODO: path of mlir-cpu-runner
    run_cmd = (
        f"./mlir-cpu-runner -O0 -e main -entry-point-result=void"
        + " -shared-libs="
        + MLIR_RUNNER_UTILS
        + " -shared-libs="
        + MLIR_C_RUNNER_UTILS
        + " "
        + build_res.filename
    )
    proc = subprocess.Popen(
        run_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        executable=os.environ["SHELL"],
    )
    (out, _) = proc.communicate()

    tstamp = time.time()
    run_cost = tstamp - tic

    if proc.returncode != 0:
        # TODO: If we get error, should we throw exception or return error_code.
        # msg = "spike run error:\n"
        # py_str = lambda x: x.decode("utf-8")
        # msg += py_str(out)
        # raise RuntimeError(msg)
        return MeasureResult(
            run_cost,
            MeasureErrorNo.RUNTIME_ERROR,
            run_cost + build_res.time_cost,
            tstamp,
        )
    return MeasureResult(run_cost, errno, run_cost + build_res.time_cost, tstamp)


class LocalRunner(Runner):
    def __init__(
        self,
        timeout=10,
        repeat=3,
        run_func=None,
        n_parallel=None,
    ):
        super(LocalRunner, self).__init__(timeout, n_parallel)

        self.timeout = timeout
        self.repeat = repeat
        if isinstance(run_func, str):
            if run_func == "default" or run_func == "cpu":
                self.run_func = cpu_run
            elif run_func == "spike":
                self.run_func = spike_run
            else:
                raise ValueError("Invalid run_func" + run_func)
        else:
            raise ValueError("Invalid run_func")
        self.executor = PopenPoolExecutor(
            timeout=timeout * (self.n_parallel + 1),
        )

    def run(self, build_results):
        results = []

        for i in range(0, len(build_results), self.n_parallel):
            futures = []
            for build_res in build_results[i : i + self.n_parallel]:
                ret = self.executor.submit(
                    self.run_func,
                    build_res,
                    self.repeat,
                )
                futures.append(ret)

            for future in futures:
                # TODO: according to run() function, especially for error handling.
                try:
                    res = future.result()
                    results.append(res)
                except Exception as ex:  # pylint: disable=broad-except
                    tb = traceback.format_exc()
                    results.append(
                        MeasureResult(
                            (tb, ex),
                            MeasureErrorNo.RUN_TIMEOUT,
                            self.timeout,
                            time.time(),
                        )
                    )

        return results


class _WrappedBuildFunc:
    """
    Wrap build_func to a function that can be used in measure.

    Note: this is a class instead of a closure so that it can be pickled when
    using multiprocessing.

    Parameters
    ----------
    build_func : The compilation function
        We expect fcompile to contain an attr "output_format".

    Returns
    -------
    wrapped_build_func : callable
        The wrapped build function
    """

    def __init__(self, build_func):
        if not hasattr(build_func, "output_format"):
            raise AttributeError(
                "Expect build_func to have the attribute output_format."
            )
        self.build_func = build_func

    def __call__(self, measure_input, output_path, **kwargs):
        """Wrapped build func."""
        tic = time.time()
        _, task, config = measure_input
        try:
            output_path = os.path.join(
                output_path,
                f"{task.name}_build_{getrandbits(64):0x}.{self.build_func.output_format}",
            )

            self.build_func(measure_input, output_path, **kwargs)

        except Exception as e:  # pylint: disable=broad-except
            tb = traceback.format_exc()
            return BuildResult(None, (tb, e), time.time() - tic)
        return BuildResult(output_path, None, time.time() - tic)
