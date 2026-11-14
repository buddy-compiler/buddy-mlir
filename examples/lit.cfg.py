# -*- Python -*-
# ===- lit.cfg.py ----------------------------------------------------------
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
# Lit configuration for the buddy-mlir examples/ tree. Registers the BUDDY-
# EXAMPLES suite, MLIR (and optional Python) suffixes, tool substitutions for
# buddy-opt and related runners, and OpenMP runtime search paths. Keep this
# aligned with tests/lit.cfg.py when adding shared substitutions or features.
#
# ===---------------------------------------------------------------------------

import os

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "BUDDY-EXAMPLES"

config.test_format = lit.formats.ShTest(execute_external=False)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]
if config.buddy_mlir_enable_python_packages:
    config.suffixes.append(".py")

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.buddy_obj_root, "examples")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

openmp_runtime_dir = config.mlir_runner_utils_dir
openmp_runtime_candidates = [
    os.path.join(config.llvm_build_dir, "lib"),
    os.path.join(
        config.llvm_build_dir,
        "runtimes",
        "runtimes-bins",
        "openmp",
        "runtime",
        "src",
    ),
]
for candidate in openmp_runtime_candidates:
    if os.path.exists(
        os.path.join(candidate, "libomp" + config.llvm_shlib_ext)
    ):
        openmp_runtime_dir = candidate
        break
config.openmp_runtime_dir = openmp_runtime_dir
config.substitutions.append(("%openmp_runtime_dir", config.openmp_runtime_dir))

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "BOSCAMEFPGA",
    "BuddyLeNet",
    "BuddyBert",
    "BuddyLlama",
    "BuddyLlama31-8B",
    "BuddyGemma4",
    "BuddyWhisper",
    "BuddyMobileNetV3",
    "BuddyStableDiffusion",
    "BuddyDeepSeekR1",
    "BuddyQwen3",
    "BuddyTensorParallel",
    "BuddyTransformer",
    "BuddyYOLO26",
    "BuddyResNet18",
    "BuddyGPU",
    "BuddyOneDNN",
    "BuddyTorq",
    "BuddyGraph",
    "ConvOpt",
    "DAPDialect",
    "DIPDialect",
    "DLModel",
    "FrontendGen",
    "MLIREmitC",
    "MLIRGPU",
    "MLIRPDL",
    "MLIRPython",
    "MLIRSCF",
    "MLIRSparseTensor",
    "MLIRTOSA",
    "MLIRTransform",
    "MLIRVectorGPU",
    "Pooling",
    "PyTorchTriton",
    "RISCVBuddyExt",
    "RVVDialect",
    "RVVExperiment",
    "ScheDialect",
    "SIMDExperiment",
    "ToyDSL",
    "VectorExpDialect",
    "log.mlir",
    "lit.cfg.py",
    "BuddyPython",
]

if "has_buckyball_external_dialects" not in config.available_features:
    config.excludes.append("BuckyballDialect")

config.buddy_tools_dir = os.path.join(config.buddy_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# So The execution engine in frontend can find out-of-tree llvm librarys
config.environment["LLVM_LIBS_DIR"] = config.mlir_runner_utils_dir

# Add the python path for both upstream MLIR and Buddy Compiler python packages.
if config.buddy_mlir_enable_python_packages:
    llvm_config.with_environment(
        "PYTHONPATH",
        [
            os.path.join(
                config.llvm_build_dir,
                "tools",
                "mlir",
                "python_packages",
                "mlir_core",
            ),
            config.buddy_python_packages_dir,
        ],
        append_path=True,
    )
    # PyTorch pulls in one OpenMP runtime; Buddy's ExecutionEngine also loads
    # libomp from the LLVM build (see frontend.py shared_libs). Two libomp
    # copies in one process trigger OMP Error #15; LLVM OpenMP allows continuing
    # when this is set (common when mixing PyTorch with MLIR JIT on e.g. RISC-V)
    llvm_config.with_environment("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Prevent CUDA context initialization in CPU-only Python tests.
    # When many workers run in parallel, each torch._dynamo import eagerly
    # initialises a CUDA context; this exhausts GPU memory and causes
    # spurious AcceleratorError / CUDA OOM failures.
    llvm_config.with_environment("CUDA_VISIBLE_DEVICES", "")

tool_dirs = [config.buddy_tools_dir, config.llvm_tools_dir]
tools = [
    "buddy-opt",
    "buddy-translate",
    "buddy-llc",
    "mlir-runner",
]
tools.extend(
    [
        ToolSubst(
            "%mlir_runner_utils_dir",
            config.mlir_runner_utils_dir,
            unresolved="ignore",
        ),
    ]
)

python_executable = config.python_executable
tools.extend(
    [
        ToolSubst("%PYTHON", python_executable, unresolved="ignore"),
    ]
)

llvm_config.add_tool_substitutions(tools, tool_dirs)
