# -*- Python -*-

import os

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "BUDDY-EXAMPLES"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

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

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "BuddyLeNet",
    "BuddyBert",
    "BuddyLlama",
    "BuddyGemma4",
    "BuddyWhisper",
    "BuddyMobileNetV3",
    "BuddyStableDiffusion",
    "BuddyDeepSeekR1",
    "BuddyQwen3",
    "BuddyTransformer",
    "BuddyYOLO26",
    "BuddyResNet18",
    "BuddyGPU",
    "BuddyOneDNN",
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

config.buddy_tools_dir = os.path.join(config.buddy_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

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

tool_dirs = [config.buddy_tools_dir, config.llvm_tools_dir]
tools = [
    "buddy-opt",
    "buddy-translate",
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
