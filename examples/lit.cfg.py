# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'BUDDY-EXAMPLES'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.buddy_obj_root, 'examples')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    'BuddyLeNet',
    'BuddyBert',
    'BuddyLlama',
    'BuddyBert',
    'BuddyResNet18',
    'ConvOpt',
    'DAPDialect',
    'DIPDialect',
    'DLModel',
    'FrontendGen',
    'MLIREmitC',
    'MLIRGPU',
    'MLIRPDL',
    'MLIRPython',
    'MLIRSCF',
    'MLIRSparseTensor',
    'MLIRTOSA',
    'MLIRTransform',
    'Pooling',
    'RISCVBuddyExt',
    'RVVDialect',
    'RVVExperiment',
    'ScheDialect',
    'SIMDExperiment',
    'ToyDSL',
    'VectorExpDialect',
    'log.mlir'
]

config.buddy_tools_dir = os.path.join(config.buddy_obj_root, 'bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.buddy_tools_dir, config.llvm_tools_dir]
tools = [
    'buddy-opt',
    'buddy-translate',
    'mlir-cpu-runner',
]
tools.extend([
    ToolSubst('%mlir_runner_utils_dir', config.mlir_runner_utils_dir, unresolved='ignore'),
])

llvm_config.add_tool_substitutions(tools, tool_dirs)
