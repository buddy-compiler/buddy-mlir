# RUN: %PYTHON %s %mlir_runner_utils_dir 2>&1 | FileCheck %s
#
# buddy_model.cmake declares the MLIR runtime libraries a .rax carries in an
# order the runtimes can dlopen them in, one after the other with
# RTLD_GLOBAL: each after the libraries it needs (the RPATH of a library does
# not find their copies on the target).

import glob
import os
import re
import subprocess
import sys
import tempfile

LIB_DIR = sys.argv[1]
CMAKE = os.path.join(
    os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen", "cmake"
)

libs = sorted(glob.glob(os.path.join(LIB_DIR, "libmlir*.so*")))
script = os.path.join(tempfile.mkdtemp(), "order.cmake")
with open(script, "w") as f:
    f.write(
        "cmake_minimum_required(VERSION 3.20)\n"
        f'include("{CMAKE}/buddy_model.cmake")\n'
        f"_buddy_order_runtime_deps(_out {';'.join(libs)})\n"
        'foreach(_lib ${_out})\n  message("${_lib}")\nendforeach()\n'
    )
ordered = subprocess.run(
    ["cmake", "-P", script], capture_output=True, text=True, check=True
).stderr.split()
names = [os.path.basename(lib) for lib in ordered]
print(" ".join(names))
# The alphabetical order puts it before mlir_float16_utils, which it needs.
# CHECK: libmlir_float16_utils.{{.*}} libmlir_c_runner_utils.


def needed(lib):
    dynamic = subprocess.run(
        ["readelf", "-d", lib], capture_output=True, text=True, check=True
    ).stdout
    return re.findall(r"Shared library: \[(.+?)\]", dynamic)


print("no duplicates:", len(set(names)) == len(names))
# CHECK-NEXT: no duplicates: True
print(
    "each after the libraries it needs:",
    all(
        n not in names or names.index(n) < i
        for i, lib in enumerate(ordered)
        for n in needed(lib)
    ),
)
# CHECK-NEXT: each after the libraries it needs: True
