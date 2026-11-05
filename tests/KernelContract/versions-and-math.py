# RUN: %PYTHON %s buddy-opt
# ===- versions-and-math.py ------------------------------------------------
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
# Versions cannot be laundered by annotation; accuracy is explicit and closed.
#
# Lit Python test driving buddy-opt. contract.version must stay a closed integer
# schema (V1/V18); annotation must not launder illegal versions. contract.math
# accuracy dictionaries are explicit, typed, and closed (V26): abs/rel bounds,
# domain, and special_values are checked; unknown fields and wrong types fail.
# Also rejects target/plan leakage and unknown dialects at annotate time.
#
# ===---------------------------------------------------------------------------

import subprocess
import sys

TOOL = sys.argv[1]


def run(source, *passes):
    return subprocess.run(
        [TOOL, *passes], input=source, text=True, capture_output=True
    )


def reject(source, rule):
    result = run(source, "--kernel-contract-annotate")
    assert result.returncode != 0, result.stdout
    assert rule in result.stderr, result.stderr


for version in ("2 : i32", '"1"', "1.0 : f64", "0 : i64"):
    reject(f"module attributes {{contract.version = {version}}} {{}}", "V1")
for version in ("2 : i32", '"1"', "1.0 : f64"):
    reject(
        f"module {{func.func @f() attributes {{contract.compute_graph = {{version = {version}}}}} {{return}}}}",
        "V18",
    )
reject(
    "module {func.func @f() attributes {contract.compute_graph = {}} {return}}",
    "V18",
)
for width in ("i32", "i64"):
    result = run(
        f"module attributes {{contract.version = 1 : {width}}} {{func.func @f() {{return}}}}",
        "--kernel-contract-annotate",
        "--kernel-contract-verify",
    )
    assert result.returncode == 0, result.stderr

accuracy = '{exp = {domain = "all-f32", special_values = "ieee", max_abs_error = 1.0e-6 : f64, max_rel_error = 1.0e-5 : f64}}'
source = (
    "module {func.func @f() attributes {contract.math = "
    + accuracy
    + "} {return}}"
)
result = run(source, "--kernel-contract-annotate", "--kernel-contract-verify")
assert result.returncode == 0, result.stderr
assert "contract.math" in result.stdout
for before, after in (
    ("exp =", "sqrt ="),
    ('"all-f32"', '"finite-f32"'),
    ('"ieee"', '"ignore"'),
    ("1.0e-6 : f64", "-1.0e-6 : f64"),
    ("1.0e-6 : f64", "1.0e-6 : f32"),
    ("max_abs_error =", "extra_field ="),
):
    reject(source.replace(before, after), "V26")
reject(
    source.replace("max_abs_error =", "extra_field = 0 : i64, max_abs_error ="),
    "V26",
)
plain = run(
    "module {func.func @f() {return}}",
    "--kernel-contract-annotate",
    "--kernel-contract-verify",
)
assert plain.returncode == 0 and "contract.math" not in plain.stdout
reject('module attributes {test.hidden = {plan.route = "dma"}} {}', "V14")
reject("module {func.func @f(%a: memref<4xf32, 3>) {return}}", "V3/V14")
reject(
    "module {module @nested attributes {contract.version = 9 : i64} {}}", "V1"
)
for dialect in ("tt", "tts", "tptr", "bosc_ame", "llvm", "unknown"):
    result = run(
        f'module {{"{dialect}.hidden"() : () -> ()}}',
        "--allow-unregistered-dialect",
        "--kernel-contract-annotate",
    )
    assert result.returncode != 0, result.stdout
    assert (
        "V3/V14" in result.stderr
        or "does not allow unknown operations" in result.stderr
    ), result.stderr
