# RUN: %PYTHON %s buddy-opt
# ===- ordered_matmul.py ---------------------------------------------------
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
# Explicit reference order preserves aliasing without claiming snapshot safety.
#
# Lit Python test. Covers explicit reference-order matmul contracts: aliasing
# operands may be declared ordered without claiming snapshot/SPM safety. Checks
# that annotate/verify accept the legal form and reject unsafe reorder claims.
#
# ===---------------------------------------------------------------------------

import re
import subprocess
import sys

TOOL = sys.argv[1]
ORDER = 'contract.compute = {kind = "matmul", order = "mnk"}'
ABI = "contract.abi = {args_disjoint = true}"


def run(source, *passes):
    return subprocess.run(
        [TOOL, *passes], input=source, text=True, capture_output=True
    )


def annotate(source):
    result = run(
        source, "--kernel-contract-annotate", "--kernel-contract-verify"
    )
    assert result.returncode == 0, result.stderr
    repeated = run(
        result.stdout, "--kernel-contract-annotate", "--kernel-contract-verify"
    )
    assert repeated.returncode == 0, repeated.stderr
    return result.stdout


def reject(source, diagnostic, verify=False):
    result = run(
        source,
        "--kernel-contract-verify" if verify else "--kernel-contract-annotate",
    )
    assert result.returncode != 0, result.stdout
    assert diagnostic in result.stderr, result.stderr
    assert "LLVM ERROR" not in result.stderr


def operation(source, label):
    return next(
        line for line in source.splitlines() if f'test.id = "{label}"' in line
    )


def graph(source):
    return re.search(r"contract.compute_graph = \{([^}]*)\}", source)[1]


def external(order=ORDER, abi="", trailing=""):
    return f"""
module attributes {{{abi}}} {{
  func.func @reference(%a: memref<4x4xi32>, %b: memref<4x4xi32>,
                       %c: memref<4x4xi32>) {{
    linalg.matmul {{{order}, test.id = "matmul"}}
      ins(%a, %b : memref<4x4xi32>, memref<4x4xi32>)
      outs(%c : memref<4x4xi32>)
    {trailing}
    return
  }}
}}
"""


# Three different arguments may alias. An explicit mechanical order permits
# execution but cannot manufacture snapshot dependencies or disjointness.
output = annotate(external())
assert re.search(r"\bcomplete = false\b", graph(output))
assert "reference-ordered matmul" in graph(output)
assert "contract.task" not in operation(output, "matmul")
assert "args_disjoint" not in output
reject(external(order="test.default = true"), "V25")
signed = """
module {
  func.func @signed(%a: memref<4x4xi8>, %b: memref<4x4xi8>,
                    %c: memref<4x4xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, FACTS}
      ins(%a, %b : memref<4x4xi8>, memref<4x4xi8>) outs(%c : memref<4x4xi32>)
    return
  }
}
"""
reject(
    signed.replace("FACTS", 'contract.compute = {kind = "dot", signed = true}'),
    "V25",
)
# Reference order is semantic: the same scalar recipe can be requested for
# signed i8, without granting permission to use an AME snapshot execution.
assert re.search(
    r"\bcomplete = false\b", graph(annotate(signed.replace("FACTS", ORDER)))
)
reject(
    """
module {
  func.func @reduce(%input: memref<4xi32>, %init: memref<i32>) {
    linalg.reduce ins(%input : memref<4xi32>) outs(%init : memref<i32>)
      dimensions = [0] (%x: i32, %old: i32) {
      %next = arith.addi %x, %old : i32
      linalg.yield %next : i32
    }
    return
  }
}
""",
    "V25",
)

# Proved independent operands retain the normal complete task representation.
independent = annotate(external(abi=ABI))
assert re.search(r"\bcomplete = true\b", graph(independent))
assert "contract.task" in operation(independent, "matmul")
assert "rematerializable = false" in operation(independent, "matmul")
reject(
    independent.replace("rematerializable = false", "rematerializable = true"),
    "V18/V25",
    verify=True,
)
# Removing the ABI does not make the same claimed task legal under an
# incomplete graph: verification must reject the false snapshot claim.
reject(
    independent.replace("args_disjoint = true", "args_disjoint = false"),
    "V15/V25",
    verify=True,
)

# The order is a narrow semantic contract, not a generic V25 override.
reject(external(order=ORDER.replace('"mnk"', '"nmk"')), "V18/V25")
reject(external(order=ORDER.replace('"matmul"', '"dot"')), "V18/V25")
reject(
    """
module {
  func.func @wrong(%c: memref<4xi32>) {
    %zero = arith.constant 0 : i32
    linalg.fill {contract.compute = {kind = "matmul", order = "mnk"}}
      ins(%zero : i32) outs(%c : memref<4xi32>)
    return
  }
}
""",
    "V18/V25",
)


def private(loop=False):
    matmul = f"""
    linalg.matmul {{{ORDER}, test.id = "matmul"}}
      ins(%c, %b : memref<4x4xi32>, memref<4x4xi32>)
      outs(%c : memref<4x4xi32>)
"""
    if loop:
        matmul = "scf.for %i = %index to %bound step %step {\n" + matmul + "}\n"
    return f"""
module {{
  func.func @local() {{
    %zero = arith.constant 0 : i32
    %index = arith.constant 0 : index
    %step = arith.constant 1 : index
    %bound = arith.constant 3 : index
    %c = memref.alloc() : memref<4x4xi32>
    %b = memref.alloc() : memref<4x4xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<4x4xi32>)
    linalg.fill ins(%zero : i32) outs(%b : memref<4x4xi32>)
    {matmul}
    %unknown = memref.load %c[%index, %index] {{test.id = "unknown"}}
      : memref<4x4xi32>
    %healthy = memref.load %b[%index, %index] {{test.id = "healthy"}}
      : memref<4x4xi32>
    linalg.fill {{test.id = "recover"}} ins(%zero : i32)
      outs(%c : memref<4x4xi32>)
    %known = memref.load %c[%index, %index] {{test.id = "known"}}
      : memref<4x4xi32>
    return
  }}
}}
"""


for loop in (False, True):
    output = annotate(private(loop))
    assert "contract.task" not in operation(output, "matmul")
    assert "contract.task" not in operation(output, "unknown")
    for label in ("healthy", "recover", "known"):
        assert "contract.task" in operation(output, label)
    assert "contract.loop_carried_slots" not in output
    assert re.search(r"\bcomplete = false\b", graph(output))

independent_loop = annotate(f"""
module {{
  func.func @independent_loop(%a: memref<4x4xi32>, %b: memref<4x4xi32>) {{
    %zero = arith.constant 0 : i32
    %index = arith.constant 0 : index
    %step = arith.constant 1 : index
    %bound = arith.constant 3 : index
    %c = memref.alloc() : memref<4x4xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<4x4xi32>)
    scf.for %i = %index to %bound step %step {{
      linalg.matmul {{{ORDER}, test.id = "independent-loop"}}
        ins(%a, %b : memref<4x4xi32>, memref<4x4xi32>)
        outs(%c : memref<4x4xi32>)
    }}
    return
  }}
}}
""")
assert re.search(r"\bcomplete = true\b", graph(independent_loop))
assert "contract.loop_carried_slots" in independent_loop
assert "contract.task" in operation(independent_loop, "independent-loop")

# A terminal store through one argument can modify another argument. A
# different SSA root is not proof of disjointness; an explicit ABI is.
external_accesses = """
module attributes {ABI} {
  func.func @external(%a: memref<4xi32>, %b: memref<4xi32>) {
    %index = arith.constant 0 : index
    %one = arith.constant 1 : i32
    memref.store %one, %a[%index] : memref<4xi32>
    %v = memref.load %b[%index] : memref<4xi32>
    return
  }
}
"""
reject(external_accesses.replace("ABI", ""), "V23/V24")
annotate(external_accesses.replace("ABI", ABI))
reject(
    external(
        trailing="""
    %index = arith.constant 0 : index
    %v = memref.load %a[%index, %index] : memref<4x4xi32>
"""
    ),
    "V23/V24",
)

print(
    "PASS explicit reference matmul order, opaque alias boundaries, and external alias checks"
)
