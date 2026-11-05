# RUN: %PYTHON %s buddy-opt
# ===- unknown-recovery.py -------------------------------------------------
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
# Recover local content facts after full kills without completing opaque scopes.
#
# Lit Python test. After opaque/incomplete scopes fully kill content facts,
# local recoverable facts may be rebuilt without marking those scopes complete.
# Rejects recoveries that would complete an opaque region implicitly.
#
# ===---------------------------------------------------------------------------

import re
import subprocess
import sys

TOOL = sys.argv[1]
TYPE = "memref<4xi32>"


def fill(target, label, scalar="%zero", type_=TYPE):
    return (
        f'linalg.fill {{test.id = "{label}"}} ins({scalar} : i32) '
        f"outs({target} : {type_})\n"
    )


def load(target, label, result=None):
    result = result or "%" + label.replace("-", "_")
    return (
        f'{result} = memref.load {target}[%index] {{test.id = "{label}"}} '
        f": {TYPE}\n"
    )


def copy(source, target, label):
    return (
        f'memref.copy {source}, {target} {{test.id = "{label}"}} '
        f": {TYPE} to {TYPE}\n"
    )


def pointwise(target, label, reads_old=False):
    rhs = "%old" if reads_old else "%x"
    return f"""
    linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}}
      ins(%healthy : {TYPE}) outs({target} : {TYPE})
      attrs = {{test.id = "{label}"}} {{
    ^bb0(%x: i32, %old: i32):
      %sum = arith.addi %x, {rhs} : i32
      linalg.yield %sum : i32
    }}
"""


def module(body, declarations="", arguments=""):
    return f"""
module {{
  {declarations}
  func.func @recover({arguments}) {{
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %index = arith.constant 0 : index
    %step = arith.constant 1 : index
    %c = memref.alloc() : {TYPE}
    %healthy = memref.alloc() : {TYPE}
    %scratch = memref.alloc() : {TYPE}
    {fill("%c", "initial")}
    {fill("%healthy", "unaffected-initial")}
    {body}
    return
  }}
}}
"""


def invoke(source, *passes):
    return subprocess.run(
        [TOOL, *passes], input=source, text=True, capture_output=True
    )


def operation(source, label):
    return next(
        line for line in source.splitlines() if f'test.id = "{label}"' in line
    )


def task(source, label):
    found = re.search(r"contract.task = \{([^}]*)\}", operation(source, label))
    return found[1] if found else None


def ids(record, field):
    found = re.search(rf"\b{field} = array<i64(?:: ([^>]*))?>", record)
    assert found, (record, field)
    return [int(value) for value in found[1].split(",")] if found[1] else []


def annotated(name, source, present=(), absent=()):
    result = invoke(
        source, "--kernel-contract-annotate", "--kernel-contract-verify"
    )
    assert result.returncode == 0, (name, result.stderr)
    text = result.stdout
    graph = re.search(r"contract.compute_graph = \{([^}]*)\}", text)[1]
    assert re.search(r"\bcomplete = false\b", graph), (name, text)
    for label in present:
        assert task(text, label) is not None, (name, label, text)
    for label in absent:
        assert task(text, label) is None, (name, label, text)
    # Reannotation must neither revive omitted tasks nor complete the scope.
    repeated = invoke(
        text, "--kernel-contract-annotate", "--kernel-contract-verify"
    )
    assert repeated.returncode == 0, (name, repeated.stderr)
    repeated_graph = re.search(
        r"contract.compute_graph = \{([^}]*)\}", repeated.stdout
    )[1]
    assert re.search(r"\bcomplete = false\b", repeated_graph), name
    for label in present:
        assert task(repeated.stdout, label) is not None, (name, label)
    for label in absent:
        assert task(repeated.stdout, label) is None, (name, label)
    return text


def restored(text, writer, reader):
    written = ids(task(text, writer), "output_values")
    assert len(written) == 1
    assert ids(task(text, reader), "input_values") == written, text
    # Recovery cannot depend on a stale writer on the other side of the opaque
    # boundary. The scope remains incomplete and prohibits crossing it.
    assert ids(task(text, writer), "input_values") == [], text


def reject(name, text):
    result = invoke(text, "--kernel-contract-verify")
    assert result.returncode != 0, name
    assert "V15/V23" in result.stderr, (name, result.stderr)
    assert (
        "LLVM ERROR" not in result.stderr
        and "PLEASE submit" not in result.stderr
    )


branch = """
    scf.if %condition {
      linalg.fill ins(%one : i32) outs(%c : memref<4xi32>)
    } else {
      linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    }
"""
after_boundary = (
    load("%c", "unknown-load")
    + fill("%c", "unknown-scalar", "%unknown_load")
    + "%prefix = memref.subview %c[0] [2] [1] : memref<4xi32> to memref<2xi32, strided<[1]>>\n"
    + fill("%prefix", "partial", type_="memref<2xi32, strided<[1]>>")
    + copy("%c", "%scratch", "unknown-source")
    + load("%healthy", "unaffected")
    + fill("%c", "recover-fill")
    + load("%c", "recovered-load")
)
text = annotated(
    "if-full-kill",
    module(branch + after_boundary, arguments="%condition: i1"),
    present=("unaffected", "recover-fill", "recovered-load"),
    absent=("unknown-load", "unknown-scalar", "partial", "unknown-source"),
)
restored(text, "recover-fill", "recovered-load")
assert ids(task(text, "recover-fill"), "predecessors") == []
graph = re.search(r"contract.compute_graph = \{([^}]*)\}", text)
tampered = (
    text[: graph.start()]
    + graph[0].replace(
        "source_lineage_complete = false", "source_lineage_complete = true"
    )
    + text[graph.end() :]
)
result = invoke(tampered, "--kernel-contract-verify")
assert result.returncode != 0
assert "kernel contract V12: source lineage completeness" in result.stderr

# A full kill may establish local facts inside an unmodeled arm without
# inventing a parent-scope merge or completing either control-flow scope.
arm_output = annotated(
    "if-arm-local-recovery",
    module(
        "scf.if %condition {\n"
        + load("%c", "arm-unknown")
        + fill("%c", "arm-recover")
        + load("%c", "arm-recovered")
        + "} else {\n}\n"
        + load("%c", "unknown-merge")
        + fill("%c", "parent-recover")
        + load("%c", "parent-recovered"),
        arguments="%condition: i1",
    ),
    present=(
        "arm-recover",
        "arm-recovered",
        "parent-recover",
        "parent-recovered",
    ),
    absent=("arm-unknown", "unknown-merge"),
)
restored(arm_output, "arm-recover", "arm-recovered")
restored(arm_output, "parent-recover", "parent-recovered")

# A full fill with a captured unknown scalar still reads unknown content, and
# replacing it with a partial fill still needs the unmodeled old snapshot.
recovery = operation(text, "recover-fill")
scalar = re.search(r"(%[^ ]+) = memref.load", operation(text, "unknown-load"))[
    1
]
reject(
    "unknown-scalar-claim",
    text.replace(
        recovery,
        re.sub(r"ins\(%[^ ]+ : i32\)", f"ins({scalar} : i32)", recovery),
    ),
)
target = re.search(r"outs\((%[^ ]+) :", recovery)[1]
partial = (
    f"    %partial_recovery = memref.subview {target}[0] [2] [1] : "
    "memref<4xi32> to memref<2xi32, strided<[1]>>\n"
    + recovery.replace(
        f"outs({target} : {TYPE})",
        "outs(%partial_recovery : memref<2xi32, strided<[1]>>)",
    )
)
reject("partial-claim", text.replace(recovery, partial))

loop = """
    scf.for %i = %index to %bound step %step {
      linalg.fill ins(%one : i32) outs(%c : memref<4xi32>)
    }
"""
text = annotated(
    "dynamic-loop-copy-recovery",
    module(
        loop
        + load("%c", "unknown-load")
        + copy("%c", "%scratch", "unknown-source")
        + copy("%healthy", "%c", "recover-copy")
        + load("%c", "recovered-load")
        + load("%healthy", "unaffected"),
        arguments="%bound: index",
    ).replace(
        # Keep this a real copy: an immutable zero source is now safely
        # rematerialized as a fill before graph construction.
        fill("%healthy", "unaffected-initial"),
        fill("%healthy", "unaffected-initial", "%one"),
    ),
    present=("recover-copy", "recovered-load", "unaffected"),
    absent=("unknown-load", "unknown-source"),
)
assert "contract.loop_carried_slots" not in text
assert ids(task(text, "recovered-load"), "input_values") == ids(
    task(text, "recover-copy"), "output_values"
)
recovery = operation(text, "recover-copy")
bad_source = re.search(
    r"memref.copy %[^,]+, (%[^ ]+)", operation(text, "unknown-source")
)[1]
reject(
    "unknown-copy-source-claim",
    text.replace(
        recovery,
        re.sub(r"memref.copy %[^,]+,", f"memref.copy {bad_source},", recovery),
    ),
)

text = annotated(
    "dynamic-loop-local-recovery",
    module(
        "scf.for %i = %index to %bound step %step {\n"
        + load("%c", "loop-unknown")
        + fill("%c", "loop-recover")
        + load("%c", "loop-recovered")
        + "}\n"
        + load("%c", "unknown-loop-exit")
        + load("%healthy", "unaffected"),
        arguments="%bound: index",
    ),
    present=("loop-recover", "loop-recovered", "unaffected"),
    absent=("loop-unknown", "unknown-loop-exit"),
)
assert "contract.loop_carried_slots" not in text
restored(text, "loop-recover", "loop-recovered")

atomic = "%previous = memref.atomic_rmw addi %one, %c[%index] : (i32, memref<4xi32>) -> i32\n"
text = annotated(
    "captured-known-scalar-can-recover",
    module(
        load("%c", "captured")
        + atomic
        + fill("%c", "recover-fill", "%captured")
        + load("%c", "recovered-load")
    ),
    present=("captured", "recover-fill", "recovered-load"),
)
restored(text, "recover-fill", "recovered-load")
captured_id = int(re.search(r"\bid = (\d+) : i64", task(text, "captured"))[1])
assert ids(task(text, "recover-fill"), "predecessors") == [captured_id]

text = annotated(
    "synchronous-effect-pointwise-recovery",
    module(
        atomic
        + pointwise("%c", "reads-old", reads_old=True)
        + fill("%c", "unknown-atomic-scalar", "%previous")
        + pointwise("%c", "recover-pointwise")
        + load("%c", "recovered-load")
        + load("%healthy", "unaffected")
    ),
    present=("recover-pointwise", "recovered-load", "unaffected"),
    absent=("reads-old", "unknown-atomic-scalar"),
)
assert ids(task(text, "recovered-load"), "input_values") == ids(
    task(text, "recover-pointwise"), "output_values"
)
# The second generic's unused output becomes a real unknown init read.
position = text.index('test.id = "recover-pointwise"')
head, tail = text[:position], text[position:]
block = re.search(r"\^bb0\((%[^:]+): i32, (%[^:]+): i32\)", tail)
lhs, old = block.groups()
reject(
    "pointwise-init-read-claim",
    head
    + tail.replace(f"arith.addi {lhs}, {lhs}", f"arith.addi {lhs}, {old}", 1),
)


def reduce(label):
    return f"""
    linalg.reduce ins(%healthy : memref<4xi32>) outs(%sum : memref<i32>)
      dimensions = [0] {{test.id = "{label}"}}
      (%x: i32, %old: i32) {{
      %next = arith.addi %x, %old : i32
      linalg.yield %next : i32
    }}
"""


# A full reduction output still reads its old init, including the single
# element of a rank-zero root. Only a preceding read-free fill recovers.
annotated(
    "reduce-init-is-a-read",
    module(
        "%sum = memref.alloc() : memref<i32>\n"
        + fill("%sum", "sum-initial", type_="memref<i32>")
        + "%atomic_old = memref.atomic_rmw addi %one, %sum[] : (i32, memref<i32>) -> i32\n"
        + reduce("unknown-init")
        + fill("%sum", "recover-init", type_="memref<i32>")
        + reduce("known-init")
        + load("%healthy", "unaffected")
    ),
    present=("recover-init", "known-init", "unaffected"),
    absent=("unknown-init",),
)

# Unknown calls cannot access a nonescaping private sibling. The passed root
# may escape or have outstanding accesses, so even a full overwrite cannot
# recover exclusive content facts for it.
annotated(
    "escape-blocks-recovery",
    module(
        "func.call @opaque(%c) : (memref<4xi32>) -> ()\n"
        + fill("%c", "blocked-fill")
        + load("%c", "blocked-load")
        + load("%healthy", "unaffected"),
        declarations="func.func private @opaque(memref<4xi32>)",
    ),
    present=("unaffected",),
    absent=("blocked-fill", "blocked-load"),
)
annotated(
    "loop-escape-blocks-recovery",
    module(
        "scf.for %i = %index to %bound step %step {\n"
        "  func.call @opaque(%c) : (memref<4xi32>) -> ()\n"
        "}\n" + fill("%c", "blocked-fill") + load("%healthy", "unaffected"),
        declarations="func.func private @opaque(memref<4xi32>)",
        arguments="%bound: index",
    ),
    present=("unaffected",),
    absent=("blocked-fill",),
)
annotated(
    "opaque-region-access-blocks-recovery",
    module(
        "scf.execute_region {\n"
        "  memref.store %one, %c[%index] : memref<4xi32>\n"
        "  scf.yield\n"
        "}\n"
        + fill("%c", "blocked-fill")
        + load("%c", "blocked-load")
        + load("%healthy", "unaffected")
    ),
    present=("unaffected",),
    absent=("blocked-fill", "blocked-load"),
)

# No-argument calls can still touch public DDR storage. A known external
# source is lost, while an unrelated nonescaped local root stays known.
annotated(
    "opaque-external-source",
    module(
        "func.call @opaque() : () -> ()\n"
        + copy("%external", "%c", "unknown-source")
        + load("%healthy", "unaffected")
        + fill("%c", "recover-fill")
        + load("%c", "recovered-load"),
        declarations="func.func private @opaque()",
        arguments="%external: memref<4xi32>",
    ),
    present=("unaffected", "recover-fill", "recovered-load"),
    absent=("unknown-source",),
)
annotated(
    "if-propagates-opaque-external-effects",
    module(
        "scf.if %condition {\n"
        "  func.call @opaque() : () -> ()\n"
        "} else {\n}\n"
        + copy("%external", "%c", "unknown-source")
        + load("%healthy", "unaffected")
        + fill("%c", "recover-fill")
        + load("%c", "recovered-load"),
        declarations="func.func private @opaque()",
        arguments="%condition: i1, %external: memref<4xi32>",
    ),
    present=("unaffected", "recover-fill", "recovered-load"),
    absent=("unknown-source",),
)

# A possible terminal external write remains visible after an unmodeled loop;
# an omitted source assertion must not hide the v1 external write/read rule.
result = invoke(
    module(
        "scf.for %i = %index to %bound step %step {\n"
        + copy("%healthy", "%external", "possible-store")
        + "}\n"
        + copy("%external", "%c", "read-after-store"),
        arguments="%bound: index, %external: memref<4xi32>",
    ),
    "--kernel-contract-annotate",
)
assert result.returncode != 0
assert "V23/V24" in result.stderr, result.stderr

print(
    "PASS unknown content: full-write recovery, scalar/source/init boundaries, escapes, and conservative scope facts"
)
