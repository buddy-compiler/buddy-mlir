#!/usr/bin/env python3
# ruff: noqa: E402
# ===- verify_layer_partition.py - Validate experimental graph split ------===//
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
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_BUDDY_PY_PKG = os.path.join(_REPO_ROOT, "build", "python_packages")
if _BUDDY_PY_PKG not in sys.path:
    sys.path.insert(0, _BUDDY_PY_PKG)

import import_model
from buddy.compiler.graph import PartitionedGraphDriver
from buddy.compiler.graph.operation import OutputOp, PlaceholderOp


def _compute_ops(graph):
    return [
        op for op in graph.body if not isinstance(op, (PlaceholderOp, OutputOp))
    ]


def _op_deps(op):
    deps = []
    for parent in op._parents:
        if isinstance(parent, str):
            deps.append(parent)
        elif hasattr(parent, "name"):
            deps.append(parent.name)
    for arg in op.args:
        if isinstance(arg, list):
            for item in arg:
                if isinstance(item, str):
                    deps.append(item)
                elif hasattr(item, "name"):
                    deps.append(item.name)
    return set(deps)


def verify_driver(graph, config: dict, kind: str) -> dict:
    strategy = import_model.layer_split_strategy(config, kind)
    driver = PartitionedGraphDriver(graph, strategy)

    compute_ops = _compute_ops(graph)
    expected = [op.name for op in compute_ops]
    grouped = [op.name for ops in driver.op_groups.values() for op in ops]

    missing = sorted(set(expected) - set(grouped))
    extra = sorted(set(grouped) - set(expected))
    duplicates = sorted({name for name in grouped if grouped.count(name) > 1})
    order_matches = expected == grouped

    op_to_group = {}
    for group_idx, (_name, ops) in enumerate(driver.op_groups.items()):
        for op in ops:
            op_to_group[op.name] = group_idx

    dependency_violations = []
    for group_idx, (_group_name, ops) in enumerate(driver.op_groups.items()):
        for op in ops:
            for dep in _op_deps(op):
                dep_group = op_to_group.get(dep)
                if dep_group is not None and dep_group > group_idx:
                    dependency_violations.append(
                        {
                            "op": op.name,
                            "op_group": group_idx,
                            "dependency": dep,
                            "dependency_group": dep_group,
                        }
                    )

    ok = (
        not missing
        and not extra
        and not duplicates
        and order_matches
        and not dependency_violations
    )

    return {
        "kind": kind,
        "ok": ok,
        "ops": len(expected),
        "partitions": len(driver.op_groups),
        "missing": missing,
        "extra": extra,
        "duplicates": duplicates,
        "order_matches": order_matches,
        "dependency_violations": dependency_violations[:20],
        "dependency_violation_count": len(dependency_violations),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate DeepSeek R1 experimental per-layer graph split."
    )
    parser.add_argument("--config", required=True, help="Full config JSON")
    parser.add_argument(
        "--report",
        default="",
        help="Optional JSON report output path.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    model = import_model.load_model(config)
    graphs_prefill, graphs_decode, _params = import_model.compile_graphs(
        model, config
    )
    import_model.apply_pre_transforms(graphs_prefill[0], graphs_decode[0])
    import_model.apply_fusion(graphs_prefill[0], graphs_decode[0])

    report = {
        "prefill": verify_driver(graphs_prefill[0], config, "prefill"),
        "decode": verify_driver(graphs_decode[0], config, "decode"),
    }
    report["ok"] = report["prefill"]["ok"] and report["decode"]["ok"]

    if args.report:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.report)), exist_ok=True
        )
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    print(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
