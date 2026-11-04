#!/usr/bin/env python3
# ===- gen_yolov8_manifest.py - RHAL manifest for YOLOv8 ------------------===//
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
#
# Usage:
#   python gen_yolov8_manifest.py --spec specs/k3_320.json -o yolov8.mlir
#
# The class labels travel as vocab_uri, one label per line (COCO; a pose
# model uses the first, person).
#
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import sys


def normalize_uri(raw: str) -> str:
    s = raw.strip()
    if ":" in s:
        return s
    return f"file:{s}"


def gen_manifest(
    spec: dict,
    runner_library: str,
    model_library: str | None = None,
    dep_shared_libs: list[str] | None = None,
    version: str = "0.1.0",
) -> str:
    model_id = spec.get("model_id", f"{spec['model_family']}_{spec['variant']}")
    params_size = int(spec["params_size"])
    img_size = int(spec["img_size"])
    num_classes = int(spec["num_classes"])
    num_keypoints = int(spec.get("num_keypoints", 0))
    num_anchors = int(spec["num_anchors"])
    channels = 4 + num_classes + 3 * num_keypoints
    so_name = spec.get("so_name", "yolov8_model.so")
    weight_file = spec.get("weight_file", "arg0.data")
    labels_file = spec.get("labels_file", "labels.txt")

    lines = []
    p = lines.append
    p("rhal.module @yolov8 attributes {")
    p(f'    version = "{version}",')
    p(f'    model_name = "{model_id}",')
    p(f'    vocab_uri = "file:{labels_file}",')
    p(f'    img_size = "{img_size}",')
    p(f'    num_classes = "{num_classes}",')
    p(f'    num_keypoints = "{num_keypoints}",')
    p(f'    num_anchors = "{num_anchors}",')
    p(f'    conf_threshold = "{float(spec.get("conf_threshold", 0.25))}",')
    p(f'    iou_threshold = "{float(spec.get("iou_threshold", 0.7))}",')
    p(f'    max_detections = "{int(spec.get("max_detections", 300))}",')
    p(
        f'    runner_library = "{runner_library}"'
        + ("," if model_library else "")
    )
    if model_library:
        p(f'    model_library = "{normalize_uri(model_library)}"}} {{')
    else:
        lines[-1] += "} {"
    p("")
    p('  rhal.constant @params {id = 1 : i32, storage = "external",')
    p(f"                         type = tensor<{params_size}xf32>,")
    p(f'                         uri = "file:{weight_file}"}}')
    p("")
    p('  rhal.codeobj @model_kernels {id = 1 : i32, kind = "host_shared_lib",')
    p('                                backend = "cpu",')
    p(f'                                uri = "file:{so_name}"}}')
    for idx, dep in enumerate(dep_shared_libs or [], start=2):
        p(
            f'  rhal.codeobj @runtime_dep_{idx - 1} {{id = {idx} : i32, kind = "host_shared_lib",'
        )
        p('                                backend = "cpu",')
        p(f'                                uri = "{normalize_uri(dep)}"}}')
    p("")
    p(
        f'  rhal.buffer @images {{space = "host", '
        f"type = tensor<1x3x{img_size}x{img_size}xf32>}}"
    )
    p(
        f'  rhal.buffer @predictions {{space = "host", '
        f"type = tensor<1x{channels}x{num_anchors}xf32>}}"
    )
    p("")
    p("  rhal.func @forward {")
    p('    inputs   = ["images"],')
    p('    outputs  = ["predictions"],')
    p('    dispatch = "model_kernels",')
    p('    args     = ["images", "predictions"]}')
    p("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RHAL manifest for the YOLOv8 model."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the variant spec JSON"
    )
    parser.add_argument(
        "--runner-library",
        default="yolov8_runner.so",
        help="Runner plugin library URI/name for module attrs.",
    )
    parser.add_argument(
        "--model-library",
        default=None,
        help="Server model plugin library URI/name for module attrs.",
    )
    parser.add_argument(
        "-o", "--output", default="-", help="Output path (- for stdout)"
    )
    parser.add_argument(
        "--version",
        default="0.1.0",
        help="RAX module version string (usually the CLI release version).",
    )
    parser.add_argument(
        "--dep-shared-lib",
        action="append",
        default=[],
        metavar="URI_OR_NAME",
        help="Additional shared library dependency URI/name (repeatable).",
    )
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    text = gen_manifest(
        spec,
        normalize_uri(args.runner_library),
        args.model_library,
        dep_shared_libs=args.dep_shared_lib,
        version=args.version,
    )

    if args.output == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w") as f:
            f.write(text)
        print(f"[gen_yolov8_manifest] Written: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
