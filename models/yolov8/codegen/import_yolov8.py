#!/usr/bin/env python3
# ===- import_yolov8.py - YOLOv8 AOT importer -----------------------------===//
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
# Imports an Ultralytics YOLOv8 detection or pose checkpoint at the fixed
# input size of the spec. The graph ends at the raw head output
# [1, 4 + num_classes + 3 * num_keypoints, num_anchors]: xywh in input pixels,
# class scores after sigmoid and, for pose, (x, y, visibility) per keypoint
# in input pixels; the runner does the confidence filter and NMS.
#
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import sys

import numpy
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolov8_ime import gen_kernels, rewrite_graph  # noqa: E402

# The kernels of the "ime" rewrite, next to subgraph0.mlir; lower_to_obj.sh
# compiles and links them with it.
KERNELS_MLIR = "yolov8_kernels.mlir"


def import_yolov8(spec: dict, output_dir: str) -> None:
    model_path = (
        os.environ.get("BUDDY_LOCAL_MODEL_PATH")
        or os.environ.get("YOLOV8_MODEL_PATH")
        or spec["checkpoint"]
    )
    img_size = int(spec["img_size"])
    os.makedirs(output_dir, exist_ok=True)

    print(f"[import_yolov8] Loading model from: {model_path}")
    # Conv + BatchNorm folded, as Ultralytics predicts.
    model = YOLO(model_path).model.fuse(verbose=False).float().eval()
    detect = model.model[-1]
    detect.export = True
    num_classes = int(detect.nc)
    if num_classes != int(spec["num_classes"]):
        raise RuntimeError(
            f"checkpoint has {num_classes} classes, spec says "
            f"{spec['num_classes']}"
        )

    example = torch.zeros((1, 3, img_size, img_size), dtype=torch.float32)
    with torch.no_grad():
        # Builds the anchors and strides of the Detect head for this size.
        ref = model(example)
    ref = ref[0] if isinstance(ref, (tuple, list)) else ref
    num_keypoints = int(spec.get("num_keypoints", 0))
    if num_keypoints and list(detect.kpt_shape) != [num_keypoints, 3]:
        raise RuntimeError(
            f"checkpoint has keypoints {list(detect.kpt_shape)}, spec says "
            f"[{num_keypoints}, 3]"
        )
    num_anchors = int(spec["num_anchors"])
    channels = 4 + num_classes + 3 * num_keypoints
    if tuple(ref.shape) != (1, channels, num_anchors):
        raise RuntimeError(
            f"head output is {tuple(ref.shape)}, spec expects "
            f"(1, {channels}, {num_anchors})"
        )

    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name="forward",
    )
    with torch.no_grad():
        graphs = dynamo_compiler.importer(model, example)
    if len(graphs) != 1:
        raise RuntimeError(f"expected one graph, got {len(graphs)}")

    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]
    if spec.get("ime"):
        # The convolutions on the matrix engine of the K3 A100 cores.
        kernels = rewrite_graph(graph, params, num_classes, num_keypoints)
        with open(os.path.join(output_dir, KERNELS_MLIR), "w") as f:
            f.write(gen_kernels(kernels))
        print(f"[import_yolov8] {len(kernels)} IME kernels in {KERNELS_MLIR}")
    elif os.path.exists(os.path.join(output_dir, KERNELS_MLIR)):
        os.remove(os.path.join(output_dir, KERNELS_MLIR))
    graph.fuse_ops([simply_fuse])
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()

    with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as f:
        print(driver.subgraphs[0]._imported_module, file=f)
    with open(os.path.join(output_dir, "forward.mlir"), "w") as f:
        print(driver.construct_main_graph(True), file=f)

    for param in params:
        if param.dtype != torch.float32:
            raise RuntimeError(f"unexpected parameter dtype {param.dtype}")
    all_param = numpy.concatenate(
        [param.detach().cpu().numpy().reshape([-1]) for param in params]
    )
    # f32 slots: the IME kernels pack their weights (ime_params_size)
    key = "ime_params_size" if spec.get("ime") else "params_size"
    if all_param.size != int(spec[key]):
        raise RuntimeError(
            f"imported {all_param.size} parameters, spec {key} says {spec[key]}"
        )
    all_param.tofile(os.path.join(output_dir, "arg0.data"))
    print(
        "[import_yolov8] Wrote forward.mlir, subgraph0.mlir, arg0.data "
        f"({all_param.size} f32) to {output_dir}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8 importer")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    import_yolov8(spec, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
