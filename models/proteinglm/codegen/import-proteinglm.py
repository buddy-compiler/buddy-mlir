#!/usr/bin/env python3
# ===- import-proteinglm.py - ProteinGLM AOT importer ---------------------===//

import argparse
import importlib.machinery
import json
import os
import sys
import types

import numpy
import torch

if "deepspeed" not in sys.modules:
    deepspeed_stub = types.ModuleType("deepspeed")
    deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)

    class _Checkpointing:
        @staticmethod
        def is_configured():
            return False

        @staticmethod
        def checkpoint(*_args, **_kwargs):
            raise RuntimeError("deepspeed checkpointing is unavailable")

    deepspeed_stub.checkpointing = _Checkpointing()
    sys.modules["deepspeed"] = deepspeed_stub

if "tomli" not in sys.modules:
    tomli_stub = types.ModuleType("tomli")

    def _tomli_unavailable(*_args, **_kwargs):
        raise RuntimeError("tomli is required only when loading trace configs")

    tomli_stub.load = _tomli_unavailable
    tomli_stub.loads = _tomli_unavailable
    sys.modules["tomli"] = tomli_stub

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform.fuse_ops import simply_fuse
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoConfig, AutoModelForMaskedLM, PreTrainedModel


class ProteinGLMMLMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return out.logits + position_ids.to(out.logits.dtype).sum() * 0.0


def import_proteinglm(spec: dict, output_dir: str) -> None:
    model_path = (
        os.environ.get("PROTEINGLM_MODEL_PATH")
        or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
        or spec["hf_model_path"]
    )
    max_seq_len = int(spec.get("max_seq_len", 1024))
    os.makedirs(output_dir, exist_ok=True)
    hf_home = os.path.join(output_dir, "hf_home")
    hf_modules_cache = os.path.join(output_dir, "hf_modules")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_MODULES_CACHE", hf_modules_cache)

    import transformers.dynamic_module_utils as dynamic_module_utils
    import transformers.utils.hub as hub_utils

    # Transformers computes this cache path at import time. Keep dynamic remote
    # code inside the build tree so sandboxed builds do not write to ~/.cache.
    dynamic_module_utils.HF_MODULES_CACHE = hf_modules_cache
    hub_utils.HF_MODULES_CACHE = hf_modules_cache
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    print(f"[import-proteinglm] Loading model from: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(config, "max_length"):
        config.max_length = int(getattr(config, "seq_length", max_seq_len))
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, config=config, trust_remote_code=True
    )
    model.eval()
    model.config.use_cache = False
    wrapped = ProteinGLMMLMWrapper(model).eval()

    input_ids = torch.zeros((1, max_seq_len), dtype=torch.long)
    attention_mask = torch.ones((1, max_seq_len), dtype=torch.long)
    position_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)

    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    with torch.no_grad():
        # Prime rotary embedding caches outside Dynamo. The model's remote code
        # builds these lazily using dynamic scalar lengths, which otherwise
        # causes graph breaks during fixed-shape AOT import.
        wrapped(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        rotary_classes = set()
        for module in model.modules():
            if module.__class__.__name__ == "RotaryEmbedding":
                rotary_classes.add(module.__class__)
        for rotary_class in rotary_classes:
            original_forward = rotary_class.forward

            def cached_forward(
                self,
                x,
                seq_dim=1,
                seq_len=None,
                _original_forward=original_forward,
            ):
                if (
                    not self.learnable
                    and self.cos_cached is not None
                    and self.sin_cached is not None
                ):
                    return self.cos_cached, self.sin_cached
                return _original_forward(
                    self, x, seq_dim=seq_dim, seq_len=seq_len
                )

            rotary_class.forward = cached_forward
        graphs = dynamo_compiler.importer(
            wrapped,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    if len(graphs) != 1:
        raise RuntimeError(f"expected one graph, got {len(graphs)}")

    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]
    graph.fuse_ops([simply_fuse])
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()

    with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as f:
        print(driver.subgraphs[0]._imported_module, file=f)
    with open(os.path.join(output_dir, "forward.mlir"), "w") as f:
        print(driver.construct_main_graph(True), file=f)

    all_param = numpy.concatenate(
        [param.detach().cpu().numpy().reshape([-1]) for param in params]
    ).astype(numpy.float32, copy=False)
    all_param.tofile(os.path.join(output_dir, "arg0.data"))
    print(
        "[import-proteinglm] Wrote forward.mlir, subgraph0.mlir, arg0.data "
        f"to {output_dir}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ProteinGLM MLM importer")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    import_proteinglm(spec, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
