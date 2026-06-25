#!/usr/bin/env python3
# ===- import-kokoro.py --------------------------------------------------------
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
# Fixed-shape Kokoro-82M importer.
#
# Kokoro is not exported as one monolithic static graph.  The model predicts a
# token duration vector and then builds an alignment matrix whose frame width is
# data dependent.  This importer keeps the neural network portions static and
# leaves that duration-to-frame expansion to the C++ driver:
#
#   forward_predictor/subgraph0_predictor:
#     input ids + reference style -> duration/style/intermediate tensors
#
#   forward_vocoder/subgraph0_vocoder:
#     fixed frame indices + predictor outputs -> waveform
#
# The file names and target structure intentionally follow the prefill/decode
# layout used by LLM examples, with "predictor" and "vocoder" naming the two
# static stages of the TTS pipeline.
#
# ===---------------------------------------------------------------------------

import argparse
import json
import os
import re
from pathlib import Path

import numpy
import torch
import torch._dynamo.config as dynamo_config
import torch.nn.functional as F
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver, OutputOp, PlaceholderOp
from buddy.compiler.graph.transform import (
    apply_classic_fusion,
    eliminate_transpose,
)
from buddy.compiler.ops import tosa
from kokoro import KModel, KPipeline
from kokoro import modules as kokoro_modules
from kokoro.model import KModelForONNX
from torch._inductor.decomposition import decompositions as inductor_decomp


def _fixed_length_text_encoder_forward(self, x, input_lengths, m):
    """Fixed-shape TextEncoder path without packed sequence conversion.

    Kokoro's eager implementation uses sequence packing around the LSTM.  That
    is a good runtime optimization, but it creates shape/data-dependent graph
    structure during export.  The example uses a fixed token length, so the
    packed path can be replaced by the equivalent padded LSTM path.
    """
    x = self.embedding(x)
    x = x.transpose(1, 2)
    m = m.unsqueeze(1)
    x.masked_fill_(m, 0.0)
    for c in self.cnn:
        x = c(x)
        x.masked_fill_(m, 0.0)
    x = x.transpose(1, 2)
    self.lstm.flatten_parameters()
    x, _ = self.lstm(x)
    x = x.transpose(-1, -2)
    x.masked_fill_(m, 0.0)
    return x


def _fixed_length_duration_encoder_forward(self, x, style, text_lengths, m):
    """Fixed-shape DurationEncoder path without packed sequence conversion.

    This mirrors the TextEncoder adjustment for the duration predictor.  The
    LSTM still runs on the same fixed-size tensor, but the graph no longer
    depends on runtime sequence lengths for pack/unpack operations.
    """
    masks = m
    x = x.permute(2, 0, 1)
    s = style.expand(x.shape[0], x.shape[1], -1)
    x = torch.cat([x, s], axis=-1)
    x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
    x = x.transpose(0, 1)
    x = x.transpose(-1, -2)
    for block in self.lstms:
        if isinstance(block, kokoro_modules.AdaLayerNorm):
            x = block(x.transpose(-1, -2), style).transpose(-1, -2)
            x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
            x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
        else:
            x = x.transpose(-1, -2)
            block.flatten_parameters()
            x, _ = block(x)
            x = F.dropout(x, p=self.dropout, training=False)
            x = x.transpose(-1, -2)
    return x.transpose(-1, -2)


def _split_return_types(return_types: str):
    return [item.strip() for item in return_types.split(",")]


def _keep_only_final_return(module_text: str, func_name: str):
    """Keep only the final result of a generated subgraph function.

    The vocoder graph contains decoder-side bookkeeping tensors that are only
    useful to Dynamo's graph-break boundary.  The C++ example needs the final
    waveform, so the generated graph is projected down to its last return value
    before the MLIR is compiled.
    """
    func_start = module_text.find(f"func.func @{func_name}")
    if func_start < 0:
        raise ValueError(f"failed to find function {func_name}")

    signature_match = re.search(
        r"\) -> \((.*?)\) \{", module_text[func_start:], re.S
    )
    if signature_match is None:
        raise ValueError(f"failed to find return signature for {func_name}")
    signature_start = func_start + signature_match.start(1)
    signature_end = func_start + signature_match.end(1)
    return_types = _split_return_types(signature_match.group(1))
    last_return_type = return_types[-1]
    module_text = (
        module_text[:signature_start]
        + last_return_type
        + module_text[signature_end:]
    )

    return_matches = list(
        re.finditer(
            r"\n    return (.*?) : (.*?)\n", module_text[func_start:], re.S
        )
    )
    if not return_matches:
        raise ValueError(f"failed to find return op for {func_name}")
    return_match = return_matches[-1]
    return_start = func_start + return_match.start()
    return_end = func_start + return_match.end()
    return_values = [item.strip() for item in return_match.group(1).split(",")]
    new_return = f"\n    return {return_values[-1]} : {last_return_type}\n"
    return module_text[:return_start] + new_return + module_text[return_end:]


def _keep_only_final_forward_return(
    module_text: str, func_name: str, callee_name: str
):
    """Project the generated forward wrapper to the final subgraph result.

    GraphDriver builds a forward wrapper around the subgraph function.  When
    the vocoder subgraph is trimmed to return only the waveform, the wrapper's
    private declaration, call op, and return op must be rewritten consistently.
    """
    decl_start = module_text.find(f"func.func private @{callee_name}")
    if decl_start < 0:
        raise ValueError(
            f"failed to find private declaration for {callee_name}"
        )
    decl_match = re.search(r"\) -> \((.*?)\)", module_text[decl_start:], re.S)
    if decl_match is None:
        raise ValueError(
            f"failed to find private declaration returns for {callee_name}"
        )
    decl_return_types = _split_return_types(decl_match.group(1))
    last_return_type = decl_return_types[-1]
    decl_ret_start = decl_start + decl_match.start(1)
    decl_ret_end = decl_start + decl_match.end(1)
    module_text = (
        module_text[:decl_ret_start]
        + last_return_type
        + module_text[decl_ret_end:]
    )

    call_match = re.search(
        rf"\n    (%[A-Za-z0-9_]+):(\d+) = call @{re.escape(callee_name)}"
        rf"\((.*?)\) : \((.*?)\) -> \((.*?)\)\n",
        module_text,
        re.S,
    )
    if call_match is None:
        raise ValueError(f"failed to find call to {callee_name}")
    call_result = call_match.group(1)
    call_args = call_match.group(3)
    call_arg_types = call_match.group(4)
    call_start, call_end = call_match.span()
    new_call = (
        f"\n    {call_result} = call @{callee_name}({call_args}) : "
        f"({call_arg_types}) -> {last_return_type}\n"
    )
    module_text = module_text[:call_start] + new_call + module_text[call_end:]

    ret_match = re.search(r"\n    return (.*?) : (.*?)\n", module_text, re.S)
    if ret_match is None:
        raise ValueError("failed to find forward return op")
    ret_start, ret_end = ret_match.span()
    new_return = f"\n    return {call_result} : {last_return_type}\n"
    module_text = module_text[:ret_start] + new_return + module_text[ret_end:]

    func_match = re.search(
        rf"func\.func @{re.escape(func_name)}\(.*?\) -> \((.*?)\) \{{",
        module_text,
        re.S,
    )
    if func_match is None:
        raise ValueError(f"failed to find {func_name} signature")
    sig_start, sig_end = func_match.span(1)
    return module_text[:sig_start] + last_return_type + module_text[sig_end:]


def _write_float_param_pack(output_dir: Path, stage_name: str, params):
    """Write the float parameter pack consumed by a generated forward graph.

    Buddy's packed-parameter ABI creates one memref per dtype.  Kokoro graph 0
    also has an int64 pack for deterministic runtime state; that pack is filled
    by the C++ bridge instead of serialized as another model-weight file.
    """
    if not params:
        return

    float_arrays = []
    for param in params:
        array = param.detach().cpu().numpy().reshape([-1])
        if str(array.dtype) == "float32":
            float_arrays.append(array)

    if float_arrays:
        path = output_dir / f"arg0_{stage_name}.data"
        numpy.concatenate(float_arrays).astype(numpy.float32).tofile(path)


def _load_model(model_path: Path):
    if model_path.is_dir():
        config_path = model_path / "config.json"
        model_file = model_path / "kokoro-v1_0.pth"
        with config_path.open("r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config_dict.setdefault("plbert", {})
        config_dict["plbert"]["_attn_implementation"] = "eager"
        return KModel(config=config_dict, model=str(model_file)).eval()
    return KModel(repo_id=str(model_path)).eval()


def _default_input_ids(model, phonemes: str, seq_len: int):
    """Create a fixed-size non-empty token sample for the C++ example.

    Kokoro's token ABI wraps phoneme IDs with leading/trailing 0 tokens.  A
    zero-filled input is syntactically valid for static import, but it
    represents empty content and produces an almost silent waveform.  The
    default phoneme sample is therefore encoded here instead of in the C++
    runtime.
    """
    token_ids = [
        model.vocab.get(phoneme)
        for phoneme in phonemes
        if model.vocab.get(phoneme) is not None
    ]
    if not token_ids:
        raise ValueError(
            "failed to convert phonemes to Kokoro token ids; pass a phoneme "
            "string such as 'həlˈoʊ wˈɜɹld', not plain text"
        )
    token_ids = [0, *token_ids[: max(0, seq_len - 2)], 0]
    token_ids.extend([0] * (seq_len - len(token_ids)))
    return numpy.asarray(token_ids[:seq_len], dtype=numpy.int64).reshape(
        1, seq_len
    )


def _phonemize_text(text: str, lang_code: str) -> str:
    """Convert raw text to Kokoro phonemes when local G2P is available."""
    try:
        pipeline = KPipeline(lang_code=lang_code, model=False)
        results = list(pipeline(text, split_pattern=None))
    except Exception as exc:
        raise RuntimeError(
            "failed to phonemize --text with Kokoro's local G2P pipeline. "
            "Use --phonemes with an already phonemized Kokoro string, or "
            "install the missing G2P assets such as the spaCy English model."
        ) from exc
    phonemes = "".join(result.phonemes for result in results if result.phonemes)
    if not phonemes:
        raise ValueError(f"failed to phonemize text: {text!r}")
    return phonemes


def _write_default_inputs(
    output_dir: Path, model_path: Path, model, phonemes: str, seq_len: int
):
    input_ids = _default_input_ids(model, phonemes, seq_len)
    input_ids.tofile(output_dir / "input_ids.data")

    ref_s = numpy.zeros((1, 256), dtype=numpy.float32)
    voice_path = model_path / "voices" / "af_heart.pt"
    if voice_path.exists():
        voice_pack = torch.load(voice_path, map_location="cpu")
        if isinstance(voice_pack, torch.Tensor) and voice_pack.numel() >= 256:
            index = min(seq_len, int(voice_pack.shape[0]) - 1)
            ref_s = (
                voice_pack[index]
                .reshape(1, 256)
                .detach()
                .cpu()
                .numpy()
                .astype(numpy.float32)
            )
    ref_s.tofile(output_dir / "ref_s.data")
    return input_ids, ref_s


def _write_graph_artifacts(
    graph,
    stage_name: str,
    output_dir: Path,
    compiler,
    *,
    keep_only_final_return: bool = False,
):
    """Lower one Dynamo graph and save its forward/subgraph MLIR files."""
    func_name = f"forward_{stage_name}"
    subgraph_name = f"subgraph0_{stage_name}"
    graph._func_name = func_name
    graph.perform([eliminate_transpose])
    graph.fuse_ops([apply_classic_fusion])
    _set_single_subgraph_from_body(graph, subgraph_name)

    driver = GraphDriver(graph)
    for subgraph in driver.subgraphs:
        subgraph.lower_to_top_level_ir()

    final_subgraph_name = driver.topological_sort_subgraph()[-1]
    subgraph_files = []
    for subgraph_name, subgraph in driver._subgraphs.items():
        subgraph_path = output_dir / f"{subgraph_name}.mlir"
        subgraph_module = str(subgraph._imported_module)
        if keep_only_final_return and subgraph_name == final_subgraph_name:
            subgraph_module = _keep_only_final_return(
                subgraph_module, subgraph_name
            )
        with subgraph_path.open("w", encoding="utf-8") as f:
            print(subgraph_module, file=f)
        subgraph_files.append(subgraph_path.name)

    forward_path = output_dir / f"{func_name}.mlir"
    forward_module = str(driver.construct_main_graph(True))
    if keep_only_final_return:
        forward_module = _keep_only_final_forward_return(
            forward_module, func_name, final_subgraph_name
        )
    with forward_path.open("w", encoding="utf-8") as f:
        print(forward_module, file=f)

    params = compiler.imported_params[graph]
    _write_float_param_pack(output_dir, stage_name, params)
    print(f"Saved {stage_name} MLIR to {', '.join(subgraph_files)}")


def _remove_stale_outputs(output_dir: Path):
    patterns = [
        "arg0_predictor.data",
        "arg0_vocoder.data",
        "input_ids.data",
        "ref_s.data",
        "forward*.mlir",
        "subgraph*.mlir",
    ]
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _graph_body_ops(graph):
    return [
        op for op in graph.body if not isinstance(op, (PlaceholderOp, OutputOp))
    ]


def _set_single_subgraph_from_body(graph, name: str):
    graph.op_groups = {name: _graph_body_ops(graph)}
    graph.group_map_device = {name: graph.device}


def main():
    parser = argparse.ArgumentParser(
        description="Import fixed-shape Kokoro-82M static pipeline graphs."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.environ.get("KOKORO_MODEL_PATH", "hexgrad/Kokoro-82M")),
        help="Local Kokoro-82M directory or HuggingFace repo id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./"),
        help="Directory to save MLIR, parameter data, and generated config.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=int(os.environ.get("KOKORO_STATIC_SEQ_LEN", "16")),
        help="Fixed phoneme token length.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Fixed Kokoro speed scalar used during graph import.",
    )
    parser.add_argument(
        "--phonemes",
        type=str,
        default="həlˈoʊ wˈɜɹld",
        help=(
            "Fixed Kokoro phoneme string used to generate input_ids.data. "
            "This is already phonemized text, not raw English."
        ),
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help=(
            "Raw text to phonemize with Kokoro's local G2P before generating "
            "input_ids.data. Requires local G2P assets."
        ),
    )
    parser.add_argument(
        "--lang-code",
        type=str,
        default="a",
        help="Kokoro language code used with --text; 'a' is American English.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(args.output_dir)
    dynamo_config.allow_rnn = True
    torch.backends.mkldnn.enabled = False
    kokoro_modules.TextEncoder.forward = _fixed_length_text_encoder_forward
    kokoro_modules.DurationEncoder.forward = (
        _fixed_length_duration_encoder_forward
    )

    base_model = _load_model(args.model_path)
    try:
        phonemes = (
            _phonemize_text(args.text, args.lang_code)
            if args.text is not None
            else args.phonemes
        )
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from None
    print(f"Using phonemes: {phonemes}")
    input_ids_data, ref_s_data = _write_default_inputs(
        args.output_dir,
        args.model_path,
        base_model,
        phonemes,
        args.seq_len,
    )
    model = KModelForONNX(base_model).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        capture_scalar_outputs=True,
    )

    input_ids = torch.from_numpy(input_ids_data.copy())
    ref_s = torch.from_numpy(ref_s_data.copy())

    with torch.no_grad():
        graphs = compiler.importer(
            model,
            input_ids=input_ids,
            ref_s=ref_s,
            speed=args.speed,
        )

    if len(graphs) != 2:
        raise RuntimeError(
            "The fixed-shape Kokoro example expects exactly 2 Dynamo graphs "
            f"(one graph break), but imported {len(graphs)} graph(s). "
            "Check the fixed-length TextEncoder/DurationEncoder patches and "
            "TorchDynamo graph-break logs."
        )

    graph_stages = ["predictor", "vocoder"]
    for index, graph in enumerate(graphs):
        _write_graph_artifacts(
            graph,
            graph_stages[index],
            args.output_dir,
            compiler,
            keep_only_final_return=(index == 1),
        )


if __name__ == "__main__":
    main()
