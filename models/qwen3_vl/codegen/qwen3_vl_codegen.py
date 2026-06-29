#!/usr/bin/env python3
"""Qwen3-VL codegen utilities.

Subcommands:
  import-vision      Import the pinned-grid vision encoder to MLIR.
  import-decoder-rt  Import the runtime-position decoder to MLIR.
  preprocess         Emit per-query tensors for the runner.
  stage              Assemble the runnable package and pack qwen3_vl.rax.

Run in the buddy Python environment:
  conda activate buddy
  export BUDDY_MLIR_BUILD_DIR=$PWD/build
  export LLVM_MLIR_BUILD_DIR=$PWD/llvm/build
  export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL_DIR = os.environ.get(
    "QWEN3_VL_MODEL_PATH", "/home/gnhuang/models/Qwen3-VL-2B-Instruct"
)
ARTIFACT_DIR = os.path.abspath(
    os.environ.get(
        "QWEN3_VL_OUT_DIR",
        os.path.join(REPO, "build", "models", "qwen3_vl", "artifacts"),
    )
)
PKG_DIR = os.path.abspath(
    os.environ.get(
        "QWEN3_VL_PKG", os.path.join(REPO, "build", "models", "qwen3_vl")
    )
)
VISION_DIR = os.path.join(ARTIFACT_DIR, "vision")
DECODER_DIR = os.path.join(ARTIFACT_DIR, "decoder_rt")
TEST_IMAGE = os.path.join(REPO, "models", "qwen3_vl", "test_text.png")
PROMPT = "Read all the text in the image."

IMAGE_TOKEN_ID = 151655
MAX_SEQ_LEN = int(os.environ.get("QWEN3_VL_MAXLEN", "160"))
VOCAB_SIZE = 151936
CANON_WH = (
    448,
    224,
)  # W,H -> grid [1,14,28], matching the compiled vision graph.


def rmsnorm(x, w, eps=1e-6):
    v = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(v + eps))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def load_processor_and_model(dtype=torch.float32, eager_attn=True):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    kwargs = {"dtype": dtype}
    if eager_attn:
        kwargs["attn_implementation"] = "eager"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, **kwargs
    ).eval()
    return processor, model


def encode_image_prompt(processor, image_path, prompt):
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize(CANON_WH)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def capture_decoder_golden():
    processor, model = load_processor_and_model(torch.float32, eager_attn=True)
    inputs = encode_image_prompt(processor, TEST_IMAGE, PROMPT)

    grab = {}
    lm = model.model.language_model
    orig = lm.forward

    def hook(*a, **kw):
        grab["inputs_embeds"] = kw["inputs_embeds"].detach()
        grab["position_ids"] = kw["position_ids"].detach()
        grab["visual_pos_masks"] = kw["visual_pos_masks"].detach()
        grab["deepstack"] = [d.detach() for d in kw["deepstack_visual_embeds"]]
        return orig(*a, **kw)

    lm.forward = hook
    with torch.no_grad():
        out = model(**inputs)
    lm.forward = orig
    grab["logits"] = out.logits.detach()
    grab["input_ids"] = inputs["input_ids"].detach()
    grab["model"] = model
    grab["lm"] = lm
    return grab


class VisionTrace(nn.Module):
    """Trace-friendly Qwen3-VL vision encoder for a single, fixed-grid image."""

    def __init__(self, vm, pos_embeds, cos, sin):
        super().__init__()
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb_vision,
        )

        self._apply_rope = apply_rotary_pos_emb_vision
        self.blocks = vm.blocks
        self.merger = vm.merger
        self.deepstack_merger_list = vm.deepstack_merger_list
        self.deepstack_visual_indexes = list(vm.deepstack_visual_indexes)
        self.num_heads = vm.blocks[0].attn.num_heads
        self.scaling = vm.blocks[0].attn.scaling
        w = vm.patch_embed.proj.weight
        self.register_buffer("pe_w", w.reshape(w.shape[0], -1).clone())
        self.register_buffer("pe_b", vm.patch_embed.proj.bias.clone())
        self.register_buffer("pos_embeds", pos_embeds.clone())
        self.register_buffer("cos", cos.clone())
        self.register_buffer("sin", sin.clone())

    def _attn(self, blk, h):
        seq_len = h.shape[0]
        attn = blk.attn
        q, k, v = (
            attn.qkv(h)
            .reshape(seq_len, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q, k = self._apply_rope(q, k, self.cos, self.sin)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        aw = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        aw = torch.softmax(aw, dim=-1)
        out = torch.matmul(aw, v).transpose(1, 2).reshape(seq_len, -1)
        return attn.proj(out)

    def forward(self, pixel_values):
        h = pixel_values @ self.pe_w.t() + self.pe_b
        h = h + self.pos_embeds
        deepstack = []
        for i, blk in enumerate(self.blocks):
            h = h + self._attn(blk, blk.norm1(h))
            h = h + blk.mlp(blk.norm2(h))
            if i in self.deepstack_visual_indexes:
                j = self.deepstack_visual_indexes.index(i)
                deepstack.append(self.deepstack_merger_list[j](h))
        pooled = self.merger(h)
        return (pooled, *deepstack)


class DecoderTraceRT(nn.Module):
    """Qwen3-VL decoder with cos/sin/cmask as runtime forward inputs."""

    def __init__(self, lm, lm_head_w, deepstack_layers):
        super().__init__()
        self.layers = lm.layers
        self.norm = lm.norm
        self.n_heads = lm.config.num_attention_heads
        self.n_kv = lm.config.num_key_value_heads
        self.head_dim = lm.config.head_dim
        self.scaling = self.head_dim**-0.5
        self.eps = lm.config.rms_norm_eps
        self.deepstack_layers = deepstack_layers
        self.register_buffer("lm_head_w", lm_head_w.clone())

    def _attn(self, attn, h, cos, sin, cmask):
        batch, seq_len, _ = h.shape
        q = rmsnorm(
            attn.q_proj(h).view(batch, seq_len, self.n_heads, self.head_dim),
            attn.q_norm.weight,
            self.eps,
        ).transpose(1, 2)
        k = rmsnorm(
            attn.k_proj(h).view(batch, seq_len, self.n_kv, self.head_dim),
            attn.k_norm.weight,
            self.eps,
        ).transpose(1, 2)
        v = (
            attn.v_proj(h)
            .view(batch, seq_len, self.n_kv, self.head_dim)
            .transpose(1, 2)
        )
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        aw = torch.matmul(q, k.transpose(2, 3)) * self.scaling + cmask
        aw = torch.softmax(aw, dim=-1)
        o = torch.matmul(aw, v).transpose(1, 2).reshape(batch, seq_len, -1)
        return attn.o_proj(o)

    def forward(self, inputs_embeds, cos, sin, cmask, ds0, ds1, ds2):
        ds = [ds0, ds1, ds2]
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            residual = h
            h = self._attn(
                layer.self_attn,
                rmsnorm(h, layer.input_layernorm.weight, self.eps),
                c,
                s,
                cmask,
            )
            h = residual + h
            residual = h
            mlp = layer.mlp
            pn = rmsnorm(h, layer.post_attention_layernorm.weight, self.eps)
            h = residual + mlp.down_proj(
                torch.nn.functional.silu(mlp.gate_proj(pn)) * mlp.up_proj(pn)
            )
            if i < self.deepstack_layers:
                h = h + ds[i]
        h = rmsnorm(h, self.norm.weight, self.eps)
        return h @ self.lm_head_w.t()


def import_graph(module, out_dir, prefix, *example_inputs):
    import numpy
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph import GraphDriver
    from buddy.compiler.graph.transform import simply_fuse
    from buddy.compiler.ops import tosa
    from torch._inductor.decomposition import decompositions as inductor_decomp

    dynamo = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name="forward",
    )
    with torch.no_grad():
        graphs = dynamo.importer(module, *example_inputs)
    print(f"[import] graphs={len(graphs)}")
    graph = graphs[0]
    params = dynamo.imported_params[graph]
    graph.fuse_ops([simply_fuse])
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    with open(os.path.join(out_dir, f"{prefix}_subgraph0.mlir"), "w") as f:
        print(driver.subgraphs[0]._imported_module, file=f)
    with open(os.path.join(out_dir, f"{prefix}_forward.mlir"), "w") as f:
        print(driver.construct_main_graph(True), file=f)
    all_param = numpy.concatenate(
        [p.detach().numpy().reshape([-1]) for p in params]
    )
    all_param.tofile(os.path.join(out_dir, f"{prefix}_arg0.data"))
    return all_param.size


def cmd_import_vision(args):
    os.makedirs(VISION_DIR, exist_ok=True)
    torch.set_grad_enabled(False)
    processor, model = load_processor_and_model(torch.float32, eager_attn=True)
    enc = encode_image_prompt(processor, TEST_IMAGE, "ocr")
    pixel_values = enc["pixel_values"].float()
    grid_thw = enc["image_grid_thw"].long()
    print(
        f"[in] pixel_values {tuple(pixel_values.shape)} grid {grid_thw.tolist()}"
    )

    vm = model.model.visual
    ref = vm(hidden_states=pixel_values, grid_thw=grid_thw)
    ref_pooled = ref.pooler_output
    ref_ds = list(ref.deepstack_features)
    print(
        f"[ref] pooled {tuple(ref_pooled.shape)} | "
        f"deepstack x{len(ref_ds)} each {tuple(ref_ds[0].shape)}"
    )

    pos_embeds = vm.fast_pos_embed_interpolate(grid_thw)
    rotary = vm.rot_pos_emb(grid_thw)
    emb = torch.cat((rotary, rotary), dim=-1)
    trace = VisionTrace(vm, pos_embeds, emb.cos(), emb.sin()).eval()
    out = trace(pixel_values)
    pooled, ds = out[0], list(out[1:])
    dp = (pooled - ref_pooled).abs().max().item()
    dd = max((a - b).abs().max().item() for a, b in zip(ds, ref_ds))
    print(f"[equiv] pooled max|delta|={dp:.3e}  deepstack max|delta|={dd:.3e}")
    assert dp < 2e-2 and dd < 2e-2, (
        "trace wrapper diverges from HF vision model"
    )
    print("[equiv] OK: trace-friendly wrapper matches HF vision model")

    if args.no_import:
        return
    print("[import] running buddy DynamoCompiler on the vision wrapper ...")
    weight_count = import_graph(trace, VISION_DIR, "vision", pixel_values)
    print(
        f"[import] OK -> {VISION_DIR}/vision_forward.mlir weights={weight_count}"
    )


def make_decoder_inputs(seq_len):
    golden = capture_decoder_golden()
    lm, model = golden["lm"], golden["model"]
    inputs_embeds = golden["inputs_embeds"]
    pos = golden["position_ids"]
    vmask = golden["visual_pos_masks"]
    deepstack = golden["deepstack"]
    _, prompt_len, hidden = inputs_embeds.shape

    rope_pos = pos[1:] if pos.shape[0] == 4 else pos
    max_pos = int(rope_pos.max())
    tail = torch.arange(max_pos + 1, max_pos + 1 + (seq_len - prompt_len))
    tail = tail.view(1, 1, -1).expand(3, 1, seq_len - prompt_len)
    rope_pos_n = torch.cat([rope_pos, tail], dim=2)
    cos, sin = lm.rotary_emb(torch.zeros(1, seq_len, hidden), rope_pos_n)
    cos, sin = cos[0], sin[0]
    cmask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), 1)
    cmask = cmask.view(1, 1, seq_len, seq_len)
    padded_embeds = torch.zeros(1, seq_len, hidden)
    padded_embeds[:, :prompt_len] = inputs_embeds
    img_pos = vmask[0].nonzero(as_tuple=True)[0]
    padded_deepstack = []
    for d in deepstack:
        f = torch.zeros(1, seq_len, hidden)
        f[0, img_pos] = d.float()
        padded_deepstack.append(f)
    trace = DecoderTraceRT(
        lm, model.lm_head.weight, deepstack_layers=len(deepstack)
    ).eval()
    return (
        trace,
        model,
        prompt_len,
        padded_embeds,
        cos,
        sin,
        cmask,
        padded_deepstack,
    )


def cmd_import_decoder_rt(args):
    os.makedirs(DECODER_DIR, exist_ok=True)
    torch.set_grad_enabled(False)
    trace, model, prompt_len, inputs_embeds, cos, sin, cmask, ds = (
        make_decoder_inputs(args.seq_len)
    )
    logits = trace(inputs_embeds, cos, sin, cmask, ds[0], ds[1], ds[2])
    tok = int(logits[0, prompt_len - 1].argmax())
    print(f"[rt] N={args.seq_len} next token at prompt end = {tok} (expect 33)")
    assert tok == 33

    model.lm_head.weight.detach().float().numpy().tofile(
        os.path.join(DECODER_DIR, "embed_table.bin")
    )
    if args.no_import:
        return
    weight_count = import_graph(
        trace,
        DECODER_DIR,
        "decoder",
        inputs_embeds,
        cos,
        sin,
        cmask,
        ds[0],
        ds[1],
        ds[2],
    )
    print(
        f"[rt] imported -> {DECODER_DIR}/decoder_forward.mlir weights={weight_count}"
    )


def cmd_preprocess(args):
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)
    processor, model = load_processor_and_model(
        torch.bfloat16, eager_attn=False
    )
    inputs = encode_image_prompt(processor, args.image_path, args.prompt)

    input_ids = inputs["input_ids"]
    grid = inputs["image_grid_thw"]
    prompt_len = input_ids.shape[1]
    assert grid.tolist() == [[1, 14, 28]], f"unexpected grid {grid.tolist()}"
    assert prompt_len < MAX_SEQ_LEN, (
        f"prompt too long: S0={prompt_len} >= N={MAX_SEQ_LEN}"
    )

    pos = model.model.compute_3d_position_ids(
        input_ids=input_ids,
        inputs_embeds=None,
        image_grid_thw=grid,
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    rope_pos = pos[-3:]
    max_pos = int(rope_pos.max())
    tail = torch.arange(max_pos + 1, max_pos + 1 + (MAX_SEQ_LEN - prompt_len))
    tail = tail.view(1, 1, -1).expand(3, 1, MAX_SEQ_LEN - prompt_len)
    rope_pos_n = torch.cat([rope_pos, tail], dim=2)
    hidden = model.config.text_config.hidden_size
    cos, sin = model.model.language_model.rotary_emb(
        torch.zeros(1, MAX_SEQ_LEN, hidden), rope_pos_n
    )
    cmask = np.triu(np.full((MAX_SEQ_LEN, MAX_SEQ_LEN), -np.inf, np.float32), 1)
    cmask = cmask.reshape(1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN)
    img_pos = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0].numpy()

    inputs["pixel_values"].float().numpy().astype(np.float32).tofile(
        os.path.join(out, "pixel_values.bin")
    )
    input_ids[0].numpy().astype(np.int64).tofile(
        os.path.join(out, "input_ids.i64")
    )
    img_pos.astype(np.int64).tofile(os.path.join(out, "img_pos.i64"))
    cos[0].float().numpy().astype(np.float32).tofile(
        os.path.join(out, "cos.bin")
    )
    sin[0].float().numpy().astype(np.float32).tofile(
        os.path.join(out, "sin.bin")
    )
    cmask.astype(np.float32).tofile(os.path.join(out, "cmask.bin"))
    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(
            f"{prompt_len} {MAX_SEQ_LEN} {len(img_pos)} {hidden} "
            f"{model.config.text_config.vocab_size}\n"
        )
    print(
        f"[preprocess] S0={prompt_len} N={MAX_SEQ_LEN} "
        f"img_tokens={len(img_pos)} -> {out}"
    )


def link(src, dst):
    if os.path.lexists(dst) and os.path.realpath(src) == os.path.realpath(dst):
        return
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def cmd_stage(args):
    os.makedirs(PKG_DIR, exist_ok=True)
    rax_pack = os.environ.get(
        "RAX_PACK", os.path.join(REPO, "build", "bin", "rax-pack")
    )
    runner_so = os.environ.get(
        "QWEN3_VL_RUNNER_SO", os.path.join(PKG_DIR, "qwen3_vl_runner.so")
    )
    vocab = os.path.join(REPO, "examples", "BuddyQwen3", "vocab.txt")
    pybin = os.environ.get("BUDDY_PYTHON", sys.executable)

    link(
        os.path.join(VISION_DIR, "vision_shim.so"),
        os.path.join(PKG_DIR, "vision_shim.so"),
    )
    link(
        os.path.join(DECODER_DIR, "decoder_shim.so"),
        os.path.join(PKG_DIR, "decoder_shim.so"),
    )
    link(
        os.path.join(VISION_DIR, "vision_arg0.data"),
        os.path.join(PKG_DIR, "vision_weights.data"),
    )
    link(
        os.path.join(DECODER_DIR, "decoder_arg0.data"),
        os.path.join(PKG_DIR, "decoder_weights.data"),
    )
    link(
        os.path.join(DECODER_DIR, "embed_table.bin"),
        os.path.join(PKG_DIR, "embed_table.bin"),
    )
    link(runner_so, os.path.join(PKG_DIR, "qwen3_vl_runner.so"))
    shutil.copy(vocab, os.path.join(PKG_DIR, "vocab.txt"))

    sh = os.path.join(PKG_DIR, "preprocess.sh")
    with open(sh, "w") as f:
        f.write(
            "#!/usr/bin/env bash\nset -e\n"
            f'export BUDDY_MLIR_BUILD_DIR="${{BUDDY_MLIR_BUILD_DIR:-{REPO}/build}}"\n'
            f'export LLVM_MLIR_BUILD_DIR="${{LLVM_MLIR_BUILD_DIR:-{REPO}/llvm/build}}"\n'
            'export PYTHONPATH="${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}"\n'
            'export CUDA_VISIBLE_DEVICES=""\n'
            f'export QWEN3_VL_MODEL_PATH="${{QWEN3_VL_MODEL_PATH:-{MODEL_DIR}}}"\n'
            f'"{pybin}" "{__file__}" preprocess "$1" "$2" "$3"\n'
        )
    os.chmod(sh, 0o755)

    print("[stage] pre-processing bundled test image ...")
    subprocess.run(["bash", sh, TEST_IMAGE, PROMPT, PKG_DIR], check=True)

    vision_weights = 405049344
    manifest = f"""rhal.module @qwen3_vl attributes {{
    version = "0.1.0",
    model_name = "qwen3_vl",
    vocab_uri = "file:vocab.txt",
    runner_library = "file:qwen3_vl_runner.so"}} {{
  rhal.constant @vision_params {{id = 1 : i32, storage = "external",
                                type = tensor<{vision_weights}xf32>,
                                uri = "file:vision_weights.data"}}
  rhal.codeobj @model_kernels {{id = 1 : i32, kind = "host_shared_lib",
                                backend = "cpu", uri = "file:vision_shim.so"}}
  rhal.buffer @pixel  {{space = "host", type = tensor<392x1536xf32>}}
  rhal.buffer @logits {{space = "host", type = tensor<1x{MAX_SEQ_LEN}x{VOCAB_SIZE}xf32>}}
  rhal.func @forward {{inputs = ["pixel"], outputs = ["logits"],
                      dispatch = "model_kernels", args = ["pixel", "logits"]}}
}}
"""
    mpath = os.path.join(PKG_DIR, "qwen3_vl.mlir")
    with open(mpath, "w") as f:
        f.write(manifest)
    rax = os.path.join(PKG_DIR, "qwen3_vl.rax")
    subprocess.run([rax_pack, mpath, "-o", rax], check=True)
    print(f"[stage] package ready: {PKG_DIR}")
    print(
        f"[run]  {REPO}/build/bin/buddy-cli --model {rax} \\\n"
        f"         --image {TEST_IMAGE} --prompt '{PROMPT}'"
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("import-vision")
    p.add_argument(
        "--no-import",
        action="store_true",
        help="only run the PyTorch equivalence check",
    )
    p.set_defaults(func=cmd_import_vision)

    p = sub.add_parser("import-decoder-rt")
    p.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--no-import", action="store_true")
    p.set_defaults(func=cmd_import_decoder_rt)

    p = sub.add_parser("preprocess")
    p.add_argument("image_path")
    p.add_argument("prompt")
    p.add_argument("out_dir")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("stage")
    p.set_defaults(func=cmd_stage)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
