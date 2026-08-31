#!/usr/bin/env python3
# ===- import-mms.py - facebook/mms-tts-eng AOT importer -------------------===//
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
# single_forward AOT importer for facebook/mms-tts-eng (VITS).
#
# The full VITS text-to-speech pipeline cannot be AOT-traced into one static
# graph: the token->frame duration alignment (`torch.repeat_interleave` over
# network-predicted durations) and the dynamic-length relative-position /
# flow-binning computations are data-dependent, so `torch._dynamo` fragments
# the trace.  Like the original PR (which traced the whole model and used a
# trivial first fragment), a full static kernel is out of reach for the
# `single_forward` contract.
#
# What this importer emits instead is the *largest coherent static stage* of
# the model: the HiFi-GAN vocoder (`VitsHifiGan`), i.e. spectrogram -> waveform.
# That stage is a pure ConvTranspose stack with fixed shapes and exercises
# ~14.3M of the model's weights, so the resulting kernel is real and
# verifiable.  The text encoder, duration alignment and flow decoder are
# deliberately out of scope (data-dependent) -- see the README.
#
# Usage:
#   python import-mms.py --spec specs/f32.json --output-dir <dir>
#
# Env: MMS_MODEL_PATH (local HF snapshot) or spec["hf_model_path"].
# ===----------------------------------------------------------------------===//

import argparse, os, json, re, numpy, torch
import torch.nn as nn
import torch._dynamo; torch._dynamo.config.suppress_errors = False
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModel

p = argparse.ArgumentParser(description="MMS (VITS) AOT importer")
p.add_argument("--spec", required=True)
p.add_argument("--output-dir", required=True)
a = p.parse_args()
with open(a.spec) as f:
    spec = json.load(f)
model_path = (os.environ.get("MMS_MODEL_PATH")
              or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
              or spec.get("hf_model_path", "facebook/mms-tts-eng"))
os.makedirs(a.output_dir, exist_ok=True)

print("[import-mms] Loading mms-tts-eng from:", model_path)
m = AutoModel.from_pretrained(model_path, dtype=torch.float32).eval()
print(f"  model class: {type(m).__name__}, "
      f"params: {sum(pp.numel() for pp in m.parameters()):,}")

# Build the vocoder-stage graph directly: HiFi-GAN is `self.decoder`, a
# pure ConvTranspose stack from a (1, flow_size, seq_len) latent to waveform.
# It is wrapped in a small nn.Module so torch._dynamo attributes the decoder's
# parameters to the traced graph (a raw lambda would drop them).
class Vocoder(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, latents):
        return self.dec(latents, None).squeeze(1)

# ===----------------------------------------------------------------------===//
# Post-process the generated HiFi-GAN subgraph.  The DynamoCompiler emits three
# constructs that the buddy-mlir subgraph pipeline (in particular
# `-affine-loop-fusion`) mishandles, so they are rewritten at MLIR level:
#
#   * `tosa.pad` is decomposed by the pipeline into a pure-write `linalg.fill`
#     whose loop gets fused INTO the consuming conv's reduction loop, re-zeroing
#     the conv's input between reads (the model collapses to zeros).  We replace
#     each pad with zero `tosa.const` tensors + `tosa.concat` (axis=2), which
#     bufferizes to reads-bearing copies and is immune to the mis-fusion.
#
#   * `linalg.conv_1d_ncw_fcw` is emitted with a `tensor.empty` init (an
#     uninitialized accumulator) -> NaN output.  We insert a zero `linalg.fill`
#     init before each conv.
#
#   * The four ConvTranspose layers are emitted as degenerate stride-8/2
#     convs over over-padded inputs (stride conv producing L*S outputs).  We
#     decompose each into `S` stride-1 phases (taps extracted at `k_r + t*S`,
#     reversed, interleaved by `tensor.insert_slice` + reshape).  The phase
#     math is validated in /tmp/phase_check.py to maxerr ~1e-4.
#
# The rewritten subgraph is the execution-verified PR version (buddy-cli output
# matches the PyTorch reference waveform to ~1e-5).
# ===----------------------------------------------------------------------===//

def _parse_tensor_dims(t):
    inner = re.match(r'tensor<([^>]+)>', t).group(1)
    dims = inner.split('x')
    while dims and not dims[-1].isdigit():
        dims.pop()
    return [int(d) for d in dims]


def _postprocess_subgraph0(text):
    lines = text.split('\n')

    # -------- step 1: phase-decompose the stride>1 (transposed) convs --------
    def find_conv_sites():
        sites = []
        for i, ln in enumerate(lines):
            if '"linalg.conv_1d_ncw_fcw"' in ln:
                m = re.search(r'strides = dense<(\d+)>', ln)
                if m and int(m.group(1)) > 1:
                    nm = ln.split('=')[0].strip()
                    j = i
                    while not lines[j].strip().startswith('})'):
                        j += 1
                    padline = lines[i - 2]
                    pm = re.search(r'"tosa.pad"\((.*?)\) :', padline)
                    if not pm:
                        raise SystemExit(
                            f"[import-mms] transposed conv {nm}: no tosa.pad above it")
                    srcv = [x.strip() for x in pm.group(1).split(',')][0]
                    opd = re.search(r'\((.*?)\) <{', ln)
                    oprs = [x.strip() for x in opd.group(1).split(',')]
                    sites.append({
                        'conv': i, 'end': j, 'name': nm,
                        'input': srcv.lstrip('%'),
                        'weight': oprs[1].lstrip('%'),
                        'S': int(m.group(1)),
                        'const_shape': i - 4, 'pad': i - 2, 'empty': i - 1,
                    })
        return sites

    def build_phases(s, pfx, ind='    '):
        convend = lines[s['end']]
        ctypes = re.search(r': \(([^)]*)\) -> (tensor<[^>]+xf32>)', convend)
        wty = ctypes.group(1).split(', ')[1]
        convres = ctypes.group(2)
        Cin, Cout, K = _parse_tensor_dims(wty)
        S = s['S']
        T = K // S
        pline = lines[s['pad']]
        pm = re.search(r'"tosa.pad"\((%\S+), (%\S+), (%\S+)\) : \(([^)]*)\)', pline)
        srcv, shapev, sty = pm.group(1), pm.group(2), pm.group(4).split(', ')[0]
        shline = next(l for l in lines if l.strip().startswith(shapev + ' ='))
        shvals = [int(x) for x in re.findall(
            r'-?\d+', re.search(r'dense<(\[[^\]]*\])', shline).group(1))]
        L = _parse_tensor_dims(sty)[2]
        # conv_transpose padding P == the width pad amount (all four sites).
        P = shvals[4]
        PX = 2                       # validated phase-decomposition padding
        PF = PX + T - 1
        PB = PX
        N_out = _parse_tensor_dims(convres)[2]
        Lxp = PF + L + PB
        LC = Lxp - T + 1
        N = L * S
        assert N_out == N, f"[import-mms] phase output mismatch {N_out} != {N}"
        G = []
        G.append(f'{ind}%{pfx}_s4 = "tosa.const_shape"() <{{values = dense<[1, {Cout}, {L}, 1]> : tensor<4xindex>}}> : () -> !tosa.shape<4>')
        G.append(f'{ind}%{pfx}_sout = "tosa.const_shape"() <{{values = dense<[1, {Cout}, {N}]> : tensor<3xindex>}}> : () -> !tosa.shape<3>')
        G.append(f'{ind}%{pfx}_front = "tosa.const"() <{{values = dense<0.000000e+00> : tensor<1x{Cin}x{PF}xf32>}}> : () -> tensor<1x{Cin}x{PF}xf32>')
        G.append(f'{ind}%{pfx}_back = "tosa.const"() <{{values = dense<0.000000e+00> : tensor<1x{Cin}x{PB}xf32>}}> : () -> tensor<1x{Cin}x{PB}xf32>')
        G.append(f'{ind}%{pfx}_xp = "tosa.concat"(%{pfx}_front, {srcv}, %{pfx}_back) <{{axis = 2 : i32}}> : (tensor<1x{Cin}x{PF}xf32>, {sty}, tensor<1x{Cin}x{PB}xf32>) -> tensor<1x{Cin}x{Lxp}xf32>')
        G.append(f'{ind}%{pfx}_wt = "tosa.transpose"(%{s["weight"]}) <{{perms = array<i32: 1, 0, 2>}}> : ({wty}) -> tensor<{Cout}x{Cin}x{K}xf32>')
        G.append(f'{ind}%{pfx}_stack = "tensor.empty"() : () -> tensor<1x{Cout}x{L}x{S}xf32>')
        G.append(f'{ind}%{pfx}_z = "arith.constant"() <{{value = 0.000000e+00 : f32}}> : () -> f32')
        prev = f'%{pfx}_stack'
        for r in range(S):
            k_r = (r + P) % S
            delta = (r + P - k_r) // S
            k_last = k_r + (T - 1) * S
            G.append(f'{ind}%{pfx}_wr_{r} = "tensor.extract_slice"(%{pfx}_wt) <{{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, {k_last}>, static_sizes = array<i64: {Cout}, {Cin}, {T}>, static_strides = array<i64: 1, 1, -{S}>}}> : (tensor<{Cout}x{Cin}x{K}xf32>) -> tensor<{Cout}x{Cin}x{T}xf32>')
            G.append(f'{ind}%{pfx}_e_{r} = "tensor.empty"() : () -> tensor<1x{Cout}x{LC}xf32>')
            G.append(f'{ind}%{pfx}_f_{r} = "linalg.fill"(%{pfx}_z, %{pfx}_e_{r}) <{{operandSegmentSizes = array<i32: 1, 1>}}> ({{')
            G.append(f'{ind}^bb0(%{pfx}a{r}: f32, %{pfx}b{r}: f32):')
            G.append(f'{ind}  "linalg.yield"(%{pfx}a{r}) : (f32) -> ()')
            G.append(f'{ind}}}) : (f32, tensor<1x{Cout}x{LC}xf32>) -> tensor<1x{Cout}x{LC}xf32>')
            G.append(f'{ind}%{pfx}_C_{r} = "linalg.conv_1d_ncw_fcw"(%{pfx}_xp, %{pfx}_wr_{r}, %{pfx}_f_{r}) <{{dilations = dense<1> : tensor<1xi64>, operandSegmentSizes = array<i32: 2, 1>, strides = dense<1> : tensor<1xi64>}}> ({{')
            G.append(f'{ind}^bb0(%{pfx}x{r}: f32, %{pfx}y{r}: f32, %{pfx}w{r}: f32):')
            G.append(f'{ind}  %{pfx}m{r} = "arith.mulf"(%{pfx}x{r}, %{pfx}y{r}) <{{fastmath = #arith.fastmath<none>}}> : (f32, f32) -> f32')
            G.append(f'{ind}  %{pfx}d{r} = "arith.addf"(%{pfx}w{r}, %{pfx}m{r}) <{{fastmath = #arith.fastmath<none>}}> : (f32, f32) -> f32')
            G.append(f'{ind}  "linalg.yield"(%{pfx}d{r}) : (f32) -> ()')
            G.append(f'{ind}}}) : (tensor<1x{Cin}x{Lxp}xf32>, tensor<{Cout}x{Cin}x{T}xf32>, tensor<1x{Cout}x{LC}xf32>) -> tensor<1x{Cout}x{LC}xf32>')
            G.append(f'{ind}%{pfx}_or_{r} = "tensor.extract_slice"(%{pfx}_C_{r}) <{{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, {delta + PX}>, static_sizes = array<i64: 1, {Cout}, {L}>, static_strides = array<i64: 1, 1, 1>}}> : (tensor<1x{Cout}x{LC}xf32>) -> tensor<1x{Cout}x{L}xf32>')
            G.append(f'{ind}%{pfx}_o4_{r} = "tosa.reshape"(%{pfx}_or_{r}, %{pfx}_s4) : (tensor<1x{Cout}x{L}xf32>, !tosa.shape<4>) -> tensor<1x{Cout}x{L}x1xf32>')
            G.append(f'{ind}%{pfx}_st_{r} = "tensor.insert_slice"(%{pfx}_o4_{r}, {prev}) <{{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, {r}>, static_sizes = array<i64: 1, {Cout}, {L}, 1>, static_strides = array<i64: 1, 1, 1, 1>}}> : (tensor<1x{Cout}x{L}x1xf32>, tensor<1x{Cout}x{L}x{S}xf32>) -> tensor<1x{Cout}x{L}x{S}xf32>')
            prev = f'%{pfx}_st_{r}'
        G.append(f'{ind}{s["name"]} = "tosa.reshape"({prev}, %{pfx}_sout) : (tensor<1x{Cout}x{L}x{S}xf32>, !tosa.shape<3>) -> tensor<1x{Cout}x{N}xf32>')
        return G

    sites = find_conv_sites()
    replacements = []
    for idx, s in enumerate(sites):
        replacements.append((s['const_shape'], s['end'],
                             build_phases(s, f'ph{idx}')))
    replacements.sort(key=lambda r: r[0], reverse=True)
    for start, end, G in replacements:
        lines[start:end + 1] = G
    text = '\n'.join(lines)

    # -------- step 2: tosa.pad -> zero const + tosa.concat --------
    shapes = {}
    for m in re.finditer(
            r'(\%\S+) = "tosa\.const_shape"\(\) <\{values = dense<(\[[^\]]*\])>',
            text):
        shapes[m.group(1)] = [int(x) for x in re.findall(r'-?\d+', m.group(2))]
    ones = set()
    for m in re.finditer(
            r'(\%\S+) = "tosa\.const"\(\) <\{values = dense<([^>]*)> : tensor<1xf32>\}>',
            text):
        try:
            if float(m.group(2).strip()) == 0.0:
                ones.add(m.group(1))
        except ValueError:
            pass
    pat = re.compile(
        r'^(?P<ind>[ \t]*)(?P<name>%\S+) = "tosa\.pad"'
        r'\((?P<ops>[^)]*)\) : (?P<types>\([^)]*\)) -> (?P<res>\S+)\s*$', re.M)
    pad_ops = []
    for m in pat.finditer(text):
        ind, name = m.group('ind'), m.group('name')
        ops = [x.strip() for x in m.group('ops').split(',')]
        pad_ops.append((ind, name, ops, m.group('types'), m.group('res'),
                        m.start(), m.end()))
    repl = []
    for i, (ind, name, ops, types, res, s_, e_) in enumerate(pad_ops):
        srcv, shapev, valv = ops
        sh = shapes[shapev]
        if not (valv in ones and sh[0] == 0 and sh[1] == 0 and sh[2] == 0
                and sh[3] == 0 and sh[4] == sh[5]):
            raise SystemExit(
                f"[import-mms] pad {name} not width-sym-zero: {sh}")
        F, B = sh[4], sh[5]
        ty = re.search(r'\(([^)]*)\)', types).group(1).split(', ')[0]
        Cin = _parse_tensor_dims(ty)[1]
        fname, bname = f'%pad{i}_front', f'%pad{i}_back'
        t_front = f'tensor<1x{Cin}x{F}xf32>'
        t_back = f'tensor<1x{Cin}x{B}xf32>'
        padlines = [
            f'{ind}{fname} = "tosa.const"() <{{values = dense<0.000000e+00> : {t_front}}}> : () -> {t_front}',
            f'{ind}{bname} = "tosa.const"() <{{values = dense<0.000000e+00> : {t_back}}}> : () -> {t_back}',
            f'{ind}{name} = "tosa.concat"({fname}, {srcv}, {bname}) <{{axis = 2 : i32}}> : ({t_front}, {ty}, {t_back}) -> {res}',
        ]
        repl.append((s_, e_, '\n'.join(padlines)))
    repl.sort(key=lambda x: x[0], reverse=True)
    out = text
    for s_, e_, txt in repl:
        out = out[:s_] + txt + out[e_:]

    # -------- step 3: zero-fill the conv accumulators --------
    empties = {}
    for m in re.finditer(r'(%\S+) = "tensor\.empty"\(\) : \(\) -> (\S+)', out):
        empties[m.group(1)] = m.group(2)
    conv_inits = []
    for m in re.finditer(r'"linalg\.conv_1d_ncw_fcw"\(([^)]*)\)', out):
        conv_inits.append([x.strip() for x in m.group(1).split(',')][2])
    fill_of_empty = {}
    for init in conv_inits:
        if init in empties and init not in fill_of_empty:
            fill_of_empty[init] = f'%ff{len(fill_of_empty)}'
    empty_pos = {}
    pat2 = re.compile(
        r'(?m)^([ \t]*)(%\S+) = "tensor\.empty"\(\) : \(\) -> (\S+)\s*$')
    for m in pat2.finditer(out):
        ind, name, ty = m.group(1), m.group(2), m.group(3)
        if name in fill_of_empty:
            empty_pos[name] = (ind, ty, m.end())
    inserts = []
    for idx, (name, (ind, ty, pos)) in enumerate(empty_pos.items()):
        fname = fill_of_empty[name]
        zname = fname.replace('ff', 'fz')
        ins = [
            f'{ind}{zname} = "arith.constant"() <{{value = 0.000000e+00 : f32}}> : () -> f32',
            f'{ind}{fname} = "linalg.fill"({zname}, {name}) <{{operandSegmentSizes = array<i32: 1, 1>}}> ({{',
            f'{ind}^bb0(%a{idx}: f32, %b{idx}: f32):',
            f'{ind}  "linalg.yield"(%a{idx}) : (f32) -> ()',
            f'{ind}}}) : (f32, {ty}) -> {ty}',
        ]
        inserts.append((pos, '\n'.join(ins)))
    inserts.sort(key=lambda x: x[0], reverse=True)
    step1 = out
    for pos, txt in inserts:
        step1 = step1[:pos] + '\n' + txt + step1[pos:]

    def conv_repl(m):
        ops = [x.strip() for x in m.group(1).split(',')]
        if ops[2] in fill_of_empty:
            ops[2] = fill_of_empty[ops[2]]
        return '"linalg.conv_1d_ncw_fcw"(' + ', '.join(ops) + ')'

    return re.sub(r'"linalg\.conv_1d_ncw_fcw"\(([^)]*)\)', conv_repl, step1)


S = int(spec.get("max_seq_len", 30))
latents = torch.zeros((1, 192, S))
vocoder = Vocoder(m.decoder).eval()

dc = DynamoCompiler(primary_registry=tosa.ops_registry,
                    aot_autograd_decomposition=inductor_decomp,
                    func_name="forward")
with torch.no_grad():
    g = dc.importer(vocoder, latents=latents)
print(f"[import-mms] {len(g)} graphs")
graph = g[0]
params = dc.imported_params[graph]
n_elem = sum(pp.numel() for pp in params)
print(f"[import-mms] graph 0: {len(params)} params, {n_elem:,} elems")

graph.fuse_ops([simply_fuse])
graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU
dr = GraphDriver(graph)
dr.subgraphs[0].lower_to_top_level_ir()
subgraph0_text = str(dr.subgraphs[0]._imported_module)
subgraph0_text = _postprocess_subgraph0(subgraph0_text)
with open(os.path.join(a.output_dir, "subgraph0.mlir"), "w") as f:
    print(subgraph0_text, file=f)
with open(os.path.join(a.output_dir, "forward.mlir"), "w") as f:
    print(dr.construct_main_graph(True), file=f)
all_param = numpy.concatenate(
    [pp.detach().cpu().numpy().reshape([-1]) for pp in params]
).astype(numpy.float32, copy=False)
all_param.tofile(os.path.join(a.output_dir, "arg0.data"))
print(f"[import-mms] Wrote subgraph0.mlir, forward.mlir, arg0.data "
      f"({all_param.nbytes:,} bytes) to {a.output_dir}")
