"""Optional Stage B linker wrappers derived from typed graph calls and reports.

The default accepts one decoder layer; explicit layer selection also supports
norm/linear boundaries in the complete model. It compares full tensors at
external adapter boundaries; wrappers call the original implementation exactly
once and never supply reference values to graph operands.
"""
import hashlib
import json
import math
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_types(text):
    result = []
    for shape, dtype in re.findall(r'tensor<([0-9x]+)x(f32|i8|i32|i64)>', text):
        result.append({'shape': list(map(int, shape.split('x'))), 'dtype': dtype})
    return result


def typed_calls(path):
    calls = {}
    for line in path.read_text().splitlines():
        match = re.search(r'\bcall @(qwen_graph_\w+)\(([^)]*)\)\s*:\s*\((.*?)\)\s*->\s*(.*)$', line)
        if not match:
            continue
        symbol, operands, types, result = match.groups()
        if symbol in calls:
            raise ValueError('Stage B requires one call per graph adapter: ' + symbol)
        args = tensor_types(types)
        outputs = tensor_types(result)
        if len(outputs) > 1 or len(args) != len(operands.split(',')):
            raise ValueError('unsupported typed graph ABI: ' + line)
        calls[symbol] = {'symbol': '_mlir_ciface_' + symbol,
                         'arguments': outputs + args, 'returns_descriptor': bool(outputs),
                         'operands': [operand.strip() for operand in operands.split(',')],
                         'ir_line': line.strip()}
    if not calls:
        raise ValueError('no typed graph external calls in ' + str(path))
    return calls


def build_mapping(args, report):
    if getattr(args, 'intermediate_layers', None) is not None:
        return build_selected_mapping(args, report)
    if args.layers != 1:
        raise ValueError('intermediate probe currently supports exactly one decoder layer')
    if args.interactive or not args.decode_ir or not args.reference_arrays:
        raise ValueError('intermediate probe requires fixed prefill/decode and numeric reference')
    if not args.intermediate_layout or not args.intermediate_graph_dir:
        raise ValueError('intermediate probe requires weight layout and typed graph directory')
    layout = json.loads(args.intermediate_layout.read_text())
    original = report['original_parameters']
    if layout.get('status') != 'PASS' or len(layout['sections']) != len(original):
        raise ValueError('original parameter / checkpoint layout mismatch')
    weights = dict(zip(original, (s['checkpoint_tensor'] for s in layout['sections'])))
    adapter_text = args.adapters.read_text()
    declarations = {}
    for name, arguments in re.findall(r'^void (_mlir_ciface_qwen_graph_\w+)\(([^)]*)\)\s*\{', adapter_text, re.M):
        ranks = re.findall(r'MemRef([1-4])\s*\*\s*a\d+', arguments)
        if len(ranks) != len(arguments.split(',')):
            raise ValueError('unsupported adapter signature: ' + name)
        declarations[name] = list(map(int, ranks))
    entries, used = [], set()
    sources = {}

    for kind, ir in [('prefill', args.graph_ir), ('decode', args.decode_ir)]:
        path = args.intermediate_graph_dir / f'subgraph0_{kind}.triton.mlir'
        calls = typed_calls(path)
        sources[str(path)] = sha(path)
        final_ir = ir.read_text()
        length = args.prefill_len if kind == 'prefill' else 1

        def add(symbol, operand, phase, key, transform='identity', semantic=None):
            call = calls.get(symbol.removeprefix('_mlir_ciface_'))
            if not call or symbol not in declarations:
                raise ValueError('report adapter absent from actual typed graph/source: ' + symbol)
            if declarations[symbol] != [len(a['shape']) for a in call['arguments']]:
                raise ValueError('typed graph and adapter ranks differ: ' + symbol)
            if final_ir.count('call void @' + symbol + '(') != 1:
                raise ValueError('final LLVM must call this adapter once via its bridge: ' + symbol)
            arg = call['arguments'][operand]
            if arg['dtype'] not in ('f32', 'i8'):
                raise ValueError('unsupported intermediate comparison dtype')
            entries.append({'index': len(entries), 'graph': kind, 'layer': 0,
                            'symbol': symbol, 'operand': operand, 'phase': phase,
                            'reference_suffix': key, 'transform': transform,
                            'shape': arg['shape'], 'dtype': arg['dtype'],
                            'rank': len(arg['shape']), 'elements': math.prod(arg['shape']),
                            'semantic': semantic or key, 'ir_line': call['ir_line']})
            used.add(symbol)

        def verify_kernel(symbol, kernel):
            marker = 'void ' + symbol + '('
            if marker not in adapter_text:
                raise ValueError('missing adapter definition: ' + symbol)
            body = adapter_text.split(marker, 1)[1].split('\n}', 1)[0]
            if body.count(kernel + '(') != 1:
                raise ValueError('adapter does not call reported Triton kernel exactly once: ' + symbol)

        # Parameter semantics come from checkpoint value-matched weight layout.
        norms = {'input_layernorm.weight': ('input_hidden', 'input_norm'),
                 'post_attention_layernorm.weight': ('attention_residual', 'post_attention_norm'),
                 'self_attn.q_norm.weight': ('self_attn.q_proj.weight_out', 'q_norm'),
                 'self_attn.k_norm.weight': ('self_attn.k_proj.weight_out', 'k_norm')}
        for row in report['replaced']:
            symbol = row['c_symbol']
            if symbol.removeprefix('_mlir_ciface_') not in calls:
                continue
            verify_kernel(symbol, row['archive_entry'])
            if row['kind'] == 'rmsnorm':
                weight = weights.get(row['operands'][1])
                if weight == 'model.norm.weight':
                    add(symbol, 1, 'before', 'model.layers.0_output_hidden')
                    # Last-token final norm is independently captured at lm_head input.
                    continue
                prefix = 'model.layers.0.'
                if not weight or not weight.startswith(prefix) or weight[len(prefix):] not in norms:
                    raise ValueError('unmapped norm parameter semantics: ' + str(weight))
                before, after = norms[weight[len(prefix):]]
                for operand, phase, name in [(1, 'before', before), (0, 'after', after)]:
                    key = prefix + name if '.weight_' in name else 'model.layers.0_' + name
                    transform = 'projection_heads' if '.weight_' in name else 'identity'
                    add(symbol, operand, phase, key, transform)
            elif row['kind'] == 'silu':
                add(symbol, 1, 'before', 'model.layers.0.mlp.gate_proj.weight_out')
                add(symbol, 0, 'after', 'model.layers.0_silu')

        seen = set()
        for linear in report['w8a8_linears']:
            if linear['node'] in seen:
                continue
            candidates = [r for r in report['w8a8_calls']
                          if r['node'].startswith('w8a8_' + linear['node'] + '_quantize_')
                          and 'qwen_graph_' + r['node'] in calls]
            if not candidates:
                continue
            if len(candidates) != 1:
                # Shared prefill/decode lm_head records are identical; deduplicate.
                candidates = list({r['node']: r for r in candidates}.values())
            if len(candidates) != 1:
                raise ValueError('ambiguous quantization mapping')
            seen.add(linear['node'])
            weight = weights[linear['weight_param']]
            if weight == 'lm_head.weight':
                weight = 'model.embed_tokens.weight'
            if not weight:
                raise ValueError('linear has no checkpoint tensor')
            qcall = candidates[0]
            symbol = '_mlir_ciface_qwen_graph_' + qcall['node']
            verify_kernel(symbol, '_mlir_ciface_kernel_' + qcall['case'])
            transform = 'last_row' if weight == 'model.embed_tokens.weight' else 'identity'
            for operand, phase, suffix in [(0, 'before', '_activation'), (1, 'after', '_q'), (2, 'after', '_a_scale')]:
                add(symbol, operand, phase, weight + suffix, transform)
            dcall = [r for r in report['w8a8_calls']
                     if r['node'].startswith('w8a8_' + linear['node'] + '_dequantize_')
                     and 'qwen_graph_' + r['node'] in calls]
            dcall = list({r['node']: r for r in dcall}.values())
            if len(dcall) != 1:
                raise ValueError('ambiguous dequantization mapping')
            dsymbol = '_mlir_ciface_qwen_graph_' + dcall[0]['node']
            verify_kernel(dsymbol, '_mlir_ciface_kernel_' + dcall[0]['case'])
            add(dsymbol, 3, 'after', weight + '_out', transform)

        # One-layer attention semantics are explicitly checked against report and
        # actual adapter kernel calls; no assumed numbering or fabricated symbol.
        attention = [r for r in report['attention_replacements'] if r['sequence'] == length]
        if len(attention) != 1 or not attention[0]['runtime_length'] or not attention[0]['native_key_layout']:
            raise ValueError('probe requires one native dynamic-position attention subgraph')
        cases = attention[0]['cases']
        for case, operand, phase, key, transform in [
            (cases[0], 0, 'before', 'model.layers.0_q_rope', 'heads_first'),
            (cases[2], 3, 'after', 'model.layers.0_attention_probabilities', 'identity'),
            (cases[3], 3, 'after', 'model.layers.0_attention_context', 'context_heads_first')]:
            matches = [c for c in calls.values() if case in c['symbol']]
            if len(matches) != 1:
                raise ValueError('ambiguous attention adapter')
            symbol = matches[0]['symbol']
            verify_kernel(symbol, '_mlir_ciface_kernel_' + case)
            add(symbol, operand, phase, key, transform)
        sources[str(ir)] = sha(ir)
    return entries, declarations, sources


def selected_layers(args):
    try:
        values = [int(v) for v in args.intermediate_layers.split(',')]
    except (ValueError, AttributeError):
        raise ValueError('--intermediate-layers requires comma-separated layer indices')
    if not values or len(set(values)) != len(values) or any(v < 0 or v >= args.layers for v in values):
        raise ValueError('--intermediate-layers has duplicate or out-of-range indices')
    return sorted(values)


def build_selected_mapping(args, report):
    """Bind selected boundaries by checkpoint identity and actual dataflow."""
    layers = selected_layers(args)
    if args.interactive or not args.decode_ir or not args.reference_arrays:
        raise ValueError('selected probes require fixed prefill/decode and numeric reference')
    if not args.intermediate_layout or not args.intermediate_graph_dir:
        raise ValueError('selected probes require weight layout and typed graph directory')
    layout = json.loads(args.intermediate_layout.read_text())
    original = report['original_parameters']
    if layout.get('status') != 'PASS' or len(layout['sections']) != len(original):
        raise ValueError('original parameter / checkpoint layout mismatch')
    weights = dict(zip(original, (s['checkpoint_tensor'] for s in layout['sections'])))
    adapter_text = args.adapters.read_text()
    declarations = {}
    for name, arguments in re.findall(r'^void (_mlir_ciface_qwen_graph_\w+)\(([^)]*)\)\s*\{', adapter_text, re.M):
        ranks = re.findall(r'MemRef([1-4])\s*\*\s*a\d+', arguments)
        if len(ranks) != len(arguments.split(',')):
            raise ValueError('unsupported adapter signature: ' + name)
        declarations[name] = list(map(int, ranks))
    entries, identities, sources = [], {}, {}
    norms = {'input_layernorm.weight': ('input_hidden', 'input_norm'),
             'post_attention_layernorm.weight': ('attention_residual', 'post_attention_norm'),
             'self_attn.q_norm.weight': ('self_attn.q_proj.weight_out', 'q_norm'),
             'self_attn.k_norm.weight': ('self_attn.k_proj.weight_out', 'k_norm')}
    for kind, ir in [('prefill', args.graph_ir), ('decode', args.decode_ir)]:
        path = args.intermediate_graph_dir / f'subgraph0_{kind}.triton.mlir'
        calls = typed_calls(path)
        sources[str(path)] = sha(path)
        sources[str(ir)] = sha(ir)
        final_ir = ir.read_text()
        from final_operator_mapping import actual_descriptors
        physical_descriptors, _ = actual_descriptors(final_ir, report['graphs'][kind]['entry_abi'])

        def verify_parameter(symbol, operand, parameter):
            call = calls[symbol.removeprefix('_mlir_ciface_')]
            ranks = [len(a['shape']) for a in call['arguments'][int(call['returns_descriptor']):]]
            physical = physical_descriptors(symbol.removeprefix('_mlir_ciface_'), ranks)[operand]
            if physical['aligned'] != {'runtime': f'entry.{parameter}.aligned', 'field': []}:
                raise ValueError('actual LLVM parameter identity mismatch: ' + symbol)

        def verify(symbol, kernel):
            if symbol not in declarations:
                raise ValueError('missing adapter definition: ' + symbol)
            body = adapter_text.split('void ' + symbol + '(', 1)[1].split('\n}', 1)[0]
            if body.count(kernel + '(') != 1:
                raise ValueError('adapter does not call reported Triton kernel exactly once: ' + symbol)
            # These generated adapters have a fixed descriptor order. Checking
            # only the callee name would accept swapped same-rank operands.
            arguments = body.split(kernel + '(', 1)[1].split(');', 1)[0]
            arguments = re.sub(r'\(MemRef[1-4]\s*\*\)', '', arguments)
            actual = [value.strip() for value in arguments.split(',')]
            expected = (['a1', 'a2', '&scratch', 'a0'] if '_kernel_rmsnorm_' in kernel
                        else [f'a{i}' for i in range(len(declarations[symbol]))])
            if actual != expected:
                raise ValueError('adapter kernel argument order mismatch: ' + symbol)

        def add(symbol, operand, phase, key, layer, transform='identity'):
            call = calls.get(symbol.removeprefix('_mlir_ciface_'))
            if not call or symbol not in declarations:
                raise ValueError('adapter absent from typed graph: ' + symbol)
            if declarations[symbol] != [len(a['shape']) for a in call['arguments']]:
                raise ValueError('typed graph and adapter ranks differ: ' + symbol)
            if final_ir.count('call void @' + symbol + '(') != 1:
                raise ValueError('final LLVM must call this adapter once via its bridge: ' + symbol)
            arg = call['arguments'][operand]
            if arg['dtype'] not in ('f32', 'i8'):
                raise ValueError('unsupported selected comparison dtype')
            identity = (kind, symbol, operand, phase)
            if identity in identities:
                old = identities[identity]
                if old['layer'] != layer or old['transform'] != transform:
                    raise ValueError('ambiguous shared probe boundary')
                if key != old['reference_suffix']:
                    old.setdefault('reference_aliases', []).append(key)
                return
            entry = {'index': len(entries), 'graph': kind, 'layer': layer,
                     'symbol': symbol, 'operand': operand, 'phase': phase,
                     'reference_suffix': key, 'transform': transform,
                     'shape': arg['shape'], 'dtype': arg['dtype'],
                     'rank': len(arg['shape']), 'elements': math.prod(arg['shape']),
                     'semantic': key, 'ir_line': call['ir_line']}
            entries.append(entry)
            identities[identity] = entry

        norm_layers = {l: set() for l in layers}
        for row in report['replaced']:
            symbol = row['c_symbol']
            if row['kind'] != 'rmsnorm' or symbol.removeprefix('_mlir_ciface_') not in calls:
                continue
            weight = weights.get(row['operands'][1], '')
            match = re.fullmatch(r'model.layers.(\d+)\.(.*)', weight)
            if not match:
                if weight == 'model.norm.weight' and args.layers - 1 in layers:
                    verify(symbol, row['archive_entry'])
                    verify_parameter(symbol, 1, row['operands'][1])
                    add(symbol, 1, 'before', f'model.layers.{args.layers-1}_output_hidden', args.layers-1)
                continue
            layer, name = int(match[1]), match[2]
            if name == 'input_layernorm.weight' and layer - 1 in layers:
                verify(symbol, row['archive_entry'])
                verify_parameter(symbol, 1, row['operands'][1])
                add(symbol, 1, 'before', f'model.layers.{layer-1}_output_hidden', layer-1)
            if layer not in layers:
                continue
            if name not in norms:
                raise ValueError('unmapped selected norm parameter: ' + weight)
            verify(symbol, row['archive_entry'])
            verify_parameter(symbol, 1, row['operands'][1])
            norm_layers[layer].add(name)
            before, after = norms[name]
            for operand, phase, name in [(1, 'before', before), (0, 'after', after)]:
                key = f'model.layers.{layer}.' + name if '.weight_' in name else f'model.layers.{layer}_' + name
                # The preceding selected layer output and this input are aliases.
                identity = (kind, symbol, operand, phase)
                if identity in identities:
                    identities[identity].setdefault('reference_aliases', []).append(key)
                else:
                    add(symbol, operand, phase, key, layer,
                        'projection_heads' if '.weight_' in name else 'identity')
        if any(found != set(norms) for found in norm_layers.values()):
            raise ValueError('missing selected layer norm boundary')

        present = {r['node']: r for r in report['w8a8_calls']
                   if 'qwen_graph_' + r['node'] in calls}
        seen = set()
        linear_layers = {l: set() for l in layers}
        for linear in report['w8a8_linears']:
            node = linear['node']
            weight = weights.get(linear['weight_param'], '')
            match = re.fullmatch(r'model.layers.(\d+)\.(.*)', weight)
            if not match or int(match[1]) not in layers:
                continue
            layer = int(match[1])
            mc = [r for n, r in present.items() if n.startswith('w8a8_' + node + '_matmul_')]
            dc = [r for n, r in present.items() if n.startswith('w8a8_' + node + '_dequantize_')]
            if not mc and not dc:
                continue
            if node in seen:
                continue
            seen.add(node)
            if len(mc) != 1 or len(dc) != 1:
                raise ValueError('missing/ambiguous selected linear call')
            mc, dc = mc[0], dc[0]
            if (mc['operands'][1] != linear['weight_param_name'] or
                    dc['operands'][2] != linear['scale_param_name'] or
                    mc['operands'][2] != dc['operands'][0]):
                raise ValueError('selected linear weight/accumulator dataflow mismatch')
            qc = [r for r in present.values() if r['case'].startswith('quantize_')
                  and r['operands'][1:3] == [mc['operands'][0], dc['operands'][1]]]
            if len(qc) != 1:
                raise ValueError('missing/ambiguous shared quantize producer')
            qc = qc[0]
            symbols = ['_mlir_ciface_qwen_graph_' + r['node'] for r in (qc, mc, dc)]
            q_operands, m_operands, d_operands = [calls[s.removeprefix('_mlir_ciface_')]['operands']
                                                 for s in symbols]
            if (m_operands[0] != q_operands[1] or d_operands[1] != q_operands[2]
                    or m_operands[2] != d_operands[0]):
                raise ValueError('typed graph shared quantize/accumulator SSA mismatch')
            for symbol, row in zip(symbols, (qc, mc, dc)):
                verify(symbol, '_mlir_ciface_kernel_' + row['case'])
            verify_parameter(symbols[1], 1, linear['weight_param_name'])
            verify_parameter(symbols[2], 2, linear['scale_param_name'])
            add(symbols[0], 0, 'before', weight + '_activation', layer)
            add(symbols[1], 0, 'before', weight + '_q', layer)
            add(symbols[2], 1, 'before', weight + '_a_scale', layer)
            add(symbols[2], 3, 'after', weight + '_out', layer)
            linear_layers[layer].add(match[2])
        expected = {'self_attn.' + n + '_proj.weight' for n in ('q', 'k', 'v', 'o')}
        expected |= {'mlp.' + n + '_proj.weight' for n in ('gate', 'up', 'down')}
        if any(found != expected for found in linear_layers.values()):
            raise ValueError('missing selected layer linear boundary')
    return entries, declarations, sources


RUNTIME = r'''
#include "support.h"
#include "nr_runtime.h"
extern const float intermediate_reference_raw[];
typedef struct { unsigned kind, rank, dtype, count; int64_t shape[4]; unsigned offset[9]; } Probe;
static unsigned current_kind, current_step, current_position, invocation, failed, active_valid;
static unsigned seen[PROBE_COUNT], checked[PROBE_COUNT];
static float maxima[PROBE_COUNT], means[PROBE_COUNT];
static void bits(float value) { union { float f; uint32_t u; } v={value}; nr_hex32(v.u); }
void qwen_intermediate_begin(unsigned position, unsigned length) {
  failed = 0;
  current_kind = length == PREFILL_LENGTH ? 0 : 1;
  current_step = current_kind ? position-PREFILL_LENGTH : 0;
  current_position = position;
  if ((invocation == 0 && (position != 0 || length != PREFILL_LENGTH)) ||
      (invocation != 0 && (position != PREFILL_LENGTH+invocation-1 || length != 1)) ||
      invocation > DECODE_STEPS || current_step >= 8) failed = 1;
  active_valid = !failed;
  for(unsigned i=0;i<PROBE_COUNT;i++) {seen[i]=checked[i]=0;maxima[i]=means[i]=0;}
}
static void check(unsigned index, const void *descriptor) {
  if(index>=PROBE_COUNT) {failed=1;return;}
  const Probe *p = &probes[index];
#if PROBE_PROGRESS
  nr_puts("[probe] entry=");nr_hex32(index);
  nr_puts(" rank=");nr_hex32(p->rank);
  nr_puts(" descriptor=");nr_hex64((uintptr_t)descriptor);
  nr_puts("\r\n");
#endif
  if(p->rank<1 || p->rank>4) {failed=1;return;}
  if(p->kind != current_kind) return;
  if (++seen[index] != 1 || !active_valid || !descriptor) { failed=1; return; }
  void *aligned; int64_t offset, size[4], stride[4];
  memcpy(&aligned, (const char*)descriptor+8, 8);
  memcpy(&offset, (const char*)descriptor+16, 8);
  memcpy(size, (const char*)descriptor+24, p->rank*8);
  memcpy(stride, (const char*)descriptor+24+p->rank*8, p->rank*8);
  if(!aligned || offset<0 || offset>0x10000000) {failed=1;return;}
  for(unsigned j=0;j<p->rank;j++)
    if(size[j]!=p->shape[j] || stride[j]<0 || stride[j]>0x10000000) {failed=1;return;}
  const float *gold=intermediate_reference_raw+p->offset[current_step];
  double sum=0; float max=0;
  for(unsigned i=0;i<p->count;i++) {
    unsigned flat=i; int64_t at=offset;
    for(unsigned j=p->rank;j>0;j--) {at += (flat%size[j-1])*stride[j-1];flat/=size[j-1];}
    float value=p->dtype ? (float)((const int8_t*)aligned)[at] : ((const float*)aligned)[at];
    float diff=value-gold[i];if(diff<0)diff=-diff;
    if(!(value<=3.402823466e38f && value>=-3.402823466e38f && diff<=3.402823466e38f)) {failed=1;return;}
    if(diff>max)max=diff;sum+=diff;
  }
  checked[index]=1;maxima[index]=max;means[index]=(float)(sum/p->count);
  if(max>MAX_ATOL || means[index]>MEAN_ATOL) failed=1;
}
int qwen_intermediate_end(void) {
  for(unsigned i=0;i<PROBE_COUNT;i++) if(probes[i].kind==current_kind) {
    if(seen[i]!=1 || checked[i]!=1)failed=1;
    nr_puts("[intermediate] position=");nr_hex32(current_position);
    nr_puts(" entry=");nr_hex32(i);nr_puts(" calls=");nr_hex32(seen[i]);
    nr_puts(" checked=");nr_hex32(checked[i]);
    nr_puts(" count=");nr_hex32(probes[i].count);
    nr_puts(" max_abs_bits=");bits(maxima[i]);
    nr_puts(" mean_abs_bits=");bits(means[i]);nr_puts("\r\n");
  }
  nr_puts("[intermediate] complete position=");nr_hex32(current_position);
  nr_puts(failed ? " FAIL\r\n" : " PASS\r\n");
  invocation++;
  return failed ? 1 : 0;
}
'''


def generate_intermediate(args, output):
    import numpy as np
    report = json.loads(args.report.read_text())
    if not 1 <= args.decode_steps <= 8 or args.prefill_len == 1:
        raise ValueError('intermediate probe requires P>1 and 1..8 decode calls')
    for value in (args.intermediate_atol, args.intermediate_mean_atol):
        if not math.isfinite(value) or value < 0:
            raise ValueError('intermediate tolerances must be finite and nonnegative')
    metadata_path = args.intermediate_arrays.with_name('quant-reference.json')
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get('layers') != args.layers or metadata.get('capacity') != args.cache_len or
            metadata.get('prompt_ids') != args.prompt_ids or metadata.get('arithmetic_profile') != 'nr-fpga'):
        raise ValueError('intermediate reference identity differs from image')
    entries, declarations, sources = build_mapping(args, report)
    values = []
    cursor = 0
    with np.load(args.intermediate_arrays) as arrays:
        for entry in entries:
            entry['offsets'] = []
            entry['reference_keys'] = []
            for step in range(args.decode_steps if entry['graph']=='decode' else 1):
                prefix = 'prefill' if entry['graph']=='prefill' else f'decode_{step}'
                key = prefix + '_' + entry['reference_suffix']
                if key not in arrays:
                    raise ValueError('missing intermediate reference ' + key)
                value = arrays[key]
                for alias in entry.get('reference_aliases', []):
                    alias_key = prefix + '_' + alias
                    if alias_key not in arrays or not np.array_equal(value, arrays[alias_key]):
                        raise ValueError('shared probe references differ: ' + alias_key)
                if entry['dtype']=='i8' and value.dtype!=np.int8:
                    raise ValueError('quantized intermediate must be int8: ' + key)
                if entry['dtype']=='f32' and value.dtype!=np.float32:
                    raise ValueError('float intermediate must be float32: ' + key)
                transform = entry['transform']
                if transform=='last_row':value=value[-1:]
                elif transform=='heads_first':value=value.transpose(1,0,2)
                elif transform=='context_heads_first':value=value.reshape(value.shape[0],16,128).transpose(1,0,2)
                elif transform=='projection_heads':
                    shape=entry['shape']
                    if value.shape != (shape[1],shape[2]*shape[3]):
                        raise ValueError('projection head reshape mismatch: ' + key)
                    value=value.reshape(shape[1:])
                shape=tuple(entry['shape'])
                if (value.shape != shape and not (shape[0]==1 and value.shape==shape[1:])):
                    raise ValueError('intermediate shape/layout mismatch: ' + key)
                if value.size!=entry['elements'] or not np.isfinite(value).all():
                    raise ValueError('intermediate extent/finite mismatch: ' + key)
                # Leading batch=1 and reshape-only linear->head views retain order.
                entry['reference_keys'].append(key)
                entry['offsets'].append(cursor)
                values.append(value.astype('<f4').ravel());cursor+=value.size
        covered={k for e in entries for k in e['reference_keys']}
        covered.update(k.removesuffix(e['reference_suffix']) + alias
                       for e in entries for k in e['reference_keys']
                       for alias in e.get('reference_aliases', []))
        uncovered=sorted(k for k in arrays.files if (k.startswith('prefill_model.') or k.startswith('decode_')) and '_logits' not in k and '_kv_' not in k and k not in covered)
    blob=output/'intermediate-reference.bin'
    blob.write_bytes(np.concatenate(values).astype('<f4').tobytes())
    asm=output/'intermediate-reference.S'
    asm.write_text('.section .rodata.intermediate_reference,"a",@progbits\n.balign 64\n'
                   '.globl intermediate_reference_raw\nintermediate_reference_raw:\n.incbin '+json.dumps(str(blob.resolve()))+'\n')
    table=['static const Probe probes[PROBE_COUNT] = {']
    for entry in entries:
        table.append('  {'+','.join(map(str,[int(entry['graph']=='decode'),entry['rank'],int(entry['dtype']=='i8'),entry['elements']]))+
                     ',{'+','.join(map(str,entry['shape']))+'},{'+','.join(map(str,entry['offsets']))+'}},')
    table.append('};')
    source=RUNTIME.replace('static unsigned current_kind', '\n'.join(table)+'\nstatic unsigned current_kind')
    source=(f'#define PROBE_COUNT {len(entries)}\n#define PREFILL_LENGTH {args.prefill_len}\n#define DECODE_STEPS {args.decode_steps}\n'
            f'#define MAX_ATOL {args.intermediate_atol:.9e}f\n#define MEAN_ATOL {args.intermediate_mean_atol:.9e}f\n'
            f'#define PROBE_PROGRESS {int(getattr(args, "intermediate_progress", False))}\n'+source)
    flags=[]
    for symbol in sorted({e['symbol'] for e in entries}):
        params=', '.join(f'MemRef{rank} *a{i}' for i,rank in enumerate(declarations[symbol]))
        call=', '.join(f'a{i}' for i in range(len(declarations[symbol])))
        source+=f'extern void __real_{symbol}({params});\nvoid __wrap_{symbol}({params}) {{\n'
        kinds = {int(e['graph']=='decode') for e in entries if e['symbol']==symbol}
        if len(kinds)==1:
            source+=f'  if(current_kind != {next(iter(kinds))}) failed=1;\n'
        for phase in ('before','after'):
            if phase=='after':source+=f'  __real_{symbol}({call});\n  ame_fence();\n'
            for entry in entries:
                if entry['symbol']==symbol and entry['phase']==phase:
                    source+=f'  check({entry["index"]}, a{entry["operand"]});\n'
        source+='}\n';flags.append('--wrap='+symbol)
    path=output/'intermediate-probe.c';path.write_text(source)
    manifest={'schema_version':1,'layers':1,'prefill_len':args.prefill_len,'decode_steps':args.decode_steps,
              'max_abs_tolerance':args.intermediate_atol,'mean_abs_tolerance':args.intermediate_mean_atol,
              'entries':entries,'uncovered_reference_tensors':uncovered,'reference_bytes':blob.stat().st_size,
              'source_sha256':{**sources,str(args.report):sha(args.report),str(args.adapters):sha(args.adapters),
                               str(args.intermediate_layout):sha(args.intermediate_layout),
                               str(args.intermediate_arrays):sha(args.intermediate_arrays),str(metadata_path):sha(metadata_path)},
              'generated_source_sha256':sha(path),'reference_blob_sha256':sha(blob),'linker_flags':flags,
              'scope':'full tensors at verified external adapter boundaries; one decoder layer only',
              'limitations':['q/k/v linears covered, but int32 accumulator not captured',
                             'K RoPE validated by existing full KV comparator; no direct intermediate hook',
                             'SwiGLU captured as down projection activation; final norm captured at lm_head input',
                             'comparison and synchronization overhead included in instrumented graph cycles']}
    if getattr(args, 'intermediate_layers', None) is not None:
        manifest.update(schema_version=2, layers=args.layers,
                        coverage_includes_aliases=True,
                        selected_layers=selected_layers(args),
                        scope='Selected full norm/linear tensors within the compiled full model; partial intermediate coverage',
                        limitations=['Attention/RoPE/SiLU are only observed through adjacent norm/linear boundaries',
                                     'Int32 accumulators are not captured; int8 and scales checked at each linear consumer',
                                     'Shared activation references are checked for equality before packing',
                                     'Each selected adapter is called once, followed by AME resync and comparison; instrumentation changes timing and layout'])
    (output/'intermediate-probe.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return path, asm, flags, manifest


def generate_boundary_sync(args, output):
    """Retain a verified probe's call boundaries without its oracle or reads."""
    from check_intermediates import validate_manifest
    prior_path = args.boundary_sync_manifest
    prior = json.loads(prior_path.read_text())
    validate_manifest(prior)
    if prior['schema_version'] != 2:
        raise ValueError('boundary sync requires a selected-layer schema v2 manifest')
    if (prior['layers'], prior['prefill_len'], prior['decode_steps']) != (
            args.layers, args.prefill_len, args.decode_steps):
        raise ValueError('boundary sync manifest dimensions differ from image')
    selected = getattr(args, 'intermediate_layers', None)
    args.intermediate_layers = ','.join(map(str, prior['selected_layers']))
    try:
        entries, declarations, sources = build_selected_mapping(args, json.loads(args.report.read_text()))
    finally:
        args.intermediate_layers = selected
    # Compare semantic entries, allowing only the reference packing metadata
    # added by generate_intermediate. No name-only boundary substitution.
    unpacked = [{k: v for k, v in entry.items() if k not in ('offsets', 'reference_keys')}
                for entry in prior['entries']]
    if entries != unpacked:
        raise ValueError('boundary sync prior mapping differs from current typed graph')
    sources.update({str(args.report): sha(args.report), str(args.adapters): sha(args.adapters),
                    str(args.intermediate_layout): sha(args.intermediate_layout)})
    previous_sources = {Path(name).resolve(): expected
                        for name, expected in prior['source_sha256'].items()}
    for name, expected in sources.items():
        if previous_sources.get(Path(name).resolve()) != expected:
            raise ValueError('boundary sync input differs from prior probe: ' + name)
    source = '#include "support.h"\n#include "nr_runtime.h"\n'
    flags, symbols = [], sorted({entry['symbol'] for entry in entries})
    for symbol in symbols:
        params = ', '.join(f'MemRef{rank} *a{i}' for i, rank in enumerate(declarations[symbol]))
        operands = ', '.join(f'a{i}' for i in range(len(declarations[symbol])))
        source += (f'extern void __real_{symbol}({params});\n'
                   f'void __wrap_{symbol}({params}) {{\n'
                   f'  __real_{symbol}({operands});\n  ame_fence();\n}}\n')
        flags.append('--wrap=' + symbol)
    path = output / 'boundary-sync.c'
    path.write_text(source)
    (output / 'boundary-sync-selection.json').write_bytes(prior_path.read_bytes())
    manifest = {'schema_version': 1, 'layers': args.layers, 'selected_layers': prior['selected_layers'],
                'prefill_len': args.prefill_len, 'decode_steps': args.decode_steps,
                'symbols': symbols, 'linker_flags': flags, 'completion_sync': 'ame-resync',
                'source_sha256': {**sources, str(prior_path): sha(prior_path)},
                'generated_source_sha256': sha(path),
                'scope': 'Selected adapter return synchronization; no intermediate numerical comparison',
                'limitations': ['No intermediate oracle, tensor reads, per-kernel UART or timing in these wrappers',
                                'Removing the probe changes image layout, traffic and timing; this is not a layout-controlled experiment']}
    (output / 'boundary-sync.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return path, flags, manifest
