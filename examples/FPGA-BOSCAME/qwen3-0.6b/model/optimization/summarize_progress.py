#!/usr/bin/env python3
"""Map partial --profile-progress UART to verified compiled graph boundaries.

This reads local artifacts only. An unmatched begin means the corresponding end
has not been observed, never that the device is hung. It is not an acceptance
checker; use archive_model_run.py for completed numerical runs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_kernel_profile import (  # noqa: E402
    PREFIX, SYMBOL, STAGE, adapter_calls, graph_calls, llvm_calls,
    llvm_function, require,
)

BEGIN = re.compile(r'\[model\] (prefill|decode) begin position=([0-9A-Fa-f]{8}) '
                   r'input_token=([0-9A-Fa-f]{8})')
KERNEL = re.compile(r'\[kernel\] (begin|end) (' + SYMBOL +
                    r') call=([0-9A-Fa-f]{16})')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_calls(body):
    """Prove that textual call order agrees with control flow, not just counts."""
    proven_counts = graph_calls(body)  # Reject conditional, cyclic call sites.
    blocks, current = {'entry': []}, 'entry'
    for original in body.splitlines():
        line = original.split(';', 1)[0].strip()
        if not line:
            continue
        label = re.fullmatch(r'([A-Za-z_0-9.$-]+):', line)
        if label:
            current = label[1]
            blocks[current] = []
        else:
            blocks[current].append(line)
    blocks = {name: lines for name, lines in blocks.items() if lines}
    edges = {name: re.findall(r'label %([A-Za-z_0-9.$-]+)', lines[-1])
             for name, lines in blocks.items()}
    call_blocks = [(name, [c for c in llvm_calls('\n'.join(lines))
                          if c.startswith('qwen_graph_')])
                   for name, lines in blocks.items()]
    call_blocks = [(name, calls) for name, calls in call_blocks if calls]

    def reaches(start, target):
        seen, pending = set(), list(edges[start])
        while pending:
            name = pending.pop()
            if name == target:
                return True
            if name not in seen:
                seen.add(name)
                pending.extend(edges[name])
        return False

    for (before, _), (after, _) in zip(call_blocks, call_blocks[1:]):
        require(reaches(before, after), 'textual call order not proved by CFG')
    calls = [call for _, names in call_blocks for call in names]
    require(Counter(calls) == proven_counts, 'CFG call sequence/count mismatch')
    return calls


def expected_sequence(build, layers, profile=None):
    report_path = build / 'replacement/triton-call-replacement.json'
    adapter_path = build / 'replacement/qwen_triton_adapters.c'
    report = json.loads(report_path.read_text())
    adapters = adapter_calls(adapter_path.read_text())
    require(set(adapters) == set(report['distinct_symbols']),
            'report and adapters differ')
    if profile:
        manifest = json.loads(profile.read_text())
        require(manifest['adapter_sha256'] == sha(adapter_path),
                'profile adapter hash differs from selected build')
        require(manifest.get('progress_uart'), 'profile has no progress UART')
    sequences, sources = {}, {str(p): sha(p) for p in (report_path, adapter_path)}
    if profile:
        sources[str(profile)] = sha(profile)
    for kind in ('prefill', 'decode'):
        path = build / f'nr-{kind}/forward_{kind}.ll'
        sources[str(path)] = sha(path)
        ir = path.read_text()
        raw = ordered_calls(llvm_function(ir, 'forward_' + kind))
        require(len(raw) == report['graphs'][kind]['external_calls'],
                kind + ': LLVM call count differs from report')
        bridges = {}
        for match in re.finditer(r'^define [^\n]*@(qwen_graph_\w+)\([^\n]*\{\n', ir, re.M):
            end = ir.find('\n}', match.end())
            require(end >= 0 and match[1] not in bridges,
                    'unterminated or duplicate RAW bridge: ' + match[1])
            bridges[match[1]] = ir[match.end():end]
        sequence, occurrences = [], Counter()
        for name in raw:
            require(name in bridges, 'missing RAW bridge: ' + name)
            bridge = bridges[name]
            require(llvm_calls(bridge) == ['_mlir_ciface_' + name],
                    'unexpected RAW-to-CIFACE bridge: ' + name)
            require(not re.search(r'^\s*(br|switch|invoke|indirectbr|callbr)\b', bridge, re.M),
                    'non-straight-line RAW-to-CIFACE bridge: ' + name)
            kernels = adapters[name]
            require(sum(kernels.values()) == 1,
                    'progress mapper requires exactly one kernel per adapter')
            kernel = next(iter(kernels))
            sequence.append({'index': len(sequence), 'adapter': name,
                             'kernel': kernel, 'symbol_call_index': occurrences[kernel]})
            occurrences[kernel] += 1
        names = [entry['kernel'].removeprefix(PREFIX) for entry in sequence]
        # The mapping is accepted only when the actual graph has exactly this
        # repeated decoder structure. No numerical suffix in a symbol is used
        # as a layer index. The block index is zero based, in execution order.
        require(names[0].startswith('embedding_w8a8_'), 'missing leading embedding')
        require(len(names) == 1 + layers * 34 + 4,
                'expected embedding + layers * 34 calls + final norm/lm_head')
        block = names[1:35]
        require(block[0].startswith('rmsnorm_') and
                block[-2].startswith('matmul_') and
                block[-1].startswith('dequantize_'), 'unexpected decoder boundaries')
        require(sum(n.startswith('matmul_') for n in block) == 7 and
                sum(n.startswith('kv_cache_update_') for n in block) == 2 and
                sum(n.startswith('attention_qk_') for n in block) == 1 and
                sum(n.startswith('attention_pv_') for n in block) == 1 and
                sum(n.startswith('rmsnorm_') for n in block) == 4 and
                sum(n.startswith('silu_') for n in block) == 1,
                'unexpected decoder kernel structure')
        for layer in range(layers):
            require(names[1 + layer * 34:1 + (layer + 1) * 34] == block,
                    'decoder kernel sequences are not identical')
        require(names[-4].startswith('rmsnorm_') and
                names[-3:] == ['quantize_1x1024', 'matmul_1x151936x1024',
                               'dequantize_1x151936'], 'unexpected final head')
        for entry in sequence:
            index = entry['index']
            entry['region'] = ('embedding' if index == 0 else
                               'decoder' if index <= layers * 34 else 'final_norm_lm_head')
            if entry['region'] == 'decoder':
                entry['decoder_block_index'] = (index - 1) // 34
                entry['call_in_decoder_block'] = (index - 1) % 34
        sequences[kind] = sequence
    return sequences, sources


def summarize(log, sequences):
    stages, errors, active, pending = [], [], None, None
    runtime = None
    # A growing UART log may end in a partial line. Do not diagnose that line.
    complete = log.rsplit('\n', 1)[0] if not log.endswith('\n') else log
    for number, line in enumerate(complete.splitlines(), 1):
        if m := BEGIN.fullmatch(line):
            if active is not None and not active['graph_return_observed']:
                errors.append(f'new graph before previous return at line {number}')
            active = {'kind': m[1], 'position': int(m[2], 16),
                      'completed_kernel_calls': 0,
                      'expected_kernel_calls': len(sequences[m[1]]),
                      'graph_return_observed': False}
            stages.append(active)
            pending = None
        elif m := KERNEL.fullmatch(line):
            if active is None:
                errors.append(f'kernel outside graph at line {number}')
                continue
            phase, symbol, occurrence = m[1], m[2], int(m[3], 16)
            seq = sequences[active['kind']]
            index = active['completed_kernel_calls']
            if phase == 'begin':
                if pending is not None or index >= len(seq):
                    errors.append(f'unexpected kernel begin at line {number}')
                    continue
                expected = seq[index]
                if (symbol, occurrence) != (expected['kernel'], expected['symbol_call_index']):
                    errors.append(f'kernel sequence mismatch at line {number}: expected {expected}')
                pending = {**expected, 'uart_line': number}
            elif pending is None or (symbol, occurrence) != (pending['kernel'], pending['symbol_call_index']):
                errors.append(f'unmatched kernel end at line {number}')
            else:
                active['completed_kernel_calls'] += 1
                active['last_completed_kernel'] = pending
                pending = None
        elif m := STAGE.fullmatch(line):
            if active is None or (m[1], int(m[2], 16)) != (active['kind'], active['position']):
                errors.append(f'unmatched graph return at line {number}')
            else:
                active['graph_return_observed'] = True
                active['compute_cycles'] = int(m[5], 16)
                if pending or active['completed_kernel_calls'] != active['expected_kernel_calls']:
                    errors.append(f'graph returned without complete kernel sequence at line {number}')
        elif line.startswith('[kernel]') or '[model]' in line and ' begin position=' in line:
            errors.append(f'malformed progress record at line {number}')
        if line.startswith('[nr] RA returned:'):
            runtime = line
    return {'status': 'PROGRESS_SEQUENCE_ERROR' if errors else 'PROGRESS_SEQUENCE_VALID_SO_FAR',
            'errors': errors, 'graph_stages': stages, 'unpaired_begin': pending,
            'runtime_completion': runtime,
            'limits': [
                'Progress only; no numerical correctness or model acceptance claim.',
                'An unpaired begin means end has not been observed, not a hang diagnosis.',
                'UART kernel rows have no timestamps; per-kernel elapsed wall time is unknown.',
                'A kernel end follows the profiler-added public ame_fence and bookkeeping.',
                'Graph cycles include diagnostic UART and per-kernel synchronization.',
                'Decoder block indices are zero based and derived from verified repeated compiled-call structure.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--uart-log', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(args.layers > 0, 'layers must be positive')
    sequences, sources = expected_sequence(args.build, args.layers, args.profile)
    # Read once: the runner may append to the log while the parser runs.
    uart_bytes = args.uart_log.read_bytes()
    result = summarize(uart_bytes.decode(errors='replace'), sequences)
    result.update({'sources_sha256': sources, 'uart_log': str(args.uart_log),
                   'uart_sha256': hashlib.sha256(uart_bytes).hexdigest(),
                   'uart_bytes': len(uart_bytes), 'layers': args.layers,
                   'snapshot_utc': datetime.now(timezone.utc).isoformat()})
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')
    return bool(result['errors'])


if __name__ == '__main__':
    sys.exit(main())
