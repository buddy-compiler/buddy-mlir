#!/usr/bin/env python3
"""Validate FPGA kernel timing against actual LLVM entry calls and ABI adapters.

This is a timing/count check, not numerical acceptance. Only unconditional,
acyclic graph call sites and straight-line generated C adapters are supported.
Unsupported control flow fails instead of guessing a dynamic invocation count.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

PREFIX = '_mlir_ciface_kernel_'
SYMBOL = r'[A-Za-z_][A-Za-z_0-9]*'
PROFILE = re.compile(r'\[profile\] position=([0-9A-Fa-f]{8}) kernel=(' + SYMBOL +
                     r') calls=([0-9A-Fa-f]{16}) cycles=([0-9A-Fa-f]{16})')
STAGE = re.compile(r'\[model\] (prefill|decode) position=([0-9A-Fa-f]{8}) '
                   r'token=([0-9A-Fa-f]{8}) logit_bits=([0-9A-Fa-f]{8}) '
                   r'compute_cycles=([0-9A-Fa-f]{16})'
                   r'(?: preparation_cycles=[0-9A-Fa-f]{16}'
                   r' selection_cycles=[0-9A-Fa-f]{16}'
                   r' cache_retention_cycles=[0-9A-Fa-f]{16}'
                   r' model_cycles=[0-9A-Fa-f]{16})?'
                   r'(?: total_with_uart_cycles=[0-9A-Fa-f]{16})?'
                   r'(?: scratch_bytes=[0-9A-Fa-f]{16})?')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def llvm_function(text, name):
    headers = list(re.finditer(r'^define [^\n]*@' + re.escape(name) +
                              r'\([^\n]*\{\n', text, re.M))
    require(len(headers) == 1, f'expected one single-line LLVM definition of {name}')
    end = re.search(r'^}', text[headers[0].end():], re.M)
    require(end is not None, f'unterminated LLVM definition {name}')
    return text[headers[0].end():headers[0].end() + end.start()]


def llvm_calls(body):
    calls = []
    for line in body.splitlines():
        line = line.split(';', 1)[0]
        if re.search(r'\b(?:call|invoke|callbr)\b', line):
            m = re.search(r'\bcall\b[^@\n]*@(' + SYMBOL + r')\(', line)
            # LLVM intrinsic names contain dots; only graph/kernel references
            # need the restricted stable-symbol parser.
            if m:
                calls.append(m[1])
            elif 'qwen_graph_' in line or '_mlir_ciface_kernel_' in line:
                raise ValueError('unsupported indirect/quoted graph call: ' + line.strip())
    return calls


def graph_calls(body):
    """Count entry calls only after proving their blocks execute once per return."""
    blocks = {'entry': []}
    current = 'entry'
    for original in body.splitlines():
        line = original.split(';', 1)[0].strip()
        if not line:
            continue
        label = re.fullmatch(r'([A-Za-z_0-9.$-]+):', line)
        if label:
            current = label[1]
            require(current not in blocks, 'duplicate LLVM block ' + current)
            blocks[current] = []
        else:
            blocks[current].append(line)
    # An explicitly named first entry has no instructions in our sentinel.
    if not blocks['entry']:
        del blocks['entry']
    require(bool(blocks), 'empty graph body')
    entry = next(iter(blocks))
    edges, returns, call_blocks = {}, set(), {}
    for label, lines in blocks.items():
        require(bool(lines), 'empty LLVM block ' + label)
        term = lines[-1]
        if term.startswith('br '):
            edges[label] = re.findall(r'label %([A-Za-z_0-9.$-]+)', term)
            require(len(edges[label]) in (1, 2), 'unsupported LLVM branch ' + term)
        elif term.startswith('ret '):
            edges[label] = []
            returns.add(label)
        else:
            raise ValueError('unsupported LLVM terminator: ' + term)
        names = llvm_calls('\n'.join(lines))
        require(not any(n.startswith('_mlir_ciface_qwen_graph_') or n.startswith(PREFIX)
                        for n in names), 'entry bypasses expected RAW-to-CIFACE bridge')
        call_blocks[label] = [n for n in names if n.startswith('qwen_graph_')]
    require(returns, 'graph has no normal return')
    require(all(n in blocks for e in edges.values() for n in e), 'unknown LLVM branch target')

    def reachable(start, forbidden=None):
        seen, pending = set(), list(start)
        while pending:
            n = pending.pop()
            if n == forbidden or n in seen:
                continue
            seen.add(n)
            pending.extend(edges[n])
        return seen

    reachable_all = reachable([entry])
    calls = []
    for label, names in call_blocks.items():
        if not names:
            continue
        require(label in reachable_all, f'unreachable kernel call block {label}')
        require(label not in reachable(edges[label]),
                f'kernel call in loop; static call count is insufficient: {label}')
        require(not (returns & reachable([entry], forbidden=label)),
                f'conditional kernel call; dynamic count not established: {label}')
        calls += names
    require(bool(calls), 'no RAW qwen graph calls in actual entry')
    return Counter(calls)


def adapter_calls(text):
    # Ignore comments and strings without changing positions; brace matching
    # still handles the descriptor initializers in generated adapters.
    clean = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"',
                   lambda m: ' ' * len(m[0]), text, flags=re.S)
    wrappers = {}
    pattern = re.compile(r'^void _mlir_ciface_(qwen_graph_\w+)\([^;{}]*\)\s*\{', re.M)
    for m in pattern.finditer(clean):
        depth, end = 1, m.end()
        while end < len(clean) and depth:
            depth += (clean[end] == '{') - (clean[end] == '}')
            end += 1
        require(depth == 0, 'unterminated adapter ' + m[1])
        body = clean[m.end():end - 1]
        require(not re.search(r'\b(if|for|while|do|switch|goto|return)\b|\?|&&|\|\|', body),
                'non-straight-line adapter requires dynamic proof: ' + m[1])
        all_calls = re.findall(r'\b(' + SYMBOL + r')\s*\(', body)
        require(all(n.startswith(PREFIX) or re.fullmatch(r'make_[1-4]', n)
                    or n == 'sizeof' for n in all_calls),
                'unsupported adapter helper/call: ' + m[1])
        kernels = Counter(n for n in all_calls if n.startswith(PREFIX))
        require(kernels, 'adapter has no kernel call: ' + m[1])
        require(m[1] not in wrappers, 'duplicate adapter: ' + m[1])
        wrappers[m[1]] = kernels
    require(wrappers, 'no generated ABI adapter definitions')
    return wrappers


def expected_counts(profile, report, adapters, irs):
    require(profile.get('adapter_sha256') == sha256(adapters.encode()),
            'kernel-profile adapter hash mismatch')
    records = profile.get('kernels', [])
    symbols = [r['symbol'] for r in records]
    require(symbols and len(set(symbols)) == len(symbols), 'missing/duplicate profile symbols')
    require(symbols == sorted(symbols) and [r['index'] for r in records] == list(range(len(records))),
            'profile symbol order/index mismatch')
    require(all(re.fullmatch(PREFIX + r'\w+', n) for n in symbols), 'invalid kernel symbol')
    require(profile.get('linker_flags') == ['--wrap=' + n for n in symbols],
            'profile linker wrappers do not match kernel list')
    wrappers = adapter_calls(adapters)
    declared = report.get('distinct_symbols')
    require(isinstance(declared, list) and len(set(declared)) == len(declared)
            and set(declared) == set(wrappers),
            'replacement distinct_symbols differ from actual adapters')
    expected, provenance = {}, {}
    for kind, ir in irs.items():
        calls = graph_calls(llvm_function(ir, 'forward_' + kind))
        graph = report['graphs'][kind]
        require(sum(calls.values()) == graph.get('external_calls'),
                f'{kind}: actual entry call count differs from replacement report')
        kernels = Counter()
        for wrapper, count in calls.items():
            require(wrapper in wrappers, 'RAW callee missing from adapters/report: ' + wrapper)
            bridge_body = llvm_function(ir, wrapper)
            require(not re.search(r'^\s*(br|switch|invoke|indirectbr|callbr)\b',
                                  bridge_body, re.M),
                    'non-straight-line RAW-to-CIFACE bridge: ' + wrapper)
            bridge = llvm_calls(bridge_body)
            require(bridge == ['_mlir_ciface_' + wrapper],
                    'unexpected RAW-to-CIFACE bridge: ' + wrapper)
            for name, multiplier in wrappers[wrapper].items():
                require(name in symbols, 'kernel not instrumented: ' + name)
                kernels[name] += count * multiplier
        expected[kind] = dict(sorted(kernels.items()))
        provenance[kind] = {'entry': 'forward_' + kind,
                            'raw_call_sites': dict(sorted(calls.items())),
                            'total_raw_calls': sum(calls.values()),
                            'expected_kernel_calls': sum(kernels.values()),
                            'unconditional_acyclic_calls_verified': True}
    return expected, provenance


def check(log, expected, *, prefill=16, steps=8):
    require(prefill > 0 and steps >= 0, 'invalid prefill/decode count')
    stage_rows, rows, errors = [], {}, []
    active = None
    completed = False
    for number, line in enumerate(log.splitlines(), 1):
        if line == '[nr] RA returned: PASS':
            completed = True
        if completed and (line.startswith('[profile]') or STAGE.fullmatch(line)):
            errors.append(f'graph/profile record after runtime completion at line {number}')
        m = STAGE.fullmatch(line)
        if m:
            kind, pos, _token, _bits, cycles = m.groups()
            pos, cycles = int(pos, 16), int(cycles, 16)
            output = pos + prefill - 1 if kind == 'prefill' else pos
            stage_rows.append({'kind': kind, 'graph_start_position': pos,
                               'position': output, 'compute_cycles': cycles})
            phases = {name: int(value, 16) for name, value in re.findall(
                r'\b(preparation_cycles|selection_cycles|cache_retention_cycles|model_cycles|'
                r'total_with_uart_cycles|scratch_bytes)=([0-9A-Fa-f]{16})', line)}
            stage_rows[-1].update(phases)
            if 'model_cycles' in phases:
                if phases['model_cycles'] != cycles + sum(phases[n] for n in
                        ('preparation_cycles', 'selection_cycles', 'cache_retention_cycles')):
                    errors.append(f'model phase sum differs from model_cycles at {output}')
                if phases.get('total_with_uart_cycles', 0) < phases['model_cycles']:
                    errors.append(f'model_cycles exceed total_with_uart_cycles at {output}')
            if output in rows:
                errors.append(f'duplicate graph output position {output}')
            rows.setdefault(output, [])
            active = output
            if cycles <= 0:
                errors.append(f'nonpositive graph compute_cycles at {output}')
        elif re.match(r'\[model\] (prefill|decode) position=', line):
            errors.append(f'malformed graph timing row at line {number}')
        elif '[profile]' in line:
            m = PROFILE.fullmatch(line)
            if not m:
                errors.append(f'malformed profile row at line {number}')
                continue
            pos, symbol, calls, cycles = m.groups()
            pos, calls, cycles = int(pos, 16), int(calls, 16), int(cycles, 16)
            if active is None or pos != active:
                errors.append(f'profile outside corresponding graph block at line {number}')
            rows.setdefault(pos, []).append({'symbol': symbol, 'calls': calls, 'cycles': cycles})
            if calls <= 0 or cycles <= 0:
                errors.append(f'nonpositive calls/cycles at line {number}')
    wanted = [('prefill', 0)] + [('decode', p) for p in range(prefill, prefill + steps)]
    if [(r['kind'], r['graph_start_position']) for r in stage_rows] != wanted:
        errors.append('missing/duplicate/out-of-order graph stages')
    if len(re.findall(r'^\[nr\] RA returned: PASS\r?$', log, re.M)) != 1:
        errors.append('missing/duplicate NR PASS completion')
    if re.search(r'\[nr\][^\r\n]*(?:FAIL|TRAP)|TRAP mcause=|verify[^\r\n]*:\s*FAIL', log):
        errors.append('runtime failure')
    for status in re.findall(r'\[nr\] launch[^\r\n]* status=0x([0-9A-Fa-f]+)', log):
        if int(status, 16):
            errors.append('nonzero runtime launch status')
    summary = []
    for stage in stage_rows:
        kind, pos = stage['kind'], stage['position']
        actual = rows.get(pos, [])
        wanted_counts = expected[kind]
        symbols = [row['symbol'] for row in actual]
        if symbols != sorted(wanted_counts):
            errors.append(f'missing/duplicate/unknown/out-of-order kernel symbols at {pos}')
        if any(row['calls'] != wanted_counts.get(row['symbol']) for row in actual):
            errors.append(f'kernel invocation count differs from actual graph at {pos}')
        total = sum(row['cycles'] for row in actual)
        if total > stage['compute_cycles']:
            errors.append(f'kernel cycles exceed graph compute_cycles at {pos}')
        summary.append({**stage, 'kernels': actual,
                        'kernel_calls': sum(row['calls'] for row in actual),
                        'kernel_cycles': total,
                        'graph_cycles_outside_kernel_measurements': stage['compute_cycles'] - total})
    return {'status': 'KERNEL_PROFILE_PASS' if not errors else 'NOT_ACCEPTED',
            'errors': errors, 'stages': summary,
            'limits': ['Timing/count validation only; numerical acceptance requires check_board_trace.py.',
                       'Kernel timing includes the profiler-selected completion sequence; inspect completion_sync in the manifest. Instrumentation changes execution cost.',
                       'Graph minus kernels includes adapter/bookkeeping, allocation, scalar work, copies and synchronization; it is not pure memory time.',
                       'Actual LLVM call sites must be unconditional and acyclic; unsupported dynamic control flow is rejected.']}


def check_progress(log, expected, *, enabled):
    """Pair diagnostic entry/exit records with the surrounding actual graph."""
    pattern = re.compile(r'\[kernel\] (begin|end) (' + SYMBOL + r') call=([0-9A-Fa-f]{16})')
    active, pending, counts = None, None, Counter()
    errors = []
    saw = False
    for line in log.splitlines():
        start = re.fullmatch(r'\[model\] (prefill|decode) begin position=[0-9A-Fa-f]{8} input_token=[0-9A-Fa-f]{8}', line)
        if start:
            if active is not None:
                errors.append('progress: graph began before prior graph ended')
            active, pending, counts = start[1], None, Counter()
        elif line.startswith('[kernel]'):
            saw = True
            record = pattern.fullmatch(line)
            if not enabled or active is None or not record:
                errors.append('progress: unexpected/malformed kernel record')
                continue
            phase, symbol, ordinal = record.groups()
            ordinal = int(ordinal, 16)
            if symbol not in expected[active]:
                errors.append('progress: unknown kernel symbol')
            if phase == 'begin':
                if pending is not None or ordinal != counts[symbol]:
                    errors.append('progress: overlapping/duplicate/out-of-order kernel entry')
                pending = (symbol, ordinal)
            else:
                if pending != (symbol, ordinal):
                    errors.append('progress: unmatched kernel exit')
                counts[symbol] += 1
                pending = None
        elif STAGE.fullmatch(line):
            if enabled and (active is None or pending is not None or counts != expected[active]):
                errors.append('progress: missing/excess kernel calls at graph completion')
            active, pending = None, None
    if enabled and (not saw or active is not None or pending is not None):
        errors.append('progress: missing/incomplete graph execution')
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('uart', 'profile', 'replacement', 'adapters', 'prefill-ir', 'decode-ir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prefill', type=int, default=16)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--profile-source', type=Path,
                   help='defaults to kernel-profile.c alongside --profile')
    a = p.parse_args()
    source = a.profile_source or a.profile.with_suffix('.c')
    paths = [a.uart, a.profile, source, a.replacement, a.adapters, a.prefill_ir, a.decode_ir]
    hashes = {str(path): sha256(path.read_bytes()) for path in paths}
    try:
        profile = json.loads(a.profile.read_text())
        require(profile.get('source_sha256') == hashes[str(source)],
                'profile generated source hash mismatch')
        expected, provenance = expected_counts(profile,
            json.loads(a.replacement.read_text()), a.adapters.read_text(),
            {'prefill': a.prefill_ir.read_text(), 'decode': a.decode_ir.read_text()})
        log = a.uart.read_text(errors='replace')
        report = check(log, expected, prefill=a.prefill, steps=a.steps)
        progress = bool(profile.get('progress_uart', False))
        completion_sync = profile.get('completion_sync', 'ame-resync')
        require(completion_sync in ('ame-resync', 'fence'), 'unknown profiler completion_sync')
        report['completion_sync'] = completion_sync
        report['errors'] += check_progress(log, expected, enabled=progress)
        if report['errors']:
            report['status'] = 'NOT_ACCEPTED'
        report['diagnostic_uart_inside_graph_timing'] = progress
        if progress:
            report['limits'].append('Diagnostic UART is inside graph timing; returned/synced markers, when present, are also inside kernel timing. These are not UART-free throughput measurements.')
        if completion_sync == 'fence':
            report['limits'].append('Experimental profiler fence-only mode: count/timing consistency does not prove AME completion or numerical correctness; production kernels and graph-final synchronization are unchanged.')
        report.update(expected_kernel_calls=expected, expectation_evidence=provenance)
    except (ValueError, KeyError, TypeError) as error:
        report = {'status': 'NOT_ACCEPTED', 'errors': [str(error)],
                  'scope': 'Unable to establish independent expected call counts; no profile acceptance.'}
    report['inputs_sha256'] = hashes
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(a.output)}))
    return 0 if report['status'] == 'KERNEL_PROFILE_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
