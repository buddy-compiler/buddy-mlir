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


def tile_probe_contract(profile):
    """An extra raw wrap is permitted only for this exact diagnostic ABI."""
    if 'tile_probe' not in profile:
        return None
    tile = profile['tile_probe']
    require(isinstance(tile, dict), 'tile probe manifest must be an object')
    symbol = tile.get('symbol', '')
    require(isinstance(symbol, str) and re.fullmatch(
        PREFIX + r'matmul_[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*', symbol),
        'tile probe selected symbol is not supported')
    index = tile.get('call_index')
    require(type(index) is int and 0 <= index <= 0xffffffffffffffff,
            'tile probe call index is invalid')
    require(profile.get('phase_probe') == {'symbol': symbol, 'call_index': index}
            and profile.get('progress_uart') is True,
            'tile probe must match the enabled phase/progress probe')
    records = [record for record in profile.get('kernels', []) if record.get('symbol') == symbol]
    require(len(records) == 1 and [re.sub(r'\s+', '', t) for t in
            records[0].get('argument_types', [])] == ['MemRef2*'] * 3,
            'tile probe high-level ABI must be three MemRef2 pointers')
    raw = 'triton_' + symbol.removeprefix(PREFIX)
    require(tile.get('raw_symbol') == raw and tile.get('linker_flag') == '--wrap=' + raw,
            'tile probe raw symbol/link flag mismatch')
    require(tile.get('raw_return_type') == 'void' and
            tile.get('raw_argument_types') == ['int64_t', 'MemRef0 *'] * 3 + ['int32_t'] * 6 and
            tile.get('raw_argument_names') == ['rank_a', 'a', 'rank_b', 'b', 'rank_c', 'c',
                                               'grid_x', 'grid_y', 'grid_z', 'x', 'y', 'z'],
            'tile probe raw 12-argument ABI mismatch')
    require(tile.get('descriptor_layout_lp64') ==
            {'allocated': 0, 'aligned': 8, 'offset': 16, 'bytes': 24},
            'tile probe MemRef0 layout mismatch')
    return tile


def profile_plan_contract(profile, plan):
    """Bind reported profiler options to the exact image's planned options."""
    require(plan.get('profile_kernels') is True, 'profile manifest absent from enabled image plan')
    require(not plan.get('profile_watch') and not profile.get('nh_watch'),
            'profile NH watch diagnostics are not supported by full profile acceptance')
    completion = plan.get('completion_sync')
    require(completion in ('ame-resync', 'fence') and profile.get('completion_sync') == completion,
            'profile completion_sync differs from image plan')
    progress = plan.get('profile_progress', False)
    require(type(progress) is bool and profile.get('progress_uart') is progress,
            'profile progress configuration differs from image plan')
    selection = plan.get('profile_probe')
    probe = None
    if selection is not None:
        require(isinstance(selection, str), 'image plan profile_probe must be SYMBOL:CALL_INDEX')
        match = re.fullmatch(r'(' + PREFIX + r'\w+):(0[xX][0-9a-fA-F]+|[0-9]+)', selection)
        require(match is not None and progress, 'image plan profile_probe requires progress and a valid selection')
        index = int(match[2], 16 if match[2].lower().startswith('0x') else 10)
        require(index <= 0xffffffffffffffff, 'image plan profile_probe index exceeds uint64')
        probe = {'symbol': match[1], 'call_index': index}
    require(profile.get('phase_probe') == probe, 'profile probe selection differs from image plan')
    tile_enabled = plan.get('profile_tile_probe', False)
    require(type(tile_enabled) is bool and tile_enabled == ('tile_probe' in profile),
            'profile tile configuration differs from image plan')
    tile = tile_probe_contract(profile)
    return {'completion_sync': completion, 'progress_uart': progress, 'phase_probe': probe,
            'tile_probe': ({key: tile[key] for key in ('symbol', 'call_index', 'raw_symbol', 'linker_flag')}
                           if tile else None)}


def verify_profile_build(profile, plan, build_record, source, adapters, *, repo_root=None):
    """A rehashed replacement source cannot stand in for the compiled wrapper."""
    configuration = profile_plan_contract(profile, plan)
    repo_root = repo_root or Path(__file__).resolve().parents[5]
    recorded = build_record.get('input_sha256', {})
    require(isinstance(recorded, dict), 'profile build input hashes missing')
    hashes = {}
    for path, key in ((source, 'source_sha256'), (adapters, 'adapter_sha256')):
        require(path.is_file() and not path.is_symlink(), 'profile source missing/symlink: ' + str(path))
        actual = sha256(path.read_bytes())
        require(profile.get(key) == actual, 'profile manifest source hash mismatch: ' + str(path))
        matching = [value for name, value in recorded.items()
                    if (Path(name) if Path(name).is_absolute() else repo_root / name).resolve() == path.resolve()]
        require(matching == [actual], 'profile build-time input hash missing/mismatched: ' + str(path))
        hashes[key] = actual
    return {'status': 'BUILD_SOURCE_AND_PLAN_VERIFIED', **hashes, 'configuration': configuration}


def tile_probe_grid(tile, text):
    """Prove a fixed grid from the known generated adapter, never from shape."""
    require(tile.get('raw_adapter_sha256') == sha256(text.encode()),
            'tile probe raw adapter hash mismatch')
    compact = re.sub(r'\s+', '', re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S))
    loops = re.findall(r'for\(int32_t([xyz])=0;[xyz]<([1-9][0-9]*);\+\+[xyz]\)', compact)
    require([axis for axis, _ in loops] == ['x', 'y', 'z'],
            'tile probe adapter must have exactly three static ordered grid loops')
    grid = [int(value) for _, value in loops]
    require(all(v <= 0x7fffffff for v in grid), 'tile probe grid exceeds int32 range')
    types = ', '.join(['int64_t, MemRef0 *'] * 3 + ['int32_t'] * 6)
    expected = ['#include "support.h"',
                'typedef struct { void *allocated, *aligned; int64_t offset; } MemRef0;',
                f'extern void {tile["raw_symbol"]}({types});',
                f'void {tile["symbol"]}(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {{']
    for i, size in enumerate((1, 1, 4)):
        expected += [f'void *p{i} = (unsigned char *)a{i}->aligned + a{i}->offset * {size};',
                     f'MemRef0 m{i} = {{p{i}, p{i}, 0}};']
    expected += [f'for (int32_t {axis}=0; {axis}<{value}; ++{axis})'
                 for axis, value in zip('xyz', grid)]
    expected += [f'{tile["raw_symbol"]}(0, &m0, 0, &m1, 0, &m2, '
                 + ', '.join(map(str, grid)) + ', x, y, z);', '}']
    require(compact == re.sub(r'\s+', '', '\n'.join(expected)),
            'tile probe adapter has unsupported body/control flow or grid arguments')
    return grid


def expected_counts(profile, report, adapters, irs):
    require(not profile.get('nh_watch'),
            'profile NH watch diagnostics are not supported by full profile acceptance')
    require(profile.get('adapter_sha256') == sha256(adapters.encode()),
            'kernel-profile adapter hash mismatch')
    records = profile.get('kernels', [])
    symbols = [r['symbol'] for r in records]
    require(symbols and len(set(symbols)) == len(symbols), 'missing/duplicate profile symbols')
    require(symbols == sorted(symbols) and [r['index'] for r in records] == list(range(len(records))),
            'profile symbol order/index mismatch')
    require(all(re.fullmatch(PREFIX + r'\w+', n) for n in symbols), 'invalid kernel symbol')
    tile = tile_probe_contract(profile)
    flags = ['--wrap=' + n for n in symbols] + ([tile['linker_flag']] if tile else [])
    require(profile.get('linker_flags') == flags,
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


def check_tile_progress(log, expected, tile=None, grid=None):
    """Check selected high-level occurrence and every raw grid call pair."""
    if tile is None:
        return ['tile probe: unexpected records without manifest'] if '[tile-probe]' in log else []
    require(grid and len(grid) == 3 and all(type(v) is int and v > 0 for v in grid),
            'tile probe needs a validated grid')
    pattern = re.compile(r'\[tile-probe\] (begin|returned) (' + SYMBOL +
        r') call=([0-9A-Fa-f]{16}) grid_x=([0-9A-Fa-f]{8}) grid_y=([0-9A-Fa-f]{8})'
        r' grid_z=([0-9A-Fa-f]{8}) x=([0-9A-Fa-f]{8}) y=([0-9A-Fa-f]{8}) z=([0-9A-Fa-f]{8})')
    kernel = re.compile(r'\[kernel\] (begin|end) (' + SYMBOL + r') call=([0-9A-Fa-f]{16})')
    active, enclosing, pending = None, None, None
    count, selections, stages = 0, 0, 0
    errors = []
    total = grid[0] * grid[1] * grid[2]
    chosen = (tile['symbol'], tile['call_index'])
    for line in log.splitlines():
        start = re.fullmatch(r'\[model\] (prefill|decode) begin position=[0-9A-Fa-f]{8} input_token=[0-9A-Fa-f]{8}', line)
        if start:
            if active is not None:
                errors.append('tile probe: graph began before previous graph completed')
            active, enclosing, pending, count, selections = start[1], None, None, 0, 0
        elif (match := kernel.fullmatch(line)):
            phase, symbol, index = match.groups()
            key = symbol, int(index, 16)
            if phase == 'begin':
                enclosing = key
                if key == chosen:
                    selections += 1
            else:
                if key == chosen and (pending is not None or count != total):
                    errors.append('tile probe: selected kernel ended with missing/excess tile calls')
                enclosing = None
        elif '[tile-probe]' in line:
            match = pattern.fullmatch(line)
            if not match or active is None or enclosing != chosen:
                errors.append('tile probe: malformed record or outside selected kernel')
                continue
            phase, symbol, index, *values = match.groups()
            values = [int(value, 16) for value in values]
            coordinates = values[3:]
            if symbol != tile['raw_symbol'] or int(index, 16) != tile['call_index']:
                errors.append('tile probe: raw symbol/occurrence mismatch')
            if values[:3] != grid:
                errors.append('tile probe: grid differs from validated adapter')
            if phase == 'begin':
                wanted = [count // (grid[1] * grid[2]), (count // grid[2]) % grid[1], count % grid[2]]
                if pending is not None or count >= total or coordinates != wanted:
                    errors.append('tile probe: overlapping/duplicate/out-of-order tile begin')
                pending = tuple(coordinates)
            else:
                if pending != tuple(coordinates):
                    errors.append('tile probe: unmatched tile return')
                pending = None
                count += 1
        elif STAGE.fullmatch(line):
            expected_selection = bool(active and tile['call_index'] < expected.get(active, {}).get(tile['symbol'], 0))
            if active is None or pending is not None or enclosing is not None or \
                    selections != int(expected_selection) or count != (total if expected_selection else 0):
                errors.append('tile probe: missing/excess/incomplete selected calls at graph completion')
            active, enclosing, pending = None, None, None
            stages += 1
    if active is not None or pending is not None or not stages:
        errors.append('tile probe: incomplete graph execution')
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('uart', 'profile', 'replacement', 'adapters', 'prefill-ir', 'decode-ir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prefill', type=int, default=16)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--profile-source', type=Path,
                   help='defaults to kernel-profile.c alongside --profile')
    p.add_argument('--tile-adapter', type=Path,
                   help='raw Triton adapter evidence; defaults to tile_probe.raw_adapter from the manifest')
    p.add_argument('--image-plan', type=Path,
                   help='exact w8a8-image-plan.json; requires --image-build for build identity binding')
    p.add_argument('--image-build', type=Path,
                   help='exact image.json build record; requires --image-plan')
    a = p.parse_args()
    source = a.profile_source or a.profile.with_suffix('.c')
    paths = [a.uart, a.profile, source, a.replacement, a.adapters, a.prefill_ir, a.decode_ir]
    hashes = {}
    try:
        hashes.update({str(path): sha256(path.read_bytes()) for path in paths})
        profile = json.loads(a.profile.read_text())
        require(profile.get('source_sha256') == hashes[str(source)],
                'profile generated source hash mismatch')
        require(bool(a.image_plan) == bool(a.image_build),
                '--image-plan and --image-build must be supplied together')
        identity = {'status': 'NOT_CHECKED', 'scope': 'standalone timing/count evidence only; no image identity binding'}
        if a.image_plan:
            for path in (a.image_plan, a.image_build):
                hashes[str(path)] = sha256(path.read_bytes())
            identity = verify_profile_build(profile, json.loads(a.image_plan.read_text()),
                json.loads(a.image_build.read_text()), source, a.adapters)
        expected, provenance = expected_counts(profile,
            json.loads(a.replacement.read_text()), a.adapters.read_text(),
            {'prefill': a.prefill_ir.read_text(), 'decode': a.decode_ir.read_text()})
        tile = tile_probe_contract(profile)
        grid = None
        if tile:
            adapter = a.tile_adapter or Path(tile['raw_adapter'])
            if not adapter.is_absolute() and not adapter.is_file():
                adapter = a.profile.parent / adapter
            raw_bytes = adapter.read_bytes()
            hashes[str(adapter)] = sha256(raw_bytes)
            grid = tile_probe_grid(tile, raw_bytes.decode())
        log = a.uart.read_text(errors='replace')
        report = check(log, expected, prefill=a.prefill, steps=a.steps)
        report['image_identity'] = identity
        progress = bool(profile.get('progress_uart', False))
        completion_sync = profile.get('completion_sync', 'ame-resync')
        require(completion_sync in ('ame-resync', 'fence'), 'unknown profiler completion_sync')
        report['completion_sync'] = completion_sync
        report['errors'] += check_progress(log, expected, enabled=progress)
        report['errors'] += check_tile_progress(log, expected, tile, grid)
        if report['errors']:
            report['status'] = 'NOT_ACCEPTED'
        report['diagnostic_uart_inside_graph_timing'] = progress
        if progress:
            report['limits'].append('Diagnostic UART is inside graph timing; returned/synced markers, when present, are also inside kernel timing. These are not UART-free throughput measurements.')
        if completion_sync == 'fence':
            report['limits'].append('Experimental profiler fence-only mode: count/timing consistency does not prove AME completion or numerical correctness; production kernels and graph-final synchronization are unchanged.')
        if tile:
            report['tile_probe'] = {'raw_symbol': tile['raw_symbol'], 'symbol': tile['symbol'],
                                    'call_index': tile['call_index'], 'grid': grid,
                                    'scope': 'diagnostic call-boundary consistency only'}
            report['limits'].append('Tile UART perturbs kernel execution. Paired raw returns establish observed call boundaries, not numerical correctness, cache freshness or a fault PC.')
        report.update(expected_kernel_calls=expected, expectation_evidence=provenance)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {'status': 'NOT_ACCEPTED', 'errors': [str(error)],
                  'scope': 'Unable to establish independent expected call counts; no profile acceptance.'}
    report['inputs_sha256'] = hashes
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(a.output)}))
    return 0 if report['status'] == 'KERNEL_PROFILE_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
