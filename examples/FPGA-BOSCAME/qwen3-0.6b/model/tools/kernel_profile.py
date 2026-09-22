"""Optional linker wrappers: measure actual Triton calls, without model arithmetic."""
import hashlib
import json
from pathlib import Path
import re


def parse_probe(value):
    """Return a stable kernel symbol and its zero-based per-graph occurrence."""
    if value is None:
        return None
    match = re.fullmatch(r'(_mlir_ciface_kernel_\w+):(0[xX][0-9a-fA-F]+|[0-9]+)', value)
    if not match:
        raise ValueError('--profile-probe must be SYMBOL:CALL_INDEX, with a nonnegative integer index')
    index = int(match[2], 16 if match[2].lower().startswith('0x') else 10)
    if index > 0xffffffffffffffff:
        raise ValueError('--profile-probe index must fit uint64')
    return {'symbol': match[1], 'call_index': index}


def validate_tile_probe(adapters, probe, raw_adapter):
    """Accept only the recorded three-memref int8 matmul adapter ABI."""
    selection = parse_probe(probe)
    if not selection:
        raise ValueError('--profile-tile-probe requires --profile-probe')
    match = re.fullmatch(r'_mlir_ciface_kernel_(matmul_[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*)',
                         selection['symbol'])
    if not match:
        raise ValueError('--profile-tile-probe supports only matmul_MxNxK symbols')
    name = match[1]
    symbol = 'triton_' + name
    high_types = re.findall(r'^extern void ' + re.escape(selection['symbol']) + r'\(([^;]*)\);$',
                            adapters.read_text(), re.M)
    if not high_types or any(re.sub(r'\s+', '', types) != 'MemRef2*,MemRef2*,MemRef2*'
                             for types in high_types):
        raise ValueError('--profile-tile-probe requires exactly three MemRef2 pointers')
    if raw_adapter is None or not raw_adapter.is_file():
        raise ValueError('--profile-tile-probe requires archive evidence/' + name + '/adapter.c')
    text = raw_adapter.read_text()
    types = ['int64_t', 'MemRef0 *'] * 3 + ['int32_t'] * 6
    declarations = re.findall(r'^extern void ' + re.escape(symbol) + r'\(([^;]*)\);$', text, re.M)
    expected = re.sub(r'\s+', '', ','.join(types))
    if len(declarations) != 1 or re.sub(r'\s+', '', declarations[0]) != expected:
        raise ValueError('--profile-tile-probe raw adapter does not have the validated 12-argument ABI')
    if not re.search(r'typedef\s+struct\s*\{\s*void\s*\*allocated\s*,\s*\*aligned\s*;'
                     r'\s*int64_t\s+offset\s*;\s*\}\s*MemRef0\s*;', text):
        raise ValueError('--profile-tile-probe raw adapter has an unsupported MemRef0 layout')
    if not re.search(r'^void ' + re.escape(selection['symbol']) +
                     r'\(MemRef2 \*a0, MemRef2 \*a1, MemRef2 \*a2\)\s*\{', text, re.M):
        raise ValueError('--profile-tile-probe raw adapter is missing the selected high-level function')
    return {
        'symbol': selection['symbol'], 'call_index': selection['call_index'],
        'raw_symbol': symbol, 'raw_return_type': 'void', 'raw_argument_types': types,
        'raw_argument_names': ['rank_a', 'a', 'rank_b', 'b', 'rank_c', 'c',
                               'grid_x', 'grid_y', 'grid_z', 'x', 'y', 'z'],
        'descriptor_layout_lp64': {'allocated': 0, 'aligned': 8, 'offset': 16, 'bytes': 24},
        'raw_adapter': str(raw_adapter),
        'raw_adapter_sha256': hashlib.sha256(raw_adapter.read_bytes()).hexdigest(),
        'linker_flag': '--wrap=' + symbol,
        'limits': [
            'selected state is scoped only around the selected high-level __real_ call; this diagnostic assumes single-threaded RA execution and non-reentrant adapters',
            'tile begin/returned records bracket the raw Triton call and forward all 12 arguments unchanged; no extra fence or model computation is added',
            'UART output changes timing and layout and is included in kernel cycles; this is not a throughput measurement',
            'a missing returned marker identifies the last observed boundary, not an exact PC or proof of hardware failure',
        ],
    }


def tile_probe_source(tile, watch=False):
    symbol = tile['raw_symbol']
    names = tile['raw_argument_names']
    args = ', '.join(t + ' ' + n for t, n in zip(tile['raw_argument_types'], names))
    lines = ['typedef struct { void *allocated, *aligned; int64_t offset; } MemRef0;',
             f'extern void __real_{symbol}({args});',
             f'void __wrap_{symbol}({args}) {{',
             '  const int selected = tile_probe_selected;']
    for phase in ('begin', 'returned'):
        if watch and phase == 'returned':
            lines.append('  if (selected) nr_diag_mark(9, (uint32_t)y);')
        lines += ['  if (selected) {',
                  f'    nr_puts("[tile-probe] {phase} {symbol} call=");',
                  f'    nr_hex64({tile["call_index"]}ULL);']
        for n in names[6:]:
            lines += [f'    nr_puts(" {n}="); nr_hex32((uint32_t){n});']
        lines += ['    nr_puts("\\r\\n");', '  }']
        if phase == 'begin':
            if watch:
                lines.append('  if (selected) nr_diag_mark(8, (uint32_t)y);')
            lines.append(f'  __real_{symbol}(' + ', '.join(names) + ');')
    lines.append('}')
    return lines


def generate_profile(adapters: Path, output: Path, *, progress=False, probe=None,
                     completion_sync='ame-resync', tile_probe=False, raw_adapter=None,
                     watch=False):
    if completion_sync not in ('ame-resync', 'fence'):
        raise ValueError('--profile-sync must be ame-resync or fence')
    tile = validate_tile_probe(adapters, probe, raw_adapter) if tile_probe else None
    probe = parse_probe(probe)
    if probe and not progress:
        raise ValueError('--profile-probe requires --profile-progress')
    text = adapters.read_text()
    prototypes = re.findall(r'^extern void (_mlir_ciface_kernel_\w+)\(([^;]*)\);$',
                            text, re.MULTILINE)
    if not prototypes:
        raise ValueError('no typed Triton kernel declarations for profiling')
    entries = {}
    for name, arguments in prototypes:
        types = [t.strip() for t in arguments.split(',')]
        if not all(re.fullmatch(r'MemRef[1-4]\s*\*', t) for t in types):
            raise ValueError('unsupported kernel ABI for profiling: ' + name)
        if name in entries and entries[name] != types:
            raise ValueError('conflicting kernel declarations: ' + name)
        entries[name] = types
    if probe and probe['symbol'] not in entries:
        raise ValueError('--profile-probe symbol absent from typed adapters: ' + probe['symbol'])
    if watch and (not probe or entries[probe['symbol']] != ['MemRef2 *'] * 3):
        raise ValueError('--profile-watch requires a selected three-MemRef2 kernel')
    records = []
    lines = ['/* Generated profiling glue; calls the linked Triton kernel. */',
             '#include "support.h"', '#include "nr_runtime.h"',
             f'static uint64_t cycles[{len(entries)}], counts[{len(entries)}];']
    if tile:
        lines.append('static int tile_probe_selected;')
    if watch:
        lines += ['#ifndef NR_HANG_DIAGNOSTICS',
                  '#error "profile-watch requires NR_HANG_DIAGNOSTICS"', '#endif']
    lines += ['void qwen_profile_reset(void) {']
    if tile:
        lines.append('  tile_probe_selected = 0;')
    lines += [
             f'  for (unsigned i=0; i<{len(entries)}; ++i) cycles[i]=counts[i]=0;',
             '}', 'void qwen_profile_report(unsigned position) {']
    for index, name in enumerate(sorted(entries)):
        lines += [f'  if (counts[{index}]) {{',
                  '    nr_puts("[profile] position="); nr_hex32(position);',
                  f'    nr_puts(" kernel={name} calls="); nr_hex64(counts[{index}]);',
                  f'    nr_puts(" cycles="); nr_hex64(cycles[{index}]);',
                  '    nr_puts("\\r\\n");', '  }']
    lines.append('}')
    if tile:
        lines += tile_probe_source(tile, watch)
    for index, (name, types) in enumerate(sorted(entries.items())):
        args = ', '.join(f'{t}a{i}' for i,t in enumerate(types))
        params = ', '.join(f'a{i}' for i in range(len(types)))
        lines += [f'extern void __real_{name}({args});',
                  f'void __wrap_{name}({args}) {{']
        if progress:
            lines += [f'  nr_puts("[kernel] begin {name} call="); nr_hex64(counts[{index}]);',
                      '  nr_puts("\\r\\n");']
        selected = probe and probe['symbol'] == name
        if selected:
            lines += [f'  const int probe_selected = counts[{index}] == {probe["call_index"]}ULL;',
                      '  if (probe_selected) {']
            for operand, type_ in enumerate(types):
                rank = int(re.fullmatch(r'MemRef([1-4])\s*\*', type_)[1])
                lines += [f'    nr_puts("[probe] descriptor {name} call="); nr_hex64(counts[{index}]);',
                          f'    nr_puts(" arg="); nr_hex32({operand});',
                          f'    nr_puts(" descriptor="); nr_hex64((uintptr_t)a{operand});',
                          f'    if (a{operand}) {{',
                          f'      nr_puts(" aligned="); nr_hex64((uintptr_t)a{operand}->aligned);',
                          f'      nr_puts(" offset="); nr_hex64((uint64_t)a{operand}->offset);',
                          f'      nr_puts(" rank="); nr_hex32({rank});']
                for dimension in range(rank):
                    lines += [f'      nr_puts(" size{dimension}="); nr_hex64((uint64_t)a{operand}->sizes[{dimension}]);',
                              f'      nr_puts(" stride{dimension}="); nr_hex64((uint64_t)a{operand}->strides[{dimension}]);']
                lines += ['    } else nr_puts(" null_descriptor");',
                          '    nr_puts("\\r\\n");']
            lines.append('  }')
        lines += ['  uint64_t begin=nr_cycles();']
        if selected and tile:
            lines.append('  tile_probe_selected = probe_selected;')
        if selected and watch:
            lines.append('  if (probe_selected) {')
            for i in range(3):
                lines += [f'    if (a{i}) nr_diag_memref({i}, (uintptr_t)a{i}, (uintptr_t)a{i}->aligned,',
                          f'      a{i}->offset, a{i}->sizes[0], a{i}->sizes[1], a{i}->strides[0], a{i}->strides[1]);',
                          f'    else nr_diag_memref({i}, 0, 0, 0, 0, 0, 0, 0);']
            lines += [f'    nr_diag_mark(NR_DIAG_KERNEL_ENTER, {probe["call_index"]}ULL);', '  }']
        lines.append(f'  __real_{name}({params});')
        if selected and watch:
            lines.append(f'  if (probe_selected) nr_diag_mark(NR_DIAG_KERNEL_RETURN, {probe["call_index"]}ULL);')
        if selected and tile:
            lines.append('  tile_probe_selected = 0;')
        if progress:
            # Separate a kernel-body stall from the additional completion
            # sequence below. Diagnostic UART perturbs these measurements.
            lines += [f'  nr_puts("[kernel-phase] returned {name} call=");',
                      f'  nr_hex64(counts[{index}]); nr_puts("\\r\\n");']
        if selected:
            lines += [f'  if (probe_selected) {{ nr_puts("[probe] returned {name} call=");',
                      f'    nr_hex64(counts[{index}]); nr_puts("\\r\\n"); }}']
        if completion_sync == 'ame-resync':
            lines += ['  ame_fence();']
        else:
            # Controlled diagnostic: only the profiler's extra synchronization
            # changes. Kernel instructions and the graph-sync choice are separate.
            lines += ['  __asm__ volatile ("fence rw, rw" ::: "memory");']
        if selected:
            lines += [f'  if (probe_selected) {{ nr_puts("[probe] synced {name} call=");',
                      f'    nr_hex64(counts[{index}]); nr_puts("\\r\\n"); }}']
        lines += [
                  f'  cycles[{index}] += nr_cycles()-begin; ++counts[{index}];']
        if progress:
            lines += [f'  nr_puts("[kernel] end {name} call="); nr_hex64(counts[{index}]-1);',
                      '  nr_puts("\\r\\n");']
        lines.append('}')
        records.append({'index':index, 'symbol':name, 'argument_types':types})
    source = output / 'kernel-profile.c'
    source.write_text('\n'.join(lines)+'\n')
    flags = ['--wrap='+name for name in sorted(entries)]
    if tile:
        flags.append(tile['linker_flag'])
    (output / 'kernel-profile.json').write_text(json.dumps({
        'adapter_sha256':hashlib.sha256(adapters.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'kernels':records, 'linker_flags':flags,
        'progress_uart': bool(progress),
        'phase_probe': probe,
        **({'nh_watch': {'selection': probe, 'phase_8': 'TILE_ENTER',
                         'phase_9': 'TILE_RETURN', 'tile_detail': 'y coordinate',
                         'limits': 'NH samples RA-owned DDR; stable samples can still be stale'}} if watch else {}),
        **({'tile_probe': tile} if tile else {}),
        'completion_sync': completion_sync,
        'scope':'per-graph kernel calls and cycles; kernel bodies are unchanged',
        'limits':[('cycles include one AME resync completion sequence after every call'
                   if completion_sync == 'ame-resync' else
                   'diagnostic fence mode adds only fence rw,rw after each call; it does not prove AME completion or validated throughput'),
                  'the selected completion_sync changes only profiler wrappers; production kernels are unchanged and graph-final synchronization is independently selected by --graph-sync',
                  'graph compute_cycles include wrapper bookkeeping overhead',
                  'with progress_uart enabled, graph cycles include diagnostic UART writes; kernel cycles include the returned marker before the completion fence',
                  'kernel-phase returned precedes the completion fence; kernel end follows it and the counter update',
                  'with phase_probe enabled, selected-call kernel cycles include returned/synced UART; descriptor UART is outside kernel cycles but inside graph cycles',
                  'phase_probe call index is per symbol and resets at every graph invocation',
                  'graph-internal copies, view materialization and scalar work are outside kernel totals',
                  'this instrumented image is not an uninstrumented throughput measurement']},
        indent=2)+'\n')
    return source, flags
