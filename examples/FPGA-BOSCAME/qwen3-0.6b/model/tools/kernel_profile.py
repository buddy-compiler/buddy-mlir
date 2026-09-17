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


def generate_profile(adapters: Path, output: Path, *, progress=False, probe=None):
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
    records = []
    lines = ['/* Generated profiling glue; calls the linked Triton kernel. */',
             '#include "support.h"', '#include "nr_runtime.h"',
             f'static uint64_t cycles[{len(entries)}], counts[{len(entries)}];',
             'void qwen_profile_reset(void) {',
             f'  for (unsigned i=0; i<{len(entries)}; ++i) cycles[i]=counts[i]=0;',
             '}', 'void qwen_profile_report(unsigned position) {']
    for index, name in enumerate(sorted(entries)):
        lines += [f'  if (counts[{index}]) {{',
                  '    nr_puts("[profile] position="); nr_hex32(position);',
                  f'    nr_puts(" kernel={name} calls="); nr_hex64(counts[{index}]);',
                  f'    nr_puts(" cycles="); nr_hex64(cycles[{index}]);',
                  '    nr_puts("\\r\\n");', '  }']
    lines.append('}')
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
        lines += ['  uint64_t begin=nr_cycles();',
                  f'  __real_{name}({params});']
        if selected:
            lines += [f'  if (probe_selected) {{ nr_puts("[probe] returned {name} call=");',
                      f'    nr_hex64(counts[{index}]); nr_puts("\\r\\n"); }}']
        lines += ['  ame_fence();']
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
    (output / 'kernel-profile.json').write_text(json.dumps({
        'adapter_sha256':hashlib.sha256(adapters.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'kernels':records, 'linker_flags':flags,
        'progress_uart': bool(progress),
        'phase_probe': probe,
        'scope':'per-graph kernel calls and cycles; kernel bodies are unchanged',
        'limits':['cycles include one AME completion fence after every call',
                  'graph compute_cycles include wrapper bookkeeping overhead',
                  'with progress_uart enabled, graph cycles also include diagnostic UART writes; kernel cycles exclude these writes',
                  'with phase_probe enabled, selected-call kernel cycles include returned/synced UART; descriptor UART is outside kernel cycles but inside graph cycles',
                  'phase_probe call index is per symbol and resets at every graph invocation',
                  'graph-internal copies, view materialization and scalar work are outside kernel totals',
                  'this instrumented image is not an uninstrumented throughput measurement']},
        indent=2)+'\n')
    return source, flags
