"""Optional linker wrappers: measure actual Triton calls, without model arithmetic."""
import hashlib
import json
from pathlib import Path
import re


def generate_profile(adapters: Path, output: Path, *, progress=False):
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
        lines += ['  uint64_t begin=nr_cycles();',
                  f'  __real_{name}({params});',
                  '  ame_fence();',
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
        'scope':'per-graph kernel calls and cycles; kernel bodies are unchanged',
        'limits':['cycles include one AME completion fence after every call',
                  'graph compute_cycles include wrapper bookkeeping overhead',
                  'with progress_uart enabled, graph cycles also include diagnostic UART writes; kernel cycles exclude these writes',
                  'graph-internal copies, view materialization and scalar work are outside kernel totals',
                  'this instrumented image is not an uninstrumented throughput measurement']},
        indent=2)+'\n')
    return source, flags
