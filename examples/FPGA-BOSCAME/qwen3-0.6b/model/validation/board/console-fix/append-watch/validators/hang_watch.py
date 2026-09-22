"""Generate a silent mailbox observer around one actual Triton kernel call.

The wrapper preserves the typed C ABI and delegates all computation to the
original linked kernel.  Runtime mailbox publication and NH-side observation
are supplied by the shared NR runtime, not by this generator.
"""

import hashlib
import json
from pathlib import Path
import re


def parse_watch(value: str) -> dict:
    """Parse a symbol and its zero-based occurrence within each graph call."""
    if not isinstance(value, str):
        raise ValueError('--hang-watch must be SYMBOL:CALL_INDEX')
    match = re.fullmatch(
        r'(_mlir_ciface_kernel_[A-Za-z0-9_]+):(0[xX][0-9a-fA-F]+|[0-9]+)',
        value)
    if match is None:
        raise ValueError('--hang-watch must be SYMBOL:CALL_INDEX, with a nonnegative integer index')
    index = int(match[2], 16 if match[2].lower().startswith('0x') else 10)
    if index > 0xffffffffffffffff:
        raise ValueError('--hang-watch index must fit uint64')
    return {'symbol': match[1], 'call_index': index}


def _validate_signature(text: str, symbol: str) -> None:
    # Generated adapter prototypes are C declarations, with types but no
    # argument names.  Ignore comments; accept only the exact ABI we can record.
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.DOTALL)
    declarations = re.findall(
        r'\bextern\s+([^;{}]*?)\b' + re.escape(symbol)
        + r'\s*\(([^;{}]*)\)\s*;', text)
    if not declarations:
        raise ValueError('--hang-watch symbol absent from typed adapters: ' + symbol)
    for result, arguments in declarations:
        types = [item.strip() for item in arguments.split(',')]
        if (result.strip() != 'void' or len(types) != 3
                or not all(re.fullmatch(r'MemRef2\s*\*', item) for item in types)):
            raise ValueError('unsupported --hang-watch ABI; requires void '
                             + symbol + '(MemRef2 *, MemRef2 *, MemRef2 *)')


def validate_watch(adapters: Path, probe: str) -> dict:
    """Validate selection and the exact ABI without creating any output files."""
    watch = parse_watch(probe)
    _validate_signature(adapters.read_text(), watch['symbol'])
    return watch


def generate_watch(adapters: Path, output: Path, probe: str):
    """Return generated C source and ld flags; write evidence alongside it.

    Only the requested kernel is wrapped.  Graph boundaries must call
    qwen_hang_reset(position), including before every decode step.  The shared
    runtime and callers are responsible for subsequent graph/sync/collect marks.
    """
    watch = parse_watch(probe)
    name, index = watch['symbol'], watch['call_index']
    adapter_bytes = adapters.read_bytes()
    _validate_signature(adapter_bytes.decode(), name)
    lines = [
        '/* Generated silent observer; computation remains in the linked Triton kernel. */',
        '#include "support.h"',
        '#include "nr_runtime.h"',
        '#ifndef NR_HANG_DIAGNOSTICS',
        '#error "hang-watch requires NR_HANG_DIAGNOSTICS"',
        '#endif',
        'static uint64_t qwen_hang_calls;',
        'void qwen_hang_reset(unsigned position) {',
        '  qwen_hang_calls = 0;',
        '  nr_diag_reset(position);',
        '}',
        f'extern void __real_{name}(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);',
        f'void __wrap_{name}(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {{',
        '  const uint64_t call_index = qwen_hang_calls++;',
        f'  const int selected = call_index == {index}ULL;',
        '  if (selected) {',
    ]
    for operand in range(3):
        arg = f'a{operand}'
        lines.extend([
            f'    if ({arg}) {{',
            f'      nr_diag_memref({operand}, (uintptr_t){arg}, (uintptr_t){arg}->aligned,',
            f'                     {arg}->offset, {arg}->sizes[0], {arg}->sizes[1],',
            f'                     {arg}->strides[0], {arg}->strides[1]);',
            '    } else {',
            f'      nr_diag_memref({operand}, 0, 0, 0, 0, 0, 0, 0);',
            '    }',
        ])
    lines.extend([
        '    nr_diag_mark(NR_DIAG_KERNEL_ENTER, call_index);',
        '  }',
        f'  __real_{name}(a0, a1, a2);',
        '  if (selected) nr_diag_mark(NR_DIAG_KERNEL_RETURN, call_index);',
        '}',
    ])
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'hang-watch.c'
    source.write_text('\n'.join(lines) + '\n')
    flags = ['--wrap=' + name]
    metadata = {
        'schema_version': 1,
        'adapter_sha256': hashlib.sha256(adapter_bytes).hexdigest(),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'watch': watch,
        'linker_flags': flags,
        'argument_types': ['MemRef2 *'] * 3,
        'return_type': 'void',
        'call_index_scope': 'zero-based per selected symbol; reset before every graph invocation',
        'reset_symbol': 'qwen_hang_reset',
        'operands': [
            {'index': i, 'rank': 2, 'dtype': None,
             'dtype_evidence': 'unknown: the typed adapter MemRef2 uses void *',
             'offset_unit': 'elements', 'stride_unit': 'elements',
             'descriptor_and_aligned_unit': 'byte address'}
            for i in range(3)
        ],
        'phases': {
            '1': 'GRAPH_BEGIN', '2': 'KERNEL_ENTER', '3': 'KERNEL_RETURN',
            '4': 'GRAPH_RETURN', '5': 'SYNC_DONE', '6': 'COLLECT_BEGIN',
            '7': 'COLLECT_DONE',
        },
        'wrapper_marks': ['KERNEL_ENTER', 'KERNEL_RETURN'],
        'mark_detail': 'selected zero-based call_index',
        'progress_uart_in_wrapper': False,
        'extra_ame_sync_in_wrapper': False,
        'limits': [
            'The wrapper observes one selected occurrence per graph and changes no kernel arithmetic.',
            'The selected call snapshots three descriptors before KERNEL_ENTER; invalid descriptor pointers can fault during observation.',
            'Null descriptor pointers are recorded as zeros and passed unchanged to the original kernel.',
            'KERNEL_RETURN means the C function returned; it does not assert AME or DDR completion.',
            'Descriptor offsets and strides are elements, not bytes; this ABI does not encode dtype or element size.',
            'The runtime mailbox and NH polling can perturb timing and depend on the platform cache contract.',
            'A last observed ENTER alone cannot distinguish a kernel stall, a visibility failure, a trap, or an observer that stopped.',
            'One observation cannot establish cache incoherence or unsupported instructions as the cause.',
            'Graph, synchronization and collection markers must be supplied by the model launch code.',
        ],
    }
    (output / 'hang-watch.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return source, flags
