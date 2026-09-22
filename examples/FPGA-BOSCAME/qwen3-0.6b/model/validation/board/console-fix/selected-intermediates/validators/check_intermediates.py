#!/usr/bin/env python3
"""Check the complete 16-token prefill + 8-decode intermediate trace.

This validates the manifest's selected full tensor boundaries, not every hidden
tensor. Archiving separately verifies the generated probe and embedded oracle.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


HEX32 = r'([0-9A-Fa-f]{8})'
RECORD = re.compile(r'\[intermediate\] position=' + HEX32 + r' entry=' + HEX32
                    + r' calls=' + HEX32 + r' checked=' + HEX32 + r' count=' + HEX32
                    + r' max_abs_bits=' + HEX32 + r' mean_abs_bits=' + HEX32)
COMPLETE = re.compile(r'\[intermediate\] complete position=' + HEX32 + r' (PASS|FAIL)')
POSITIONS = [0, *range(16, 24)]
SCOPE = ('Partial hidden-state coverage: selected full tensors at external adapter '
         'boundaries in one decoder layer, 16-token prefill and eight decode calls; '
         'this does not establish acceptance of every intermediate tensor.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def uint(value, *, positive=False):
    return type(value) is int and (0 < value if positive else 0 <= value) and value <= 0xffffffff


def validate_manifest(manifest):
    """Reject an incomplete, ambiguous, or unsupported probe specification."""
    require(isinstance(manifest, dict), 'intermediate manifest must be an object')
    version = manifest.get('schema_version')
    require(type(version) is int and version in (1, 2), 'unsupported intermediate manifest schema_version')
    for name, wanted in [('prefill_len', 16), ('decode_steps', 8)]:
        require(type(manifest.get(name)) is int and manifest[name] == wanted,
                'unsupported intermediate manifest ' + name)
    if version == 1:
        require(type(manifest.get('layers')) is int and manifest['layers'] == 1,
                'unsupported intermediate manifest layers')
        selected_layers = [0]
    else:
        require(uint(manifest.get('layers'), positive=True), 'invalid intermediate model layer count')
        selected_layers = manifest.get('selected_layers')
        require(isinstance(selected_layers, list) and selected_layers
                and all(uint(layer) and layer < manifest['layers'] for layer in selected_layers)
                and selected_layers == sorted(set(selected_layers)),
                'invalid intermediate selected_layers')
        require(isinstance(manifest.get('scope'), str) and bool(manifest['scope'].strip()),
                'missing intermediate scope')
        limits = manifest.get('limitations')
        require(isinstance(limits, list) and limits
                and all(isinstance(limit, str) and bool(limit.strip()) for limit in limits),
                'missing intermediate limitations')
    for name in ('max_abs_tolerance', 'mean_abs_tolerance'):
        value = manifest.get(name)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                'invalid intermediate tolerance: ' + name)
    entries = manifest.get('entries')
    require(isinstance(entries, list) and entries, 'expected nonempty intermediate entries')
    if version == 1:
        require(len(entries) == 92, 'expected exactly 92 intermediate entries')
    groups = {'prefill': [], 'decode': []}
    cursor, identities = 0, set()
    for index, entry in enumerate(entries):
        require(isinstance(entry, dict), 'intermediate entry must be an object')
        require(type(entry.get('index')) is int and entry['index'] == index, 'intermediate entry index/order mismatch')
        graph = entry.get('graph')
        require(isinstance(graph, str) and graph in groups
                and uint(entry.get('layer')) and entry['layer'] in selected_layers,
                'intermediate graph/layer mismatch')
        if version == 1:
            require(graph == ('prefill' if index < 46 else 'decode'), 'intermediate graph/layer mismatch')
        require(graph != 'prefill' or not groups['decode'], 'intermediate graph grouping/order mismatch')
        groups[graph].append(entry)
        symbol = entry.get('symbol')
        require(isinstance(symbol, str) and re.fullmatch(r'_mlir_ciface_qwen_graph_\w+', symbol),
                'invalid intermediate adapter symbol')
        require(uint(entry.get('operand')) and entry.get('phase') in ('before', 'after'),
                'invalid intermediate operand/phase')
        identity = (graph, symbol, entry['operand'], entry['phase'])
        require(identity not in identities, 'duplicate intermediate adapter boundary')
        identities.add(identity)
        shape = entry.get('shape')
        require(isinstance(shape, list) and 1 <= len(shape) <= 4 and all(uint(x, positive=True) for x in shape),
                'invalid intermediate tensor shape')
        require(type(entry.get('rank')) is int and entry['rank'] == len(shape)
                and uint(entry.get('elements'), positive=True) and entry['elements'] == math.prod(shape),
                'intermediate rank/elements mismatch')
        require(entry.get('dtype') in ('f32', 'i8'), 'unsupported intermediate dtype')
        require(entry.get('transform') in ('identity', 'last_row', 'heads_first', 'context_heads_first', 'projection_heads'),
                'unsupported intermediate reference transform')
        suffix = entry.get('reference_suffix')
        require(isinstance(suffix, str) and bool(suffix), 'missing intermediate reference suffix')
        if version == 2 and 'reference_aliases' in entry:
            aliases = entry['reference_aliases']
            require(isinstance(aliases, list)
                    and all(isinstance(alias, str) and bool(alias.strip()) and alias != suffix for alias in aliases)
                    and len(aliases) == len(set(aliases)), 'invalid intermediate reference aliases')
        count = 1 if graph == 'prefill' else 8
        keys = [('prefill' if graph == 'prefill' else f'decode_{i}') + '_' + suffix for i in range(count)]
        require(entry.get('reference_keys') == keys, 'intermediate reference key/trajectory mismatch')
        offsets = entry.get('offsets')
        require(isinstance(offsets, list) and all(uint(x) for x in offsets)
                and offsets == [cursor + i * entry['elements'] for i in range(count)],
                'intermediate reference offsets overlap, omit data, or differ from packing order')
        cursor += count * entry['elements']
    for graph, group in groups.items():
        require(bool(group), 'missing intermediate graph entries: ' + graph)
        require({entry['layer'] for entry in group} == set(selected_layers),
                'selected intermediate layers lack entries in ' + graph)
    require(type(manifest.get('reference_bytes')) is int and manifest['reference_bytes'] == cursor * 4,
            'intermediate reference byte count mismatch')
    flags = ['--wrap=' + symbol for symbol in sorted({e['symbol'] for e in entries})]
    require(manifest.get('linker_flags') == flags, 'intermediate linker wrapper list mismatch')
    for name in ('generated_source_sha256', 'reference_blob_sha256'):
        value = manifest.get(name)
        require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid intermediate ' + name)
    sources = manifest.get('source_sha256')
    require(isinstance(sources, dict) and sources and all(isinstance(k, str) and k and isinstance(v, str)
            and re.fullmatch('[0-9a-f]{64}', v) for k, v in sources.items()), 'missing/invalid intermediate source hashes')
    uncovered = manifest.get('uncovered_reference_tensors')
    require(isinstance(uncovered, list) and all(isinstance(x, str) for x in uncovered)
            and uncovered == sorted(set(uncovered)), 'invalid uncovered intermediate tensor list')
    return groups


def manifest_scope(manifest):
    if manifest['schema_version'] == 1:
        return SCOPE
    layers = ', '.join(map(str, manifest['selected_layers']))
    return ('Partial hidden-state coverage: selected full tensors at external adapter '
            f'boundaries in decoder layers [{layers}] of {manifest["layers"]}, '
            '16-token prefill and eight decode calls; '
            'this does not establish acceptance of every intermediate tensor.')


def float_bits(value):
    return struct.unpack('<f', struct.pack('<I', int(value, 16)))[0]


def check(log, manifest):
    groups = validate_manifest(manifest)
    expected_comparisons = len(groups['prefill']) + manifest['decode_steps'] * len(groups['decode'])
    expected = []
    for position in POSITIONS:
        graph = 'prefill' if position == 0 else 'decode'
        expected += [('entry', position, e['index']) for e in groups[graph]]
        expected.append(('complete', position, 'PASS'))
    expected.append(('return',))
    events, records, completions, errors = [], [], [], []
    for number, line in enumerate(log.splitlines(), 1):
        match = RECORD.fullmatch(line)
        if match:
            pos, index, calls, checked, count = (int(x, 16) for x in match.groups()[:5])
            maximum, mean = (float_bits(x) for x in match.groups()[5:])
            events.append(('entry', pos, index))
            entry = manifest['entries'][index] if index < len(manifest['entries']) else None
            if entry is None or entry['graph'] != ('prefill' if pos == 0 else 'decode') or pos not in POSITIONS:
                errors.append(f'unknown entry or wrong graph position at line {number}')
            if calls != 1 or checked != 1:
                errors.append(f'entry not called and fully checked exactly once at line {number}')
            if entry is None or count != entry['elements']:
                errors.append(f'intermediate element count mismatch at line {number}')
            for name, value, limit in [('max', maximum, manifest['max_abs_tolerance']),
                                       ('mean', mean, manifest['mean_abs_tolerance'])]:
                if not math.isfinite(value) or value < 0 or value > limit:
                    errors.append(f'intermediate {name} error nonfinite, negative, or above tolerance at line {number}')
            if math.isfinite(maximum) and math.isfinite(mean) and mean > maximum:
                errors.append(f'intermediate mean error exceeds maximum at line {number}')
            record = {'position': pos, 'entry': index, 'calls': calls, 'checked': checked,
                      'elements': count, 'max_abs': maximum if math.isfinite(maximum) else None,
                      'mean_abs': mean if math.isfinite(mean) else None}
            if entry is not None and pos in POSITIONS:
                step = 0 if pos == 0 else pos - 16
                if step < len(entry['reference_keys']):
                    record.update(reference_key=entry['reference_keys'][step], reference_offset=entry['offsets'][step])
            records.append(record)
        elif (match := COMPLETE.fullmatch(line)):
            pos, status = int(match[1], 16), match[2]
            events.append(('complete', pos, status))
            completions.append({'position': pos, 'status': status})
        elif '[intermediate' in line or 'intermediate]' in line:
            errors.append(f'malformed intermediate record at line {number}')
        elif line == '[nr] RA returned: PASS':
            events.append(('return',))
        elif 'RA returned' in line:
            errors.append(f'malformed or unsuccessful NR return at line {number}')
        if re.search(r'\b(?:FAIL|TRAP)\b|\bmcause\s*=', line):
            errors.append(f'runtime failure/trap at line {number}')
        if (match := re.search(r'\[nr\] launch[^\r\n]* status=0x([0-9A-Fa-f]+)', line)) and int(match[1], 16):
            errors.append(f'nonzero runtime launch status at line {number}')
    if events != expected:
        errors.append('missing, duplicate, unknown, or out-of-order intermediate entries/completions/NR return')
    if len(records) != expected_comparisons:
        errors.append(f'expected exactly {expected_comparisons} intermediate comparisons, found {len(records)}')
    if len(completions) != 9:
        errors.append(f'expected exactly nine intermediate completions, found {len(completions)}')
    return {'status': 'INTERMEDIATES_PASS' if not errors else 'NOT_ACCEPTED',
            'scope': manifest_scope(manifest), 'schema_version': manifest['schema_version'],
            'layers': manifest['layers'],
            'selected_layers': manifest['selected_layers'] if manifest['schema_version'] == 2 else [0],
            'errors': errors, 'comparisons': len(records), 'expected_comparisons': expected_comparisons,
            'entries_per_graph': {graph: len(entries) for graph, entries in groups.items()}, 'graph_positions': POSITIONS,
            'max_abs_tolerance': manifest['max_abs_tolerance'],
            'mean_abs_tolerance': manifest['mean_abs_tolerance'],
            'completions': completions, 'records': records,
            'uncovered_reference_tensors': manifest['uncovered_reference_tensors'],
            'limits': ['The UART checker trusts the supplied manifest; archive_model_run.py separately checks source, NPZ packing, and embedded ELF bytes.',
                       'Integer matmul accumulators and all unlisted hidden tensors are outside this intermediate check.',
                       'Probe comparisons, synchronization, and diagnostic output affect instrumented execution time.']
                      + (manifest['limitations'] if manifest['schema_version'] == 2 else [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('uart', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    hashes = {}
    try:
        for path in (args.uart, args.manifest):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = check(args.uart.read_text(errors='replace'), json.loads(args.manifest.read_text()))
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
        report = {'status': 'NOT_ACCEPTED', 'scope': 'Intermediate coverage could not be established.',
                  'errors': [str(error)]}
    report['inputs_sha256'] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output)}))
    return 0 if report['status'] == 'INTERMEDIATES_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
