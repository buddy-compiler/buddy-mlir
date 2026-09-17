#!/usr/bin/env python3
"""Verify computed IDs/decoded bytes from a tokenizer probe log, not just PASS.

Execution provenance must be explicitly supplied. FPGA mode also requires the
shared NR completion marker. Host shim output can test this verifier without
being labelled as board evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def verify(log, plan, execution):
    errors, result = [], []
    patterns = {
        'ids': r'\[tokenizer\] case=([0-9a-fA-F]{8}) count=([0-9a-fA-F]{8}) ids=([0-9a-fA-F,]*)',
        'decoded': r'\[tokenizer\] case=([0-9a-fA-F]{8}) decoded_hex=([0-9a-fA-F]*)',
        'cycles': r'\[tokenizer\] case=([0-9a-fA-F]{8}) encode_cycles=([0-9a-fA-F]{16}) decode_cycles=([0-9a-fA-F]{16})',
        'pass': r'verify tokenizer case ([0-9a-fA-F]{8}): (PASS|FAIL)',
    }
    found = {kind: {} for kind in patterns}
    for kind, pattern in patterns.items():
        for match in re.finditer(pattern, log):
            key = int(match.group(1), 16)
            if key in found[kind]: errors.append(f'duplicate {kind} case {key}')
            found[kind][key] = match.groups()[1:]
    wanted = {f['case'] for f in plan['fixtures']}
    for kind in found:
        if set(found[kind]) != wanted: errors.append(f'{kind}: expected all cases exactly once')
    for fixture in plan['fixtures']:
        case = fixture['case']
        if any(case not in found[kind] for kind in patterns): continue
        count, raw_ids = found['ids'][case]
        ids = [int(item, 16) for item in raw_ids.split(',') if item]
        decoded = found['decoded'][case][0].lower()
        status = (int(count, 16) == len(ids) and ids == fixture['expected_ids']
                  and decoded == fixture['expected_decoded_hex']
                  and found['pass'][case][0] == 'PASS')
        if not status: errors.append(f'case {case}: computed result differs from oracle')
        result.append({'case':case, 'status':'PASS' if status else 'FAIL',
                       'actual_ids':ids, 'expected_ids':fixture['expected_ids'],
                       'actual_decoded_hex':decoded,
                       'encode_ticks':int(found['cycles'][case][0], 16),
                       'decode_ticks':int(found['cycles'][case][1], 16)})
    if 'verify tokenizer suite: PASS' not in log: errors.append('missing suite completion')
    if re.search(r'\bFAIL\b|\btrap\b', log, re.I): errors.append('failure/trap marker present')
    if execution == 'fpga' and 'verify NR runtime: PASS' not in log:
        errors.append('missing NR runtime completion')
    return {'status':'FAIL' if errors else 'PASS', 'execution':execution,
            'board_tokenizer_verified':execution == 'fpga' and not errors,
            'tick_unit':'NR rdcycle' if execution == 'fpga' else 'host shim clock() ticks',
            'cases':result, 'errors':errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--uart-log', type=Path, required=True)
    p.add_argument('--execution', choices=('host', 'fpga'), required=True)
    p.add_argument('--run-id')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.manifest.read_text())
    raw = a.uart_log.read_bytes()
    report = verify(raw.decode('utf-8', errors='replace'), plan, a.execution)
    report.update(manifest=str(a.manifest), manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                  uart_log=str(a.uart_log), uart_log_sha256=hashlib.sha256(raw).hexdigest(),
                  binary_sha256=plan['bin_sha256'], run_id=a.run_id)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'{report["status"]}: {len(report["cases"])} tokenizer cases; execution={a.execution}')
    for error in report['errors']: print(error)
    return int(bool(report['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
