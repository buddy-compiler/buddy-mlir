#!/usr/bin/env python3
"""Build (never launch) a shared-NR-runtime tokenizer board validation image.

Literal UTF-8 inputs and the tokenizer resource are embedded read-only. Firmware
renders each chat template, encodes it and incrementally decodes its actual IDs.
Expected IDs/bytes are only the test oracle; firmware prints computed results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from pack_tokenizer import pack


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(kind, name, values):
    return f'static const {kind} {name}[] = {{' + ','.join(str(v) for v in values) + '};'


def main():
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    model = Path(__file__).resolve().parents[1]
    repo = model.parents[3]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=model/'assets/official')
    p.add_argument('--fixtures', type=Path, default=model/'validation/tokenizer-encode-audit.json')
    p.add_argument('--output', type=Path, default=model/'build/tokenizer-probe')
    p.add_argument('--llvm-bin', type=Path, default=repo/'llvm/build-2d26/bin')
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads(a.fixtures.read_text())['chat_fixtures']
    assert len(fixtures) == 8, 'expected the eight reviewed literal-text fixtures'
    reference = Tokenizer.from_file(str(a.assets/'tokenizer.json'))
    hf = AutoTokenizer.from_pretrained(str(a.assets), local_files_only=True)
    rows, lines, manifest = [], ['#include <stddef.h>', '#include <stdint.h>',
        'typedef struct { const uint8_t *user; size_t user_size; int thinking;',
        '  const uint32_t *expected_ids; size_t expected_count;',
        '  const uint8_t *decoded; size_t decoded_size; } TokenizerProbeFixture;'], []
    for i, fixture in enumerate(fixtures):
        user = fixture['user_text'].encode()
        messages = [{'role':'user', 'content':fixture['user_text']}]
        rendered = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=fixture['enable_thinking'])
        expected = reference.encode(rendered, add_special_tokens=False).ids
        assert expected == fixture['host_reference_prompt_ids'], 'fixture/reference drift'
        decoded = reference.decode(expected, skip_special_tokens=False).encode()
        lines += [array('uint8_t', f'user_{i}', user),
                  array('uint32_t', f'expected_{i}', expected),
                  array('uint8_t', f'decoded_{i}', decoded)]
        rows.append(f'  {{user_{i},sizeof(user_{i}),{int(fixture["enable_thinking"])},'
                    f'expected_{i},{len(expected)},decoded_{i},sizeof(decoded_{i})}}')
        manifest.append({'case':i, 'user_text':fixture['user_text'],
                         'enable_thinking':fixture['enable_thinking'],
                         'expected_ids':expected, 'expected_decoded_hex':decoded.hex()})
    lines += [f'#define TOKENIZER_PROBE_CASE_COUNT {len(rows)}u',
              'static const TokenizerProbeFixture tokenizer_probe_fixtures[] = {',
              ',\n'.join(rows), '};', '']
    (out/'tokenizer_probe_fixtures.h').write_text('\n'.join(lines))
    resource = pack(a.assets, out/'tokenizer.bin')
    (out/'tokenizer_blob.S').write_text(
        '.section .rodata.qwen_tokenizer,"a",@progbits\n.balign 64\n'
        '.globl qwen_tokenizer_blob_start\nqwen_tokenizer_blob_start:\n'
        '.incbin ' + json.dumps(str(out/'tokenizer.bin')) + '\n'
        '.globl qwen_tokenizer_blob_end\nqwen_tokenizer_blob_end:\n'
        '.section .note.GNU-stack,"",@progbits\n')
    build_cmd = ['make', '-f', str(model/'tests/tokenizer_probe.mk'),
                 f'MODEL={model}', f'BUILD={out}', f'LLVM_BIN={a.llvm_bin.resolve()}', '-j4']
    result = subprocess.run(build_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/'build.log').write_text(result.stdout)
    print(result.stdout, end='')
    result.check_returncode()
    elf, binary = out/'tokenizer_probe.elf', out/'tokenizer_probe.bin'
    audit_cmd = [sys.executable, str(repo/'examples/FPGA-BOSCAME/tools/check_nr_elf.py'),
                 str(elf), '--objdump', str(a.llvm_bin.resolve()/'llvm-objdump'),
                 '--output', str(out/'elf-audit.json')]
    subprocess.run(audit_cmd, check=True)
    nm = subprocess.check_output([a.llvm_bin/'llvm-nm', '--undefined-only', elf], text=True)
    assert not nm.strip(), 'unresolved symbols: ' + nm
    symbol_list = subprocess.check_output([a.llvm_bin/'llvm-nm', '-n', elf], text=True)
    (out/'symbols.txt').write_text(symbol_list)
    sections = subprocess.check_output([a.llvm_bin/'llvm-readelf', '-lSW', elf], text=True)
    (out/'elf-layout.txt').write_text(sections)
    sources = list((model/'text').glob('*.c')) + list((model/'text').glob('*.h'))
    sources += [model/'tests/tokenizer_probe.c', model/'tests/tokenizer_probe.mk', Path(__file__).resolve()]
    sources += list((repo/'examples/FPGA-BOSCAME/common/nr').glob('*.[cSh]'))
    sources += [repo/'examples/FPGA-BOSCAME/common/nr/nr.ld',
                repo/'examples/FPGA-BOSCAME/common/nr/nr.mk',
                repo/'examples/FPGA-BOSCAME/common/uart/uart.h']
    plan = {'status':'BUILT_AND_ISA_AUDITED_NOT_RUN', 'fpga_executed':False,
            'entry':'0x80000000', 'elf':str(elf), 'bin':str(binary),
            'elf_sha256':sha(elf), 'bin_sha256':sha(binary), 'bin_bytes':binary.stat().st_size,
            'resource_bytes':resource['bytes'], 'resource_sha256':resource['sha256'],
            'fixtures':manifest, 'expected_success_marker':'verify tokenizer suite: PASS',
            'commands':[build_cmd, audit_cmd], 'undefined_symbols':[],
            'source_sha256':{str(x.relative_to(repo)):sha(x) for x in sources},
            'fixture_source_sha256':sha(a.fixtures),
            'notes':['No model inference and no UART RX is needed by this automatic probe.',
                     'Expected data is used for comparison only; output IDs/bytes are computed by firmware.',
                     'Token resource is embedded in readonly ELF data; startup BSS clear cannot overwrite it.']}
    (out/'probe.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
    script = ['#!/usr/bin/env bash', 'set -euo pipefail',
              '# Generated replay of local build/audit; no upload or board launch.',
              shlex.join(build_cmd), shlex.join(audit_cmd), '']
    (out/'rebuild.sh').write_text('\n'.join(script)); os.chmod(out/'rebuild.sh', 0o755)
    print(f'BUILT AND AUDITED (not run on FPGA): {binary}; {binary.stat().st_size} bytes')


if __name__ == '__main__':
    main()
