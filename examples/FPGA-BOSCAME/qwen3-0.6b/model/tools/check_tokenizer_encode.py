#!/usr/bin/env python3
"""Host differential tests of the same freestanding encoder used in NR firmware.

This never accesses an FPGA. It compares C encoding, NFC, split boundaries and
chat-template->encoding against the pinned official tokenizer. Output is explicit
about host-only provenance. Use check_text.py for incremental decoding checks.
"""
import argparse
import ctypes as C
import hashlib
import itertools
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_text import Resource
from pack_tokenizer import EXPECTED_REGEX, pack


def main():
    import tokenizers
    from tokenizers import Tokenizer, Regex, pre_tokenizers, normalizers
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cc', default='cc')
    a = p.parse_args()
    a.build.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    blob_path = a.build / 'tokenizer.bin'
    pack(a.assets, blob_path)
    library = a.build / 'libqwen_encode_check.so'
    cmd = [*shlex.split(a.cc), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
           '-shared', '-fPIC', str(root / 'text/tokenizer_resource.c'),
           str(root / 'tests/tokenizer_primitives_check.c'),
           str(root / 'text/unicode_tables.c'), '-o', str(library)]
    subprocess.run(cmd, check=True)
    lib = C.CDLL(str(library.resolve()))
    lib.qwen_test_resource_size.restype = C.c_size_t
    assert lib.qwen_test_resource_size() == C.sizeof(Resource), 'ctypes Resource ABI drift'
    lib.qwen_tokenizer_open.argtypes = [C.POINTER(Resource), C.c_void_p, C.c_size_t]
    lib.qwen_encode.argtypes = [C.POINTER(Resource), C.c_void_p, C.c_size_t,
                               C.POINTER(C.c_uint32), C.c_size_t, C.POINTER(C.c_size_t)]
    lib.qwen_test_nfc.argtypes = [C.POINTER(C.c_uint32), C.c_uint32,
                                C.POINTER(C.c_uint32), C.c_uint32]
    lib.qwen_test_piece_lengths.argtypes = lib.qwen_test_nfc.argtypes
    lib.qwen_chat_single_turn.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t),
                                         C.c_void_p, C.c_size_t, C.c_int,
                                         C.c_void_p, C.c_size_t, C.c_int]
    raw = blob_path.read_bytes()
    memory, r = C.create_string_buffer(raw), Resource()
    assert lib.qwen_tokenizer_open(C.byref(r), memory, len(raw)) == 0
    official = Tokenizer.from_file(str(a.assets / 'tokenizer.json'))
    hf = AutoTokenizer.from_pretrained(str(a.assets), local_files_only=True)
    split = pre_tokenizers.Split(Regex(EXPECTED_REGEX), behavior='isolated')
    nfc, nfd = normalizers.NFC(), normalizers.NFD()
    corpus = ['', 'Hello, world!', '你好，世界！', 'a\0b', 'Å', '\u0340',
              'A\u0327\u0301', 'A\u030a\u0301', 'a\u0301\u0301', '\u0301A\u0327',
              '\u1100\u1161\u11a8', '\uac00\u11a8', '\U00011938',
              'x\n \ny', '\r \n \t\r\nx', "a'ſb", '\x1cA\x1d1\x1e!\x1f?',
              'a\U00011f02b', 'a\U0001e4d0b', '👩\u200d💻🚀🙂', 'مرحبا بالعالم',
              'नमस्ते दुनिया', '한글 日本語']
    doc = json.loads((a.assets / 'tokenizer.json').read_text())
    for row in doc['added_tokens']:
        for prefix, suffix in [('', ''), ('!', '?'), ('\u0301', '\u0327'), ('x\0', '\0y')]:
            corpus.append(prefix + row['content'] + suffix)
    # Every canonical normalization scalar and algorithmic Hangul syllable;
    # independently exercise decomposed spellings and blocking/reordering.
    normal_cases = []
    for cp in range(0x110000):
        if 0xd800 <= cp <= 0xdfff:
            continue
        ch = chr(cp)
        decomposed = nfd.normalize_str(ch)
        if decomposed != ch:
            normal_cases.extend((ch, decomposed))
    for text in normal_cases:
        values = (C.c_uint32 * len(text))(*map(ord, text))
        output = (C.c_uint32 * (len(text) * 4 + 1))()
        count = lib.qwen_test_nfc(values, len(text), output, len(output))
        actual = ''.join(map(chr, output[:count])) if count >= 0 else None
        assert actual == nfc.normalize_str(text), (repr(text), repr(actual), repr(nfc.normalize_str(text)))
    # Unicode range boundaries make newly added scripts and whitespace semantics
    # visible without relying only on conventional English/Chinese prompts.
    import re
    table = (root / 'text/unicode_tables.c').read_text()
    for name in ('letter', 'number', 'space'):
        section = table.split(f'qwen_{name}_ranges[] = {{')[1].split('};')[0]
        for left, right in re.findall(r'\{0x([0-9a-f]+)u, 0x([0-9a-f]+)u\}', section):
            for cp in (int(left, 16)-1, int(left, 16), int(right, 16), int(right, 16)+1):
                if 0 <= cp < 0x110000 and not 0xd800 <= cp <= 0xdfff:
                    corpus.append('a' + chr(cp) + '1!')
    rng = random.Random(20260917)
    alphabet = list("abXYZ' \t\r\n012!?,") + list('éÅÅ\u0301\u0327\u030a\u0345\u0334\u1100\u1161\u11a8你好🙂\u0085\u00a0\u2003\x1c')
    corpus.extend(''.join(rng.choices(alphabet, k=rng.randrange(1, 40))) for _ in range(2000))
    corpus.extend('a'+''.join(chars)+'b' for chars in itertools.product(' \t\r\n', repeat=4))
    corpus = list(dict.fromkeys(corpus))
    fixtures = []
    for text in corpus:
        raw_text = text.encode()
        output = (C.c_uint32 * (len(raw_text) + 1))()
        written = C.c_size_t(123)
        status = lib.qwen_encode(C.byref(r), C.c_char_p(raw_text), len(raw_text), output,
                                 len(output), C.byref(written))
        expected = official.encode(text, add_special_tokens=False).ids
        assert status == 0 and list(output[:written.value]) == expected, \
            (repr(text), status, list(output[:written.value]), expected)
        # Direct regex split checks isolate boundaries even when BPE happens to
        # produce the same ids on either side of a wrongly split piece.
        normalized = nfc.normalize_str(text)
        values = (C.c_uint32 * len(normalized))(*map(ord, normalized))
        lengths = (C.c_uint32 * (len(normalized) + 1))()
        count = lib.qwen_test_piece_lengths(values, len(values), lengths, len(lengths))
        assert list(lengths[:count]) == [end-start for _, (start, end) in split.pre_tokenize_str(normalized)], repr(text)
        if len(fixtures) < 24:
            fixtures.append({'text': text, 'host_reference_ids': expected,
                             'host_c_encoder_ids': list(output[:written.value]), 'fpga_ids': None})
    # End-to-end text path on HOST, with embedded NUL and special spellings.
    chat_checks = 0
    chat_fixtures = []
    for system, text, thinking in itertools.product((None, '', 'You are helpful.\0你好'), corpus[:24], (False, True)):
        messages = [] if system is None else [{'role': 'system', 'content': system}]
        messages.append({'role': 'user', 'content': text})
        expected_text = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        buffer = C.create_string_buffer(len(expected_text.encode()) + 1)
        used = C.c_size_t()
        sb, ub = (system or '').encode(), text.encode()
        assert lib.qwen_chat_single_turn(buffer, len(buffer), C.byref(used), C.c_char_p(sb), len(sb), system is not None, C.c_char_p(ub), len(ub), thinking) == 0
        rendered = bytes(buffer)[:used.value]
        assert rendered == expected_text.encode()
        ids = (C.c_uint32 * (len(rendered) + 1))()
        count = C.c_size_t()
        assert lib.qwen_encode(C.byref(r), buffer, used.value, ids, len(ids), C.byref(count)) == 0
        expected_ids = hf.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking, return_dict=False)
        assert list(ids[:count.value]) == expected_ids
        if system is None and text in ('Hello, world!', '你好，世界！', 'Å', 'A\u0327\u0301'):
            chat_fixtures.append({'user_text': text, 'enable_thinking': thinking,
                                  'host_reference_prompt_ids': expected_ids,
                                  'host_c_prompt_ids': list(ids[:count.value]),
                                  'fpga_prompt_ids': None})
        chat_checks += 1
    # Exact-capacity success, output overflow, malformed UTF-8, input overflow,
    # and normalization expansion overflow. A failure must never look truncated.
    rejected = 0
    malformed = [b'\xc0\xaf', b'\xc1\xbf', b'\xe0\x80\xaf', b'\xed\xa0\x80',
                 b'\xf0\x80\x80\xaf', b'\xf4\x90\x80\x80', b'\xf5\x80\x80\x80',
                 b'\xe4\xb8', b'abc\xc2', b'a\xffb', b'\x80', b'\xe4x\xad']
    for value in malformed + [b'1'*8193, b'1'*32769, '\u0344'.encode()*4097]:
        ids = (C.c_uint32 * 32768)(); count = C.c_size_t(999)
        assert lib.qwen_encode(C.byref(r), C.c_char_p(value), len(value), ids, len(ids), C.byref(count)) == -1, value[:30]
        assert count.value == 0
        rejected += 1
    for text in ('', 'Hello!', '你好', '1'*8192):
        value = text.encode(); expected = official.encode(text, add_special_tokens=False).ids
        ids = (C.c_uint32 * (len(expected)+1))(); count = C.c_size_t()
        assert lib.qwen_encode(C.byref(r), C.c_char_p(value), len(value), ids, len(expected), C.byref(count)) == 0
        assert list(ids[:count.value]) == expected
        if expected:
            assert lib.qwen_encode(C.byref(r), C.c_char_p(value), len(value), ids, len(expected)-1, C.byref(count)) == -1
            assert count.value == 0
            rejected += 1
    report = {'status': 'PASS', 'execution': 'host only; portable C; no FPGA/UART/model execution',
              'board_tokenizer_verified': False, 'tokenizers_version': tokenizers.__version__,
              'encoder_cases': len(corpus), 'identical_encoder_cases': len(corpus),
              'nfc_cases': len(normal_cases), 'regex_boundary_cases': len(corpus),
              'chat_template_then_encoder_cases': chat_checks, 'rejected_inputs': rejected,
              'resource_struct_size': C.sizeof(Resource), 'tokenizer_blob_sha256': hashlib.sha256(raw).hexdigest(),
              'fixtures': fixtures, 'chat_fixtures': chat_fixtures, 'compile_command': cmd,
              'sources': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [root/'text/tokenizer_encode.c', root/'text/tokenizer_resource.c',
                                       root/'text/tokenizer_resource.h', root/'text/unicode_tables.c',
                                       root/'tools/check_tokenizer_encode.py']}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'PASS host only: {len(corpus)} encoder+regex, {len(normal_cases)} NFC, {chat_checks} chat paths, {rejected} rejected inputs')


if __name__ == '__main__':
    main()
