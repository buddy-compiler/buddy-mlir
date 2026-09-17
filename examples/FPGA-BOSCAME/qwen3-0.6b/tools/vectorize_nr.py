#!/usr/bin/env python3
"""Apply Buddy's real matmul vectorization passes to NR FP32 kernels.

The source remains linalg. This driver invokes compiler passes and validates
that the requested FP32 matrix work was lowered into vector FMA; it does not
replace kernels with handwritten C or assembly implementations.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

KINDS = frozenset({'matmul_f32', 'attention_qk', 'attention_pv'})


def transpose_strided_b(source):
    """Expose Triton's physical [N,K] B layout to Buddy's decode pass.

    Triton bufferization represents the same storage as logical [K,N] with
    strides [1,row_stride]. A memref.transpose changes only its descriptor, so the
    transpose-B matmul pass can load contiguous K vectors without packing.
    This intentionally accepts only the canonical, static FP32 layout emitted
    by our Triton exporter and fails closed if its syntax or layout changes.
    """
    pattern = re.compile(
        r'^(?P<indent>\s*)linalg\.matmul ins\('
        r'(?P<a>%[\w.$-]+), (?P<b>%[\w.$-]+) : '
        r'(?P<atype>memref<[^\n]+?>), '
        r'(?P<btype>memref<(?P<k>\d+)x(?P<n>\d+)xf32, '
        r'strided<\[1, (?P<stride>\d+)\], offset: (?P<offset>\?|\d+)>>)\) '
        r'(?P<outs>outs\([^\n]+)$', re.MULTILINE)
    count = 0

    def replace(match):
        nonlocal count
        k, n, stride = (int(match[key]) for key in ('k', 'n', 'stride'))
        if stride < k:
            raise ValueError(f'B row stride {stride} overlaps a K={k} subtile')
        name = f'%nr_b_transpose_{count}'
        if name in source:
            raise ValueError(f'duplicate generated SSA name {name}')
        count += 1
        transposed = (f'memref<{n}x{k}xf32, strided<[{stride}, 1], '
                      f'offset: {match["offset"]}>>')
        maps = ('[affine_map<(m,n,k)->(m,k)>, '
                'affine_map<(m,n,k)->(n,k)>, affine_map<(m,n,k)->(m,n)>]')
        return (f'{match["indent"]}{name} = memref.transpose {match["b"]} '
                f'(d0, d1) -> (d1, d0) : {match["btype"]} to {transposed}\n'
                f'{match["indent"]}linalg.matmul indexing_maps = {maps} '
                f'ins({match["a"]}, {name} : {match["atype"]}, {transposed}) '
                f'{match["outs"]}')

    result = pattern.sub(replace, source)
    if not count or re.search(r'\blinalg\.matmul\s+ins\(', result):
        raise ValueError('expected static FP32 matmul B with strides [1,row_stride>=K]')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-llvm', action='store_true')
    parser.add_argument('--transpose-strided-b', action='store_true')
    args = parser.parse_args()
    if args.transpose_strided_b:
        args.output.write_text(transpose_strided_b(args.input.read_text()))
        return
    if args.metadata is None:
        parser.error('--metadata is required unless --transpose-strided-b is used')
    metadata = json.loads(args.metadata.read_text())
    if metadata.get('kind') not in KINDS:
        args.output.write_text(args.input.read_text())
        return
    if args.prepare_llvm:
        # LLVM may otherwise vectorize the C ABI's small i64 descriptor
        # aggregate loads/stores, independently of source loop vectorization.
        # noimplicitfloat leaves integer ABI marshaling scalar while the
        # linalg-derived kernel retains explicit RVV computation.
        lines = []
        for line in args.input.read_text().splitlines():
            if line.startswith('define ') and '@_mlir_ciface_' in line and 'noimplicitfloat' not in line:
                line = re.sub(r'(\))(?=\s*(?:#[0-9]+\s*)?\{)', r'\1 noimplicitfloat', line)
            lines.append(line)
        args.output.write_text('\n'.join(lines) + '\n')
        return
    command = shlex.split(os.environ['QWEN_CFG_BUDDY_OPT'])
    option = 'FP32_PASS' if metadata['kind'] == 'matmul_f32' else 'BATCH_FP32_PASS'
    flags = shlex.split(os.environ['QWEN_CFG_' + option])
    subprocess.run([*command, str(args.input), *flags, '--canonicalize', '--cse',
                    '-o', str(args.output)], check=True)
    result = args.output.read_text()
    if 'vector.fma' not in result or re.search(r'\blinalg\.(?:matmul|batch_matmul)', result):
        args.output.unlink(missing_ok=True)
        raise SystemExit('NR FP32 vectorization did not replace every matrix operation with vector FMA')


if __name__ == '__main__':
    main()
