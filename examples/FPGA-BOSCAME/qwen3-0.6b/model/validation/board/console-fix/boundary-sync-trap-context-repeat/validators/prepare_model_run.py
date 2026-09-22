#!/usr/bin/env python3
"""Prepare checked DDR segments from the linked image, never guessed addresses.

The boot image must be the exact ELF objcopy output (optional 64-byte zero pad).
Each uploaded resource must fit its own linker-labelled arena, not merely the
entire workspace containing caches and temporaries.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def aligned(size):
    return (size + 63) & ~63


def check_image_bytes(image, extracted):
    """Accept objcopy's bytes, or exactly its canonical 64-byte zero padding."""
    expected_size = extracted.stat().st_size
    actual_size = image.stat().st_size
    if not expected_size or actual_size not in (expected_size, aligned(expected_size)):
        raise ValueError('boot bin length does not match this ELF')
    with image.open('rb') as actual, extracted.open('rb') as expected:
        while block := expected.read(1024 * 1024):
            if actual.read(len(block)) != block:
                raise ValueError('boot bin bytes do not match this ELF; rebuild or select the matching pair')
        if any(actual.read()):
            raise ValueError('boot bin has nonzero alignment padding')
    return expected_size


def check_arena(symbols, stem, size):
    """Return exact [start,end) only when this resource fills its allocated slot."""
    begin_name, end_name = stem + '_raw', stem + '_end'
    if begin_name not in symbols or end_name not in symbols:
        raise ValueError(f'missing {begin_name}/{end_name}; rebuild image with labelled arena ends')
    begin, end = symbols[begin_name], symbols[end_name]
    if (size <= 0 or begin % 64 or end % 64 or
            not symbols['__workspace_start'] <= begin < end <= symbols['__workspace_end']):
        raise ValueError(f'{stem}: invalid linked arena')
    if aligned(size) != end - begin:
        raise ValueError(f'{stem}: resource padded size {aligned(size)} differs from linked arena capacity {end-begin}')
    return begin, end


def main():
    tools = Path(__file__).resolve().parents[5]/'llvm/build-2d26/bin'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--elf', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--weight-manifest', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--nm', type=Path, default=tools/'llvm-nm')
    p.add_argument('--objcopy', type=Path, default=tools/'llvm-objcopy')
    a = p.parse_args()
    manifest = json.loads(a.weight_manifest.read_text())
    if a.weights.stat().st_size != manifest['bytes'] or digest(a.weights) != manifest['sha256']:
        raise ValueError('weight segment does not match its manifest')
    raw = subprocess.check_output([str(a.nm), '--defined-only', str(a.elf)], text=True)
    symbols = {fields[2]:int(fields[0], 16) for line in raw.splitlines()
               if len(fields := line.split()) == 3}
    if symbols.get('_start') != 0x80000000:
        raise ValueError('unexpected boot address')
    if not 0xb8000000 <= symbols['__workspace_start'] < symbols['__workspace_end'] <= 0x100000000:
        raise ValueError('workspace exceeds NR linker address range')
    with tempfile.TemporaryDirectory(prefix='qwen-elf-bin-check-') as temporary:
        extracted = Path(temporary)/'image.bin'
        subprocess.run([str(a.objcopy), '-O', 'binary', str(a.elf), str(extracted)], check=True)
        raw_image_bytes = check_image_bytes(a.image, extracted)
    weight_begin, weight_end = check_arena(symbols, 'weight_arena', manifest['bytes'])
    inputs = [('model', a.image, 'image.bin', symbols['_start'], None),
              ('weights', a.weights, 'weights-w8a8.bin', weight_begin, weight_end)]
    if ('tokenizer_blob_raw' in symbols) != bool(a.tokenizer):
        raise ValueError('tokenizer resource presence differs from linked image')
    if a.tokenizer:
        tok_begin, tok_end = check_arena(symbols, 'tokenizer_blob', a.tokenizer.stat().st_size)
        inputs.append(('tokenizer', a.tokenizer, 'tokenizer.bin', tok_begin, tok_end))
    # Validate every range before copying any potentially large weight file.
    ranges = []
    for name, source, filename, address, end in inputs:
        size = aligned(source.stat().st_size)
        if address % 64 or not 0x80000000 <= address < address+size <= 0x100000000:
            raise ValueError(f'{name} is outside NR address range')
        if address < 0xb8000000 and address+size > 0xb0000000:
            raise ValueError(f'{name} overlaps RA inaccessible aperture')
        if end is not None and address+size != end:
            raise ValueError(f'{name} exceeds its linked arena')
        if (a.output/filename).resolve() == source.resolve():
            raise ValueError('output directory must differ from input file directory')
        ranges.append((address, address+size))
    ranges.sort()
    if any(x[1] > y[0] for x,y in zip(ranges,ranges[1:])):
        raise ValueError('segments overlap')
    a.output.mkdir(parents=True, exist_ok=True)
    segments = []
    for name, source, filename, address, end in inputs:
        dest = a.output/filename
        shutil.copyfile(source, dest)
        size = dest.stat().st_size
        with dest.open('ab') as f:
            f.write(b'\0'*(-size%64))
        item = dict(name=name, file=filename, address=address,
                    size=dest.stat().st_size, sha256=digest(dest))
        if end is not None:
            item['arena_end'] = end
        segments.append(item)
    text = ['# Platform loader DDR window; segments also satisfy the stricter NR linker bounds.',
            'version = 1', 'ddr_base = 0x80000000', 'ddr_size = 0x400000000', 'alignment = 64']
    for s in segments:
        text += ['', '[[segments]]', f'name = "{s["name"]}"', f'file = "{s["file"]}"',
                 f'address = 0x{s["address"]:x}', f'size = {s["size"]}', f'sha256 = "{s["sha256"]}"']
    (a.output/'ddr-load.plan').write_text('\n'.join(text)+'\n')
    (a.output/'deployment.json').write_text(json.dumps({
        'elf':str(a.elf.resolve()), 'elf_sha256':digest(a.elf), 'segments':segments,
        'boot_bin_matches_elf':True, 'elf_objcopy_bytes':raw_image_bytes,
        'per_resource_arena_bounds_verified':True,
        'workspace_end':hex(symbols['__workspace_end']),
        'execution':'prepared only; no hardware result'}, indent=2)+'\n')
    print(a.output/'ddr-load.plan')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError) as error:
        raise SystemExit(str(error))
