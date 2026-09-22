#!/usr/bin/env python3
"""Validate and atomically archive a completed fixed-trajectory FPGA model run.

Requires a schema-1 independent NR numeric oracle. Optional intermediate probes
establish only their selected adapter boundaries. Large weight/reference arrays
are identified by streaming hashes, not duplicated. Source/IR/object snapshots
are evidence, not a substitute for the numerical and optional profile checkers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib

from prepare_model_run import check_arena, check_image_bytes, digest

REPO = Path(__file__).resolve().parents[5]
TOOLS = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def artifact(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def safe_file(directory, name):
    require(isinstance(name, str) and Path(name).name == name and name not in ('.', '..'),
            'segment filename must be a single safe component')
    path = directory / name
    require(not path.is_symlink() and path.is_file(), 'missing/symlink segment: ' + str(path))
    return path


def verify_run_identity(run, prepared, elf_hash, uart):
    result, manifest = read(run/'result.json'), read(run/'run-manifest.json')
    deployment = read(prepared/'deployment.json')
    plan_path = prepared/'ddr-load.plan'
    plan = tomllib.loads(plan_path.read_text())
    require(result.get('status') == 'OK' and result.get('ddr_readback_matches') is True
            and result.get('completion_marker_seen') is True, 'run incomplete or DDR readback failed')
    require(result.get('uart_bytes') == len(uart), 'UART byte count differs from completed worker result')
    require(deployment.get('elf_sha256') == elf_hash, 'deployment ELF hash mismatch')
    require(manifest.get('plan_sha256') == digest(plan_path), 'uploaded plan hash mismatch')
    require(plan.get('version') == 1 and plan.get('ddr_base') == 0x80000000
            and plan.get('ddr_size') == 0x400000000 and plan.get('alignment') == 64,
            'unexpected DDR plan header')
    segments = plan.get('segments', [])
    require(segments and segments == [{k: v for k, v in s.items() if k != 'arena_end'}
                                      for s in deployment.get('segments', [])],
            'deployment segments differ from actual plan')
    names = [s['file'] for s in segments]
    require(len(set(names)) == len(names), 'duplicate DDR segment filename')
    require(manifest.get('segments') == [{k: s[k] for k in ('file', 'size', 'sha256')}
                                        for s in segments], 'uploaded segments differ from plan')
    require(result.get('segment_readbacks') == {n: True for n in names},
            'worker did not verify every exact segment readback')
    require([s['name'] for s in segments].count('model') == 1, 'missing/duplicate model segment')
    require([s['name'] for s in segments].count('weights') == 1, 'missing/duplicate weight segment')
    checks, spans = [], []
    for s in segments:
        p = safe_file(prepared, s['file'])
        require(p.stat().st_size == s['size'] and digest(p) == s['sha256'],
                'prepared segment bytes differ from uploaded manifest: ' + s['file'])
        begin, end = s['address'], s['address'] + s['size']
        require(s['size'] > 0 and s['size'] % 64 == 0 and begin % 64 == 0
                and 0x80000000 <= begin < end <= 0x100000000
                and not (begin < 0xb8000000 and end > 0xb0000000), 'invalid NR segment bounds')
        spans.append((begin, end))
        actual_readback = run/(s['file'] + '.readback')
        downloaded = actual_readback.exists()
        if downloaded:
            require(not actual_readback.is_symlink() and actual_readback.stat().st_size == s['size']
                    and digest(actual_readback) == s['sha256'], 'downloaded DDR readback mismatch')
        checks.append({'file': s['file'], 'sha256': s['sha256'], 'worker_readback_verified': True,
                       'downloaded_readback_rehashed': downloaded})
        if s['name'] == 'model':
            require(s['sha256'] == result.get('sha256'), 'worker boot hash mismatch')
    spans.sort()
    require(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), 'overlapping DDR segments')
    return result, deployment, segments, checks


def elf_blob_matches(elf, address, blob):
    """Validate the exact oracle bytes in one file-backed PT_LOAD mapping."""
    with elf.open('rb') as f:
        header = f.read(64)
        require(len(header) == 64 and header[:6] == b'\x7fELF\x02\x01', 'expected ELF64 little-endian')
        require(struct.unpack_from('<H', header, 18)[0] == 243, 'expected RISC-V ELF')
        phoff = struct.unpack_from('<Q', header, 32)[0]
        size, count = struct.unpack_from('<HH', header, 54)
        require(size >= 56 and count > 0, 'missing ELF program headers')
        mappings = []
        for i in range(count):
            f.seek(phoff + i * size)
            data = f.read(56)
            require(len(data) == 56, 'truncated ELF program header')
            ty, flags, offset, vaddr, _, filesz, _, _ = struct.unpack('<IIQQQQQQ', data)
            if ty == 1 and vaddr <= address and address + blob.stat().st_size <= vaddr + filesz:
                mappings.append(offset + address - vaddr)
        require(len(mappings) == 1, 'oracle not in exactly one file-backed ELF load segment')
        f.seek(mappings[0])
        with blob.open('rb') as reference:
            for block in iter(lambda: reference.read(1 << 20), b''):
                require(f.read(len(block)) == block, 'oracle bytes embedded in ELF differ from reference blob')
    return mappings[0]


def verify_reference(manifest, metadata, arrays, blob, *, layers, prefill, steps):
    import numpy as np
    for key, value in {'schema_version': 1, 'layers': layers, 'prefill_len': prefill,
                       'decode_steps': steps, 'vocab_size': 151936,
                       'arithmetic_profile': 'nr-fpga', 'trajectory_verified': True}.items():
        require(manifest.get(key) == value, 'numeric manifest mismatch: ' + key)
    require(manifest.get('source_sha256') == digest(arrays), 'reference NPZ hash mismatch')
    require(manifest.get('blob_sha256') == digest(blob) and manifest.get('bytes') == blob.stat().st_size,
            'reference blob hash/size mismatch')
    require(metadata.get('layers') == layers and metadata.get('capacity') == manifest.get('capacity')
            and metadata.get('arithmetic_profile') == 'nr-fpga'
            and metadata.get('prompt_ids') == manifest.get('prompt_ids')
            and len(manifest['prompt_ids']) == prefill, 'reference metadata configuration mismatch')
    n = prefill + steps
    with np.load(arrays) as data:
        logits = [data['prefill_logits'].reshape(-1, 151936)[-1]]
        logits += [data[f'decode_logits_{i}'].reshape(151936) for i in range(steps)]
        tokens = [int(np.argmax(x)) for x in logits]
        packed = list(logits)
        for name in ('kv_key_used', 'kv_value_used'):
            a = data[name]
            if a.shape == (layers, n, 8, 128):
                a = a.transpose(0, 2, 1, 3)
            require(a.shape == (layers, 8, n, 128), 'reference cache dimensions mismatch')
            packed.append(a)
        digest_packed = hashlib.sha256()
        for a in packed:
            a = np.asarray(a, dtype='<f4')
            require(bool(np.isfinite(a).all()), 'reference contains nonfinite values')
            digest_packed.update(a.tobytes(order='C'))
    require(digest_packed.hexdigest() == manifest['blob_sha256'], 'NPZ contents do not reconstruct oracle blob')
    require(metadata.get('prefill_argmax_last') == tokens[0], 'reference prefill argmax mismatch')
    trajectory = metadata.get('decode_steps_recorded', [])
    require(len(trajectory) >= steps, 'reference lacks decode steps')
    for i, row in enumerate(trajectory[:steps]):
        require(all(row.get(k) == v for k, v in {'step': i, 'cache_position': prefill+i,
            'input_token': tokens[i], 'generated_token': tokens[i+1]}.items()),
            'reference token/cache trajectory mismatch')
    return tokens


def find_input(build_record, path):
    matches = [value for name, value in build_record.get('input_sha256', {}).items()
               if (Path(name) if Path(name).is_absolute() else REPO/ name).resolve() == path.resolve()]
    require(matches == [digest(path)], 'build-time input hash missing/mismatched: ' + str(path))


def profile_identity(image, plan, build_record, adapters):
    """Verify all profiler variants against their actual compiled source."""
    manifest_path = image/'kernel-profile.json'
    source_path = image/'kernel-profile.c'
    if not plan.get('profile_kernels'):
        require(not any(path.exists() or path.is_symlink() for path in (manifest_path, source_path)),
                'profile evidence absent from exact image plan')
        require(not any(plan.get(name) for name in
                        ('profile_progress', 'profile_probe', 'profile_tile_probe', 'profile_watch')),
                'profile options require enabled profiling in exact image plan')
        return None
    from check_kernel_profile import verify_profile_build
    manifest = read(safe_file(image, manifest_path.name))
    identity = verify_profile_build(manifest, plan, build_record,
        safe_file(image, source_path.name), adapters, repo_root=REPO)
    return {'manifest_sha256': digest(manifest_path), **identity}


def decode_run_uart(image, plan, build_record, adapters, uart):
    """Bind observer evidence to the image before recovering split RA lines."""
    from decode_hang_watch import decode_bytes
    from hang_watch import validate_watch
    manifest_path = image/'hang-watch.json'
    selection = plan.get('hang_watch')
    capacity = plan.get('console', {}).get('capacity_bytes', 65536)
    clean, report = decode_bytes(uart, console_capacity=capacity)
    if plan.get('profile_watch'):
        require(not selection and not manifest_path.exists(), 'conflicting watch configurations')
        require(plan.get('profile_kernels') and plan.get('profile_progress'),
                'profile-watch requires enabled progress profiling')
        profile_path = safe_file(image, 'kernel-profile.json')
        profile = read(profile_path)
        watch = validate_watch(adapters, plan.get('profile_probe'))
        require(profile.get('phase_probe') == watch and
                profile.get('nh_watch', {}).get('selection') == watch,
                'profile-watch selection differs from image plan')
        source = safe_file(image, 'kernel-profile.c')
        require(profile.get('source_sha256') == digest(source) and
                profile.get('adapter_sha256') == digest(adapters), 'profile-watch source hash mismatch')
        for path in (source, adapters):
            find_input(build_record, path)
        require(report['status'] == 'FRAMES_DECODED' and report['frame_count'] > 0
                and not report['uart_acceptance_incomplete'] and b'\x1eNRWATCH' not in clean,
                'profile-watch UART missing valid complete NH frames')
        return clean, report, {
            'kind': 'profile-watch', 'manifest_sha256': digest(profile_path),
            'generated_source_sha256': digest(source), 'adapter_sha256': digest(adapters),
            'watch': watch, 'reset_symbol': 'qwen_profile_reset',
            'raw_sha256': report['raw_sha256'], 'ra_sha256': report['ra_sha256'],
            'diagnostic_only': True, 'throughput_acceptance': 'NOT_EVALUATED'}
    if not selection:
        require(not manifest_path.exists() and not manifest_path.is_symlink(),
                'hang-watch manifest absent from exact image plan')
        require(report['status'] == 'NO_DIAGNOSTIC_FRAMES'
                and not report['uart_acceptance_incomplete'] and clean == uart
                and b'\x1eNRWATCH' not in uart,
                'UART has NH diagnostic framing absent from exact image plan')
        return uart, None, None
    manifest = read(safe_file(image, manifest_path.name))
    source = safe_file(image, 'hang-watch.c')
    watch = validate_watch(adapters, selection)
    require(manifest.get('schema_version') == 1 and manifest.get('watch') == watch,
            'hang-watch manifest selection/schema mismatch')
    require(manifest.get('argument_types') == ['MemRef2 *'] * 3
            and manifest.get('return_type') == 'void'
            and manifest.get('reset_symbol') == 'qwen_hang_reset'
            and manifest.get('linker_flags') == ['--wrap=' + watch['symbol']],
            'hang-watch manifest ABI/link contract mismatch')
    require(manifest.get('adapter_sha256') == digest(adapters),
            'hang-watch adapter hash mismatch')
    require(manifest.get('source_sha256') == digest(source),
            'hang-watch generated source hash mismatch')
    for path in (source, adapters):
        find_input(build_record, path)
    require(report['status'] == 'FRAMES_DECODED' and report['frame_count'] > 0
            and not report['uart_acceptance_incomplete'],
            'hang-watch UART missing frames or has dropped/truncated/inconsistent observations')
    # Reserved markers may not survive outside complete diagnostic frames.
    require(b'\x1eNRWATCH' not in clean, 'malformed NH diagnostic marker in recovered UART')
    identity = {'manifest_sha256': digest(manifest_path),
                'generated_source_sha256': digest(source),
                'adapter_sha256': digest(adapters), 'watch': watch,
                'raw_sha256': report['raw_sha256'], 'ra_sha256': report['ra_sha256'],
                'diagnostic_only': True, 'throughput_acceptance': 'NOT_EVALUATED'}
    return clean, report, identity


def write_decoded_uart(directory, uart, report):
    """Keep recovered UART separate from raw worker/run identity evidence."""
    path = directory/'uart.ra.log'
    path.write_bytes(uart)
    (directory/'nh-watch.json').write_text(json.dumps(report, indent=2) + '\n')
    return path


def intermediate_manifest(image, plan, uart):
    path = image/'intermediate-probe.json'
    if not path.exists():
        require(not path.is_symlink(), 'missing/symlink intermediate manifest')
        require(not plan.get('intermediate_probe'), 'image plan declares a missing intermediate manifest')
        require(b'[intermediate' not in uart and b'intermediate]' not in uart,
                'UART has intermediates absent from exact image manifest')
        return None
    manifest = read(safe_file(image, path.name))
    # Earlier instrumented images predate the optional image-plan field.
    if 'intermediate_probe' in plan:
        require(plan['intermediate_probe'] == manifest, 'image plan intermediate manifest mismatch')
    from check_intermediates import validate_manifest
    validate_manifest(manifest)
    return manifest


def intermediate_sources(manifest):
    sources = [(Path(name) if Path(name).is_absolute() else REPO/name, expected)
               for name, expected in manifest['source_sha256'].items()]
    for path, expected in sources:
        require(not path.is_symlink() and path.is_file() and digest(path) == expected,
                'intermediate source hash missing/mismatched: ' + str(path))
    arrays = [path for path, _ in sources if path.suffix == '.npz']
    require(len(arrays) == 1, 'intermediate manifest must identify exactly one reference NPZ')
    metadata = arrays[0].with_name('quant-reference.json')
    find_input({'input_sha256': manifest['source_sha256']}, metadata)
    return arrays[0], metadata, [path for path, _ in sources]


def verify_intermediate_reference(manifest, arrays, blob, source, metadata_path):
    """Recreate every generated float from the independently captured tensors."""
    import numpy as np
    from check_intermediates import validate_manifest
    validate_manifest(manifest)
    inputs = {'input_sha256': manifest['source_sha256']}
    for path in (arrays, metadata_path):
        find_input(inputs, path)
    require(digest(source) == manifest['generated_source_sha256'], 'intermediate generated source hash mismatch')
    require(blob.stat().st_size == manifest['reference_bytes']
            and digest(blob) == manifest['reference_blob_sha256'], 'intermediate reference blob hash/size mismatch')
    metadata = read(metadata_path)
    prompt = metadata.get('prompt_ids')
    require(metadata.get('layers') == manifest['layers'] and metadata.get('arithmetic_profile') == 'nr-fpga'
            and isinstance(prompt, list) and len(prompt) == manifest['prefill_len'],
            'intermediate reference metadata configuration mismatch')
    packed, cursor, covered = hashlib.sha256(), 0, set()
    with np.load(arrays, allow_pickle=False) as data:
        for entry in manifest['entries']:
            shape = tuple(entry['shape'])
            for key, offset in zip(entry['reference_keys'], entry['offsets']):
                require(key in data, 'missing intermediate reference tensor: ' + key)
                value = data[key]
                prefix = key.removesuffix(entry['reference_suffix'])
                for alias in entry.get('reference_aliases', []):
                    alias_key = prefix + alias
                    require(alias_key in data and np.array_equal(value, data[alias_key]),
                            'shared intermediate reference mismatch: ' + alias_key)
                dtype = np.int8 if entry['dtype'] == 'i8' else np.float32
                require(value.dtype == dtype, 'intermediate reference dtype mismatch: ' + key)
                transform = entry['transform']
                if transform == 'last_row':
                    require(value.ndim >= 1 and value.shape[0] > 0, 'intermediate last-row extent mismatch: ' + key)
                    value = value[-1:]
                elif transform == 'heads_first':
                    require(value.ndim == 3, 'intermediate head layout mismatch: ' + key)
                    value = value.transpose(1, 0, 2)
                elif transform == 'context_heads_first':
                    require(value.ndim == 2 and value.shape[1] == 16 * 128,
                            'intermediate context layout mismatch: ' + key)
                    value = value.reshape(value.shape[0], 16, 128).transpose(1, 0, 2)
                elif transform == 'projection_heads':
                    require(len(shape) == 4 and shape[0] == 1 and value.shape == (shape[1], shape[2] * shape[3]),
                            'intermediate projection layout mismatch: ' + key)
                    value = value.reshape(shape[1:])
                require(value.shape == shape or (shape[0] == 1 and value.shape == shape[1:]),
                        'intermediate reference shape mismatch: ' + key)
                require(value.size == entry['elements'] and bool(np.isfinite(value).all()),
                        'intermediate reference extent/nonfinite mismatch: ' + key)
                require(offset == cursor, 'intermediate reference packing offset mismatch')
                packed.update(value.astype('<f4').tobytes(order='C'))
                cursor += value.size
                covered.add(key)
        uncovered = sorted(key for key in data.files
                           if (key.startswith('prefill_model.') or key.startswith('decode_'))
                           and '_logits' not in key and '_kv_' not in key and key not in covered)
    require(uncovered == manifest['uncovered_reference_tensors'], 'intermediate uncovered tensor list mismatch')
    require(cursor * 4 == manifest['reference_bytes'] and packed.hexdigest() == manifest['reference_blob_sha256'],
            'intermediate NPZ contents do not reconstruct reference blob')
    return metadata


def run_validator(command, output):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output.with_suffix('.validator.log')).write_text('$ ' + ' '.join(map(str, command)) + '\n' + result.stdout)
    require(result.returncode == 0 and output.is_file(), 'validator failed: ' + result.stdout[-2000:])
    return read(output)


def archive(args):
    require(not args.output.exists() and not args.output.is_symlink(),
            'refusing to overwrite existing archive: ' + str(args.output))
    require(1 <= args.layers <= 28 and args.prefill > 0 and args.steps > 0, 'invalid validation dimensions')
    image, build, prepared, run = args.image_dir, args.build, args.prepared_dir, args.run
    elf, raw = image/'qwen_model.elf', image/'qwen_model.bin'
    build_record, plan, audit = read(image/'image.json'), read(image/'w8a8-image-plan.json'), read(image/'elf-audit.json')
    require(build_record.get('status') == 'PASS' and build_record.get('undefined_symbols') == [],
            'image build/audit did not pass')
    require(audit.get('status') == 'PASS' and audit.get('elf_sha256') == digest(elf), 'ELF ISA audit mismatch')
    require(plan.get('layers') == args.layers and not plan.get('interactive'), 'image dimensions/mode mismatch')
    require(plan.get('numeric_reference') == read(image/'numeric-reference.json'), 'image plan oracle mismatch')
    raw_uart = (run/'uart.raw.log').read_bytes()
    result, deployment, segments, readbacks = verify_run_identity(run, prepared, digest(elf), raw_uart)
    adapters = build/'replacement/qwen_triton_adapters.c'
    profiling_identity = profile_identity(image, plan, build_record, adapters)
    uart, watch_report, watch_identity = decode_run_uart(image, plan, build_record, adapters, raw_uart)
    probe = intermediate_manifest(image, plan, uart)
    if probe and probe.get('schema_version') == 2:
        for name in ('intermediate-probe.c', 'intermediate-reference.bin'):
            find_input(build_record, safe_file(image, name))
    symbols_raw = subprocess.check_output([str(args.nm), '--defined-only', str(elf)], text=True)
    symbols = {p[2]: int(p[0], 16) for line in symbols_raw.splitlines() if len(p := line.split()) == 3}
    require(symbols.get('_start') == 0x80000000, 'unexpected ELF boot address')
    if watch_identity is not None:
        require('__wrap_' + watch_identity['watch']['symbol'] in symbols
                and watch_identity.get('reset_symbol', 'qwen_hang_reset') in symbols,
                'hang-watch wrapper/reset missing from ELF')
    require(0xb8000000 <= symbols['__workspace_start'] < symbols['__workspace_end'] <= 0x100000000,
            'ELF workspace outside NR bounds')
    for s in segments:
        if s['name'] in ('weights', 'tokenizer'):
            stem = 'weight_arena' if s['name'] == 'weights' else 'tokenizer_blob'
            begin, end = check_arena(symbols, stem, s['size'])
            require(s['address'] == begin and s['address']+s['size'] == end, 'segment differs from exact ELF arena')
        else:
            require(s['name'] == 'model' and s['address'] == symbols['_start'], 'unknown deployment segment')
    require(bool(plan.get('tokenizer_bytes')) == any(s['name'] == 'tokenizer' for s in segments),
            'tokenizer resource presence mismatch')
    manifest = read(image/'numeric-reference.json')
    metadata_path = args.reference_metadata or args.quant_reference.with_name('quant-reference.json')
    require(manifest.get('metadata_sha256') == digest(metadata_path), 'reference metadata hash mismatch')
    tokens = verify_reference(manifest, read(metadata_path), args.quant_reference,
        image/'numeric-reference.bin', layers=args.layers, prefill=args.prefill, steps=args.steps)
    offset = elf_blob_matches(elf, symbols['model_reference_raw'], image/'numeric-reference.bin')
    replacement = build/'replacement/triton-call-replacement.json'
    replacement_report = read(replacement)
    for kind in ('prefill', 'decode'):
        abi = replacement_report['graphs'][kind]['entry_abi']
        require(abi.get('result_descriptor_count') == 3 * args.layers + 1
                and len(abi.get('outputs', [])) == 3 * args.layers + 1,
                kind + ': actual graph result ABI does not match expected layer count')
    irs = [build/f'nr-{kind}/forward_{kind}.ll' for kind in ('prefill', 'decode')]
    library = build/'model-lib/libqwen_triton.a'
    for path in [adapters, *irs, library]:
        find_input(build_record, path)
    probe_identity, probe_arrays, probe_metadata_path, probe_sources = None, None, None, []
    if probe is not None:
        require(all(probe.get(k) == v for k, v in
                    {'layers': args.layers, 'prefill_len': args.prefill, 'decode_steps': args.steps}.items()),
                'intermediate dimensions differ from numeric validation')
        probe_arrays, probe_metadata_path, probe_sources = intermediate_sources(probe)
        probe_blob = safe_file(image, 'intermediate-reference.bin')
        probe_metadata = verify_intermediate_reference(probe, probe_arrays, probe_blob,
            safe_file(image, 'intermediate-probe.c'), probe_metadata_path)
        numeric_metadata = read(metadata_path)
        for key in ('layers', 'capacity', 'prompt_ids', 'arithmetic_profile', 'config',
                    'quantization', 'floating_arithmetic', 'checkpoint', 'source_sha256'):
            require(probe_metadata.get(key) == numeric_metadata.get(key),
                    'intermediate/numeric reference identity mismatch: ' + key)
        # Trace NPZs may add hidden tensors to a separate reference run. Require
        # their logits, effective KV, and token trajectory to reconstruct the
        # very same numerical oracle already verified above.
        trace_numeric = {**manifest, 'source_sha256': digest(probe_arrays)}
        verify_reference(trace_numeric, probe_metadata, probe_arrays, image/'numeric-reference.bin',
                         layers=args.layers, prefill=args.prefill, steps=args.steps)
        for path in (adapters, replacement, *irs):
            find_input({'input_sha256': probe['source_sha256']}, path)
        require('intermediate_reference_raw' in symbols, 'missing embedded intermediate reference symbol')
        probe_offset = elf_blob_matches(elf, symbols['intermediate_reference_raw'], probe_blob)
        probe_identity = {'manifest_sha256': digest(image/'intermediate-probe.json'),
                          'generated_source_sha256': probe['generated_source_sha256'],
                          'reference_blob_sha256': probe['reference_blob_sha256'],
                          'reference_npz': artifact(probe_arrays),
                          'reference_metadata': artifact(probe_metadata_path),
                          'embedded_address': hex(symbols['intermediate_reference_raw']),
                          'elf_offset': probe_offset, 'npz_repacked_and_embedded_bytes_match': True,
                          'numeric_reference_and_trajectory_match': True,
                          'plan_manifest_bound': 'intermediate_probe' in plan}
    library_manifest = read(build/'model-lib/archive.json')
    require(library_manifest.get('sha256') == digest(library)
            and library_manifest.get('contains_test_launch') is False
            and library_manifest.get('contains_runtime') is False, 'Triton archive identity/content audit mismatch')
    for case in library_manifest.get('cases', []):
        for name, expected in case['files_sha256'].items():
            source = (build/'model-lib'/(case['case'] + '.' + name)
                      if name in ('kernel.o', 'adapter.o') else
                      build/'model-lib/evidence'/case['case']/name)
            require(source.is_file() and digest(source) == expected,
                    'archive kernel compilation evidence mismatch: ' + str(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.model-archive-', dir=args.output.parent) as temp:
        dest = Path(temp)
        validator_uart = (write_decoded_uart(dest, uart, watch_report)
                          if watch_report is not None else run/'uart.raw.log')
        extracted = dest/'elf-objcopy.bin'
        subprocess.run([str(args.objcopy), '-O', 'binary', str(elf), str(extracted)], check=True)
        check_image_bytes(raw, extracted)
        boot = prepared/next(s['file'] for s in segments if s['name'] == 'model')
        check_image_bytes(boot, extracted)
        require(deployment.get('elf_objcopy_bytes') == extracted.stat().st_size, 'deployment raw ELF length mismatch')
        extracted.unlink()
        numeric = run_validator([sys.executable, str(TOOLS/'check_board_trace.py'), '--uart', str(validator_uart),
            '--host-graph', str(args.host_graph), '--quant-reference', str(args.quant_reference),
            '--embedded-reference', str(image/'numeric-reference.json'), '--layers', str(args.layers),
            '--prefill', str(args.prefill), '--steps', str(args.steps), '--output', str(dest/'numeric-verification.json')], dest/'numeric-verification.json')
        require(numeric.get('status') == 'FULL_LOGITS_KV_PASS', 'full numerical validation missing')
        intermediates = None
        if probe is not None:
            intermediates = run_validator([sys.executable, str(TOOLS/'check_intermediates.py'),
                '--uart', str(validator_uart), '--manifest', str(image/'intermediate-probe.json'),
                '--output', str(dest/'intermediate-verification.json')], dest/'intermediate-verification.json')
            require(intermediates.get('status') == 'INTERMEDIATES_PASS', 'intermediate verification did not pass')
        profile = None
        if plan.get('profile_kernels'):
            profile = run_validator([sys.executable, str(TOOLS/'check_kernel_profile.py'), '--uart', str(validator_uart),
                '--profile', str(image/'kernel-profile.json'), '--replacement', str(replacement), '--adapters', str(adapters),
                '--image-plan', str(image/'w8a8-image-plan.json'), '--image-build', str(image/'image.json'),
                '--prefill-ir', str(irs[0]), '--decode-ir', str(irs[1]), '--prefill', str(args.prefill), '--steps', str(args.steps),
                '--output', str(dest/'kernel-profile-verification.json')], dest/'kernel-profile-verification.json')
            require(profile.get('status') == 'KERNEL_PROFILE_PASS', 'kernel profiling did not pass')
        elif b'[profile]' in uart:
            raise ValueError('UART has profiling absent from exact image plan')
        text_report = None
        if plan.get('fixed_prompt') is not None:
            require(args.assets is not None, 'fixed-text run requires --assets official tokenizer oracle')
            # The separate checker owns official tokenizer/template/decode checks.
            text_report = run_validator([sys.executable, str(TOOLS/'check_fixed_text.py'), '--uart', str(validator_uart),
                '--image-plan', str(image/'w8a8-image-plan.json'), '--assets', str(args.assets),
                '--output', str(dest/'fixed-text-verification.json')], dest/'fixed-text-verification.json')
            require(text_report.get('status') == 'FIXED_TEXT_PASS', 'fixed-text verification did not pass')
        archived = {}
        def copy(source, relative):
            target = dest/relative
            require(not source.is_symlink() and source.is_file(), 'evidence source missing/symlink: '+str(source))
            require(not target.exists(), 'archive path collision: '+str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            archived[str(relative)] = artifact(source)
            require(digest(target) == archived[str(relative)]['sha256'], 'source changed during archive')
        for name in ('result.json','run-manifest.json','uart.raw.log','worker.log','uvhs.log'):
            copy(run/name, Path('run')/name)
        require(digest(dest/'run/uart.raw.log') == hashlib.sha256(raw_uart).hexdigest(),
                'raw UART changed during validation/archive')
        for name in ('deployment.json','ddr-load.plan'):
            copy(prepared/name, Path('deployment')/name)
        for source in sorted(image.iterdir()):
            if source.is_file() and source.suffix in ('.c','.S','.s','.o','.map','.json','.elf'):
                copy(source, Path('image')/source.name)
        for folder in ('replacement','nr-prefill','nr-decode','model-lib'):
            for source in sorted((build/folder).rglob('*')):
                if source.is_file() and source.suffix in ('.c','.h','.ll','.mlir','.ttir','.py','.json','.txt','.map','.s','.o','.a'):
                    copy(source, Path('build')/source.relative_to(build))
        copy(metadata_path, Path('references/quant-reference.json'))
        if probe is not None:
            copy(probe_metadata_path, Path('references/intermediate-quant-reference.json'))
            # Retain mapping inputs even if they live outside the usual build
            # subdirectories (for example, checkpoint weight-layout metadata).
            for index, source in enumerate(probe_sources):
                if source.suffix != '.npz' and source.resolve() not in {
                        Path(item['path']) for item in archived.values()}:
                    copy(source, Path('intermediate-sources')/str(index)/source.name)
        if args.host_graph.with_name('host-run.json').is_file():
            copy(args.host_graph.with_name('host-run.json'), Path('references/host-run.json'))
        sources = []
        for name, expected in build_record['input_sha256'].items():
            source = Path(name) if Path(name).is_absolute() else REPO/name
            matching = source.is_file() and digest(source) == expected
            sources.append({'path': name, 'build_sha256': expected, 'current_source_matches': matching})
            if matching and source.suffix in ('.c','.h','.ld','.inc') and REPO in source.resolve().parents:
                relative = Path('sources')/source.resolve().relative_to(REPO)
                copy(source, relative)
        validator_files = ['archive_model_run.py','check_board_trace.py','check_kernel_profile.py',
                           'prepare_model_run.py']
        if text_report is not None:
            validator_files.append('check_fixed_text.py')
        if probe is not None:
            validator_files.append('check_intermediates.py')
        if watch_report is not None:
            validator_files += ['decode_hang_watch.py', 'hang_watch.py']
        for filename in validator_files:
            copy(TOOLS/filename, Path('validators')/filename)
        large = [args.host_graph, args.quant_reference, image/'numeric-reference.bin', raw]
        if probe is not None:
            large += [image/'intermediate-reference.bin']
            if probe_arrays.resolve() != args.quant_reference.resolve():
                large.append(probe_arrays)
        large += [prepared/s['file'] for s in segments]
        report = {'status': 'MODEL_RUN_NUMERIC_PASS', 'run_id': run.name, 'fpga': result['fpga'],
            'scope': {'layers': args.layers, 'prefill': args.prefill, 'decode_steps': args.steps,
                      'vocab_size': 151936, 'arithmetic_profile': 'nr-fpga'},
            'numerical_status': numeric['status'], 'predicted_tokens': tokens,
            'profile_status': profile['status'] if profile else 'NOT_INSTRUMENTED',
            'intermediate_status': intermediates['status'] if intermediates else 'NOT_INSTRUMENTED',
            'hang_watch_status': watch_report['status'] if watch_report else 'NOT_INSTRUMENTED',
            'intermediate_coverage': ({'scope': 'Partial hidden-state coverage at selected external adapter boundaries; not all hidden tensors.',
                                       'uncovered_reference_tensors': probe['uncovered_reference_tensors'],
                                       'limitations': probe.get('limitations', [])} if probe else None),
            'performance': {'stages': profile['stages'] if profile else
                numeric['fpga_vs_independent_quantized_reference']['stages'],
                'scope': plan.get('cycle_scope', 'graph computation; see raw trace for additional phases')},
            'fixed_text': text_report,
            'identity': {'elf_sha256': digest(elf), 'boot_matches_elf_objcopy': True,
                         'raw_uart_sha256': hashlib.sha256(raw_uart).hexdigest(),
                         'oracle_embedded_address': hex(symbols['model_reference_raw']),
                         'oracle_elf_offset': offset, 'oracle_npz_repacked_and_embedded_bytes_match': True,
                         'intermediate_probe': probe_identity,
                         'kernel_profile': profiling_identity,
                         'hang_watch': watch_identity,
                         'readbacks': readbacks, 'source_checks': sources},
            'limits': [('Full last-position vocabulary logits and effective KV, plus partial hidden-state checks at selected adapter boundaries; not all hidden tensors.'
                        if probe else 'Full last-position vocabulary logits and effective KV only; no hidden-state acceptance.'),
                       'Fixed raw prompt (when enabled) is board tokenized; UART RX and arbitrary conversation are separate.',
                       'Readbacks without local files are attested by the exact completed worker result and uploaded manifest, not rehashed locally.',
                       'A smaller layer count does not establish 28-layer acceptance; profiling is not uninstrumented throughput.'],
            'archived_sources': archived, 'large_artifacts_not_copied': [artifact(p) for p in large]}
        if watch_report is not None:
            report['performance']['throughput_acceptance'] = 'NOT_EVALUATED'
            report['limits'].append('Hang-watch instrumentation changes timing and layout; cycles are diagnostic observations, not uninstrumented throughput. Complete frames do not establish cache freshness or a hang root cause.')
        report['archive_sha256'] = {str(p.relative_to(dest)): digest(p) for p in sorted(dest.rglob('*')) if p.is_file()}
        (dest/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
        require(not args.output.exists() and not args.output.is_symlink(),
                'archive destination appeared during validation')
        dest.rename(args.output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run','build','image-dir','prepared-dir','host-graph','quant-reference','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--prefill', type=int, default=16)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--reference-metadata', type=Path)
    parser.add_argument('--assets', type=Path, help='official tokenizer for post-run fixed-text verification')
    parser.add_argument('--nm', type=Path, default=REPO/'llvm/build-2d26/bin/llvm-nm')
    parser.add_argument('--objcopy', type=Path, default=REPO/'llvm/build-2d26/bin/llvm-objcopy')
    args = parser.parse_args()
    try:
        report = archive(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit('archive rejected: '+str(error))
    print(json.dumps({'status': report['status'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
