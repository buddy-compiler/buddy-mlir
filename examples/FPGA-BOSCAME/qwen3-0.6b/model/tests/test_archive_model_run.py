"""Tamper rejection for reusable model evidence archiving; no board required."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('model_archive', TOOLS/'archive_model_run.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def dump(path, value):
    path.write_text(json.dumps(value))


class ArchiveModelRunTests(unittest.TestCase):
    def setup_intermediates(self, directory):
        """Small independent data exercises every generator layout transform."""
        d = Path(directory)
        arrays, blob, source, metadata = (d/name for name in
            ('arrays.npz', 'intermediate-reference.bin', 'intermediate-probe.c', 'quant-reference.json'))
        source.write_text('/* generated probe test fixture */\n')
        dump(metadata, {'layers': 1, 'capacity': 512, 'arithmetic_profile': 'nr-fpga', 'prompt_ids': list(range(16))})
        entries, data, packed, cursor = [], {}, [], 0
        for index in range(92):
            graph = 'prefill' if index < 46 else 'decode'
            transform = ('identity', 'last_row', 'heads_first', 'context_heads_first', 'projection_heads')[index % 5]
            shape = {'identity': [1, 2, 2], 'last_row': [1, 1, 2], 'heads_first': [2, 1, 3],
                     'context_heads_first': [16, 1, 128], 'projection_heads': [1, 2, 2, 3]}[transform]
            dtype = 'i8' if index % 5 == 0 else 'f32'
            elements = int(np.prod(shape))
            suffix = f'model.layers.0_tensor_{index}'
            entry = {'index': index, 'graph': graph, 'layer': 0, 'symbol': f'_mlir_ciface_qwen_graph_probe_{index}',
                     'operand': 0, 'phase': 'before', 'reference_suffix': suffix, 'transform': transform,
                     'shape': shape, 'dtype': dtype, 'rank': len(shape), 'elements': elements,
                     'reference_keys': [], 'offsets': []}
            for step in range(1 if graph == 'prefill' else 8):
                key = ('prefill' if graph == 'prefill' else f'decode_{step}') + '_' + suffix
                if transform == 'last_row':
                    value = np.arange(6, dtype=np.float32).reshape(3, 2) + step
                    expected = value[-1:]
                elif transform == 'heads_first':
                    value = np.arange(elements, dtype=np.float32).reshape(1, 2, 3) + step
                    expected = np.stack((value[:, 0, :], value[:, 1, :]))
                elif transform == 'context_heads_first':
                    value = np.arange(elements, dtype=np.float32).reshape(1, 2048) + step
                    expected = np.stack([value[:, i*128:(i+1)*128] for i in range(16)])
                elif transform == 'projection_heads':
                    value = np.arange(elements, dtype=np.float32).reshape(2, 6) + step
                    expected = value.reshape(2, 2, 3)
                else:
                    value = np.arange(elements, dtype=np.int8).reshape(2, 2) + step
                    expected = value
                data[key] = value
                packed.append(expected.astype('<f4').tobytes(order='C'))
                entry['reference_keys'].append(key)
                entry['offsets'].append(cursor)
                cursor += elements
            entries.append(entry)
        data['prefill_model.layers.0_unhooked'] = np.zeros(2, dtype=np.float32)
        np.savez(arrays, **data)
        blob.write_bytes(b''.join(packed))
        manifest = {'schema_version': 1, 'layers': 1, 'prefill_len': 16, 'decode_steps': 8,
                    'max_abs_tolerance': 0.001, 'mean_abs_tolerance': 0.0001, 'entries': entries,
                    'reference_bytes': cursor * 4, 'reference_blob_sha256': archive.digest(blob),
                    'generated_source_sha256': archive.digest(source),
                    'source_sha256': {str(p): archive.digest(p) for p in (arrays, metadata)},
                    'uncovered_reference_tensors': ['prefill_model.layers.0_unhooked'],
                    'linker_flags': ['--wrap=' + symbol for symbol in sorted(e['symbol'] for e in entries)]}
        return SimpleNamespace(manifest=manifest, arrays=arrays, blob=blob, source=source, metadata=metadata, data=data)

    def verify_intermediates(self, fixture):
        return archive.verify_intermediate_reference(fixture.manifest, fixture.arrays, fixture.blob,
                                                     fixture.source, fixture.metadata)

    def setup_run(self, directory):
        d = Path(directory); run = d/'run'; prep = d/'prepared';run.mkdir();prep.mkdir()
        uart = b'[nr] RA returned: PASS\r\n'
        segments = []
        for name, file, address in [('model','image.bin',0x80000000),('weights','weights-w8a8.bin',0xb8000000)]:
            (prep/file).write_bytes(bytes([len(name)])*64)
            segments.append(dict(name=name,file=file,address=address,size=64,sha256=archive.digest(prep/file)))
        plan = ['version=1','ddr_base=0x80000000','ddr_size=0x400000000','alignment=64']
        for s in segments:
            plan += ['[[segments]]']+[f'{k}={json.dumps(v)}' for k,v in s.items()]
        (prep/'ddr-load.plan').write_text('\n'.join(plan))
        deployment = {'elf_sha256':'elf-hash','segments':segments}
        result = {'status':'OK','ddr_readback_matches':True,'completion_marker_seen':True,
                  'uart_bytes':len(uart),'sha256':segments[0]['sha256'],
                  'segment_readbacks':{s['file']:True for s in segments}}
        manifest = {'plan_sha256':archive.digest(prep/'ddr-load.plan'),
                    'segments':[{k:s[k] for k in ('file','size','sha256')} for s in segments]}
        dump(run/'result.json',result);dump(run/'run-manifest.json',manifest)
        dump(prep/'deployment.json',deployment)
        return run,prep,uart,result,manifest

    def test_exact_manifest_completion_and_optional_local_readbacks(self):
        with tempfile.TemporaryDirectory() as d:
            run,prep,uart,result,manifest=self.setup_run(d)
            validated=archive.verify_run_identity(run,prep,'elf-hash',uart)
            self.assertTrue(all(x['worker_readback_verified'] for x in validated[3]))
            self.assertFalse(any(x['downloaded_readback_rehashed'] for x in validated[3]))
            (run/'image.bin.readback').write_bytes((prep/'image.bin').read_bytes())
            self.assertTrue(archive.verify_run_identity(run,prep,'elf-hash',uart)[3][0]['downloaded_readback_rehashed'])
            (run/'image.bin.readback').write_bytes(b'X'*64)
            with self.assertRaisesRegex(ValueError,'downloaded DDR'):
                archive.verify_run_identity(run,prep,'elf-hash',uart)

    def test_rejects_missing_segment_wrong_hash_false_success_and_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            run,prep,uart,result,manifest=self.setup_run(d)
            mutants=[]
            r=copy.deepcopy(result);del r['segment_readbacks']['weights-w8a8.bin'];mutants.append(r)
            r=copy.deepcopy(result);r['status']='ERROR';mutants.append(r)
            r=copy.deepcopy(result);r['uart_bytes']+=1;mutants.append(r)
            r=copy.deepcopy(result);r['sha256']='wrong';mutants.append(r)
            for r in mutants:
                dump(run/'result.json',r)
                with self.assertRaises(ValueError): archive.verify_run_identity(run,prep,'elf-hash',uart)
            dump(run/'result.json',result)
            (prep/'weights-w8a8.bin').write_bytes(b'X'*64)
            with self.assertRaisesRegex(ValueError,'segment bytes'):
                archive.verify_run_identity(run,prep,'elf-hash',uart)
            with self.assertRaises(ValueError): archive.safe_file(prep,'../escape')
            (prep/'linked').symlink_to(prep/'image.bin')
            with self.assertRaises(ValueError): archive.safe_file(prep,'linked')

    def test_oracle_must_reconstruct_from_npz_and_match_full_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);npz=d/'a.npz';blob=d/'oracle.bin'
            logits=np.zeros((1,1,151936),dtype=np.float32);logits[0,0,42]=1
            kv=np.arange(8*3*128,dtype=np.float32).reshape(1,3,8,128)
            np.savez(npz,prefill_logits=logits,decode_logits_0=logits,kv_key_used=kv,kv_value_used=kv)
            blob.write_bytes(b''.join(x.astype('<f4').tobytes() for x in
                (logits,logits,kv.transpose(0,2,1,3),kv.transpose(0,2,1,3))))
            m={'schema_version':1,'layers':1,'prefill_len':2,'decode_steps':1,'vocab_size':151936,
               'arithmetic_profile':'nr-fpga','trajectory_verified':True,'capacity':512,'prompt_ids':[1,2],
               'source_sha256':archive.digest(npz),'blob_sha256':archive.digest(blob),'bytes':blob.stat().st_size}
            meta={'layers':1,'capacity':512,'arithmetic_profile':'nr-fpga','prompt_ids':[1,2],
                  'prefill_argmax_last':42,'decode_steps_recorded':[dict(step=0,cache_position=2,input_token=42,generated_token=42)]}
            self.assertEqual(archive.verify_reference(m,meta,npz,blob,layers=1,prefill=2,steps=1),[42,42])
            with self.assertRaisesRegex(ValueError,'layers'):
                archive.verify_reference(m,meta,npz,blob,layers=4,prefill=2,steps=1)
            bad=copy.deepcopy(meta);bad['decode_steps_recorded'][0]['input_token']=43
            with self.assertRaisesRegex(ValueError,'trajectory'):
                archive.verify_reference(m,bad,npz,blob,layers=1,prefill=2,steps=1)
            blob.write_bytes(b'\0'*blob.stat().st_size);m['blob_sha256']=archive.digest(blob)
            with self.assertRaisesRegex(ValueError,'reconstruct'):
                archive.verify_reference(m,meta,npz,blob,layers=1,prefill=2,steps=1)

    def test_elf_uses_actual_embedded_bytes_not_manifest_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);elf=d/'a.elf';blob=d/'blob.bin';blob.write_bytes(b'ORACLE')
            data=bytearray(256);data[:6]=b'\x7fELF\x02\x01'
            struct.pack_into('<H',data,18,243);struct.pack_into('<Q',data,32,64)
            struct.pack_into('<HH',data,54,56,1)
            struct.pack_into('<IIQQQQQQ',data,64,1,4,128,0x80000000,0x80000000,64,64,64)
            data[132:138]=b'ORACLE';elf.write_bytes(data)
            self.assertEqual(archive.elf_blob_matches(elf,0x80000004,blob),132)
            data[132]=0;elf.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'embedded'):
                archive.elf_blob_matches(elf,0x80000004,blob)
            with self.assertRaisesRegex(ValueError,'file-backed'):
                archive.elf_blob_matches(elf,0x90000000,blob)

    def test_refuses_archive_overwrite_before_other_work(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);(d/'old.txt').write_text('preserve')
            with self.assertRaisesRegex(ValueError,'overwrite'):
                archive.archive(SimpleNamespace(output=d))
            self.assertEqual((d/'old.txt').read_text(),'preserve')

    def test_intermediate_reference_reconstructs_every_layout_and_resolves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_intermediates(directory)
            self.assertEqual(self.verify_intermediates(f)['layers'], 1)
            arrays, metadata, sources = archive.intermediate_sources(f.manifest)
            self.assertEqual((arrays, metadata), (f.arrays, f.metadata))
            self.assertEqual(set(sources), {f.arrays, f.metadata})
            layout = Path(directory)/'weight-layout.json'
            layout.write_text('{}')
            f.manifest['source_sha256'][str(layout)] = archive.digest(layout)
            archive.intermediate_sources(f.manifest)
            layout.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                archive.intermediate_sources(f.manifest)

    def test_intermediate_generated_source_npz_and_metadata_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_intermediates(directory)
            original = f.source.read_bytes()
            f.source.write_bytes(original + b'changed')
            with self.assertRaisesRegex(ValueError, 'generated source hash'):
                self.verify_intermediates(f)
            f.source.write_bytes(original)
            for path in (f.arrays, f.metadata):
                expected = f.manifest['source_sha256'][str(path)]
                f.manifest['source_sha256'][str(path)] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'input hash'):
                    self.verify_intermediates(f)
                f.manifest['source_sha256'][str(path)] = expected
            metadata = archive.read(f.metadata)
            metadata['prompt_ids'] = [1]
            dump(f.metadata, metadata)
            f.manifest['source_sha256'][str(f.metadata)] = archive.digest(f.metadata)
            with self.assertRaisesRegex(ValueError, 'metadata configuration'):
                self.verify_intermediates(f)

    def test_intermediate_rehashed_blob_still_must_match_independent_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_intermediates(directory)
            original = f.blob.read_bytes()
            f.blob.write_bytes(b'\0' * len(original))
            with self.assertRaisesRegex(ValueError, 'blob hash/size'):
                self.verify_intermediates(f)
            f.manifest['reference_blob_sha256'] = archive.digest(f.blob)
            with self.assertRaisesRegex(ValueError, 'reconstruct'):
                self.verify_intermediates(f)

    def test_intermediate_npz_rejects_wrong_dtype_shape_missing_nonfinite_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_intermediates(directory)
            key = f.manifest['entries'][0]['reference_keys'][0]
            float_key = f.manifest['entries'][1]['reference_keys'][0]
            cases = []
            bad = dict(f.data); bad[key] = bad[key].astype(np.float32); cases.append((bad, 'dtype'))
            bad = dict(f.data); bad[key] = bad[key].reshape(-1); cases.append((bad, 'shape'))
            bad = dict(f.data); del bad[key]; cases.append((bad, 'missing intermediate'))
            bad = dict(f.data); bad[float_key] = np.full_like(bad[float_key], np.nan); cases.append((bad, 'nonfinite'))
            bad = dict(f.data); bad['decode_0_model.layers.0_another_hidden'] = np.zeros(1); cases.append((bad, 'uncovered'))
            # Same flattened values and byte count do not establish a valid head view.
            projection_key = f.manifest['entries'][4]['reference_keys'][0]
            bad = dict(f.data); bad[projection_key] = bad[projection_key].reshape(1, 12); cases.append((bad, 'projection'))
            for data, message in cases:
                with self.subTest(message=message):
                    np.savez(f.arrays, **data)
                    f.manifest['source_sha256'][str(f.arrays)] = archive.digest(f.arrays)
                    with self.assertRaisesRegex(ValueError, message):
                        self.verify_intermediates(f)

    def test_intermediate_offsets_cannot_reorder_or_alias_the_reference_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_intermediates(directory)
            f.manifest['entries'][1]['offsets'][0] = 0
            with self.assertRaisesRegex(ValueError, 'offsets'):
                self.verify_intermediates(f)

    def test_optional_intermediate_manifest_is_bound_to_plan_and_uart(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory)
            self.assertIsNone(archive.intermediate_manifest(image, {}, b'[nr] RA returned: PASS'))
            for uart in (b'[intermediate] complete position=00000000 PASS', b'[intermediate broken',
                         b'intermediate] entry=00000000'):
                with self.assertRaisesRegex(ValueError, 'absent'):
                    archive.intermediate_manifest(image, {}, uart)
            with self.assertRaisesRegex(ValueError, 'missing intermediate'):
                archive.intermediate_manifest(image, {'intermediate_probe': {'claimed': True}}, b'')
            f = self.setup_intermediates(directory)
            dump(image/'intermediate-probe.json', f.manifest)
            self.assertEqual(archive.intermediate_manifest(image, {}, b''), f.manifest)
            self.assertEqual(archive.intermediate_manifest(image, {'intermediate_probe': f.manifest}, b''), f.manifest)
            with self.assertRaisesRegex(ValueError, 'plan intermediate'):
                archive.intermediate_manifest(image, {'intermediate_probe': None}, b'')


if __name__=='__main__':unittest.main()
