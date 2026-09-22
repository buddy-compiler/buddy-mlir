#!/usr/bin/env python3
"""Compare UART samples with compiled-host and independent quantized references.

Sampled UART values and optional firmware full-logits/valid-KV summaries have
separate validation scopes. Every profile requires all generation steps.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import numpy as np

VOCAB = 151936
SELECTED_IDS = [49000,374,264,3146,7407,304,4787,5159,11]


def runtime_passed(log):
    return (len(re.findall(r'^\[nr\] RA returned: PASS\r?$',log,re.M)) == 1
            and not re.search(r'\[nr\][^\r\n]*(?:FAIL|TRAP)|TRAP mcause=|'
                              r'verify[^\r\n]*:\s*FAIL',log))


def f32(bits):
    return struct.unpack('<f',struct.pack('<I',int(bits,16)))[0]


def errors(actual,expected):
    delta=np.abs(np.asarray(actual,dtype=np.float64)-np.asarray(expected,dtype=np.float64))
    finite=bool(np.isfinite(delta).all())
    return {'samples':len(delta),'max_abs_error':float(delta.max(initial=0)) if finite else None,
            'mean_abs_error':float(delta.mean()) if len(delta) and finite else None,
            'all_finite':finite}


def check(log, reference, *, prefill=16, steps=8, layers=1):
    stages={}
    sequence=[]
    for kind,pos,tok,bits,cycles in re.findall(
            r'\[model\] (prefill|decode) position=([0-9A-Fa-f]+) token=([0-9A-Fa-f]+) '
            r'logit_bits=([0-9A-Fa-f]+) compute_cycles=([0-9A-Fa-f]+)',log):
        position=int(pos,16)
        sequence.append((kind,position))
        key='prefill' if kind=='prefill' else f'decode_{position-prefill}'
        # The firmware stage row names the graph invocation's START position;
        # tensor diagnostics name the LAST position produced by that call.
        stages[key]={'position':position,'position_semantics':'graph_start',
                     'last_output_position':position+prefill-1 if kind=='prefill' else position,
                     'token':int(tok,16),'logit':f32(bits),
                     'compute_cycles':int(cycles,16)}
    expected_keys=['prefill']+[f'decode_{i}' for i in range(steps)]
    sampled=[]
    for position,values in re.findall(r'\[model\] selected position=([0-9A-Fa-f]+)([^\r\n]*)',log):
        pos=int(position,16)
        key='prefill_logits' if pos==prefill-1 else f'decode_logits_{pos-prefill}'
        if key not in reference: raise ValueError(f'reference missing {key}')
        scores=reference[key].reshape(-1,151936)[-1]
        pairs=re.findall(r'([0-9A-Fa-f]{8}):([0-9A-Fa-f]{8})',values)
        ids=[int(t,16) for t,_ in pairs]
        if any(t < 0 or t >= VOCAB for t in ids):
            metric={'samples':len(ids),'all_finite':False,'max_abs_error':None,'mean_abs_error':None}
        else:
            metric=errors([f32(v) for _,v in pairs],scores[ids])
        sampled.append({'position':pos,'token_ids':ids,**metric})
    kv=[]
    for pos,layer,k,v in re.findall(r'\[model\] kv position=([0-9A-Fa-f]+) layer=([0-9A-Fa-f]+) k=([0-9A-Fa-f]+) v=([0-9A-Fa-f]+)',log):
        pos,layer=int(pos,16),int(layer,16)
        def value(name):
            a=reference[name]
            if a.shape[1]==8: return float(a[layer,0,pos,0])
            if a.shape[2]==8: return float(a[layer,pos,0,0])
            raise ValueError(f'unrecognized {name} layout {a.shape}')
        kv.append({'position':pos,'layer':layer,
                   **errors([f32(k),f32(v)],[value('kv_key_used'),value('kv_value_used')])})
    for key,row in stages.items():
        array='prefill_logits' if key=='prefill' else 'decode_logits_'+key.split('_')[1]
        if array not in reference or not 0 <= row['token'] < VOCAB:
            row.update(reference_token=None,token_matches=False,
                       selected_max_logit_abs_error=None)
            continue
        scores=reference[array].reshape(-1,151936)[-1]
        row.update(reference_token=int(np.argmax(scores)),
                   token_matches=row['token']==int(np.argmax(scores)),
                   selected_max_logit_abs_error=abs(row['logit']-float(scores[row['token']]))
                   if math.isfinite(row['logit']) else None)
        if not math.isfinite(row['logit']): row['logit']=None
    positions=list(range(prefill-1,prefill+steps))
    complete=(sequence==[('prefill',0)]+[('decode',p) for p in range(prefill,prefill+steps)]
              and sorted(stages)==sorted(expected_keys)
              and [s['position'] for s in sampled]==positions
              and all(s['token_ids']==SELECTED_IDS for s in sampled)
              and [(s['position'],s['layer']) for s in kv]==
                  [(p,l) for p in positions for l in range(layers)]
              and runtime_passed(log))
    return {'scope':'argmax, nine logits, and head0/dim0 of K/V at written positions',
            'trace_complete':complete,'stages':stages,'selected_logits':sampled,
            'kv_samples_head0_dim0':kv,
            'sampled_numeric_pass_at_1e_3':complete and all(r['token_matches'] for r in stages.values())
                and all(r['selected_max_logit_abs_error'] is not None and
                        r['selected_max_logit_abs_error']<=1e-3 for r in stages.values())
                and all(r['all_finite'] and r['max_abs_error']<=1e-3 for r in sampled+kv)}


def check_full_reference(log, manifest, *, prefill=16, steps=8, layers=1):
    max_atol, mean_atol = manifest['max_abs_tolerance'], manifest['mean_abs_tolerance']
    if any(not math.isfinite(v) or v < 0 for v in (max_atol,mean_atol)):
        raise ValueError('numeric reference tolerances must be finite and nonnegative')
    dimensions={'layers':layers,'prefill_len':prefill,'decode_steps':steps,'vocab_size':VOCAB}
    if 'schema_version' in manifest and manifest['schema_version'] != 1:
        raise ValueError('unsupported numeric reference schema')
    for field,value in dimensions.items():
        if field in manifest and manifest[field] != value:
            raise ValueError('embedded oracle dimension mismatch: '+field)
    checks=[]
    for name,pos,count,maxbits,meanbits,verdict in re.findall(
            r'^\[compare\] (logits|key_cache|value_cache) position=([0-9A-Fa-f]+) '
            r'count=([0-9A-Fa-f]+) max_abs_bits=([0-9A-Fa-f]{8}) '
            r'mean_abs_bits=([0-9A-Fa-f]{8}) (PASS|FAIL)\r?$',log,re.M):
        maximum,mean=f32(maxbits),f32(meanbits)
        checks.append(dict(tensor=name,position=int(pos,16),count=int(count,16),
                           max_abs_error=maximum if math.isfinite(maximum) else None,
                           mean_abs_error=mean if math.isfinite(mean) else None,
                           all_finite=math.isfinite(maximum) and math.isfinite(mean),
                           firmware_verdict=verdict))
    expected=[(p,n,VOCAB if n=='logits' else layers*8*(p+1)*128)
              for p in range(prefill-1,prefill+steps)
              for n in ('logits','key_cache','value_cache')]
    actual=[(r['position'],r['tensor'],r['count']) for r in checks]
    passed=(actual==expected and runtime_passed(log) and all(
        r['firmware_verdict']=='PASS' and r['all_finite'] and
        0 <= r['max_abs_error'] <= max_atol and 0 <= r['mean_abs_error'] <= mean_atol
        for r in checks))
    return {'reference_manifest':manifest,'checks':checks,'pass':passed,
            'metadata_verified':manifest.get('schema_version')==1 and
                                all(field in manifest for field in dimensions),
            'scope':'firmware compares every last-position logit and every valid K/V element; no hidden-state capture'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uart',type=Path,required=True)
    p.add_argument('--host-graph',type=Path,required=True)
    p.add_argument('--quant-reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--embedded-reference', type=Path,
                   help='numeric-reference.json from the exact executed ELF')
    p.add_argument('--layers',type=int,default=1)
    p.add_argument('--prefill',type=int,default=16)
    p.add_argument('--steps',type=int,default=8)
    a=p.parse_args()
    log=a.uart.read_text(errors='replace')
    report={'execution':'FPGA, compared after execution on host',
            'scope':{'layers':a.layers,'prefill':a.prefill,'decode_steps':a.steps},
            'limits':['UART contains nine selected logits plus argmax, not full logits',
                      'KV samples are head 0 dimension 0 at written positions, not all KV elements',
                      'Full 28-layer and UART interactive acceptance are separate requirements'],
            'inputs':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in (a.uart,a.host_graph,a.quant_reference)}}
    for name,path in [('fpga_vs_compiled_host',a.host_graph),
                      ('fpga_vs_independent_quantized_reference',a.quant_reference)]:
        with np.load(path) as ref:
            report[name]=check(log,ref,prefill=a.prefill,steps=a.steps,layers=a.layers)
    if a.embedded_reference:
        manifest=json.loads(a.embedded_reference.read_text())
        if manifest['source_sha256'] != hashlib.sha256(a.quant_reference.read_bytes()).hexdigest():
            raise ValueError('embedded oracle differs from the independent reference being reported')
        full=check_full_reference(log,manifest,prefill=a.prefill,steps=a.steps,layers=a.layers)
        sampled=report['fpga_vs_independent_quantized_reference']
        full['generation_trace_complete']=sampled['trace_complete']
        full['generated_tokens_match_reference']=all(r['token_matches'] for r in sampled['stages'].values())
        full['sampled_values_consistent_with_full_bound']=all(
            row['selected_max_logit_abs_error'] is not None and
            row['selected_max_logit_abs_error']<=manifest['max_abs_tolerance']
            for row in sampled['stages'].values()) and all(
                row['all_finite'] and row['max_abs_error']<=manifest['max_abs_tolerance']
                for row in sampled['selected_logits']+sampled['kv_samples_head0_dim0'])
        full['pass'] &= (full['generation_trace_complete'] and full['generated_tokens_match_reference']
                         and full['sampled_values_consistent_with_full_bound'])
        report['full_tensor_fpga_vs_quantized_reference']=full
        report['inputs'][str(a.embedded_reference)]=hashlib.sha256(a.embedded_reference.read_bytes()).hexdigest()
        report['limits']=['Full logits/valid-KV errors are computed on FPGA; UART carries summaries',
                          'Compiled-host comparison remains sampled; hidden states are not captured',
                          'Full 28-layer and UART interactive acceptance are separate requirements']
    report['status']='SAMPLED_PASS' if all(report[n]['sampled_numeric_pass_at_1e_3'] for n in
       ('fpga_vs_compiled_host','fpga_vs_independent_quantized_reference')) else 'NOT_ACCEPTED'
    if a.embedded_reference:
        report['status']='FULL_LOGITS_KV_PASS' if full['pass'] else 'NOT_ACCEPTED'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'output':str(a.output)}))
    return 0 if report['status'] in ('SAMPLED_PASS','FULL_LOGITS_KV_PASS') else 1

if __name__=='__main__': raise SystemExit(main())
