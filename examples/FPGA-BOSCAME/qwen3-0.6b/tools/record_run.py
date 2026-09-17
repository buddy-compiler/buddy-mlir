#!/usr/bin/env python3
"""Validate complete suite UART coverage and retain compact, auditable evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

def main():
    p=argparse.ArgumentParser()
    p.add_argument('manifest',type=Path)
    p.add_argument('run_dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    result=json.loads((args.run_dir/'result.json').read_text())
    raw=(args.run_dir/'uart.raw.log').read_bytes()
    text=raw.decode('ascii',errors='replace')
    if result.get('status')!='OK' or not result.get('ddr_readback_matches'):
        p.error('runner did not complete with matching DDR readback')
    if result['sha256']!=manifest['sha256']:p.error('loaded image differs from suite manifest')
    if re.search(r'\bFAIL\b|TRAP mcause=',text):p.error('UART contains failure/trap')
    records=[]
    for name in manifest['cases']:
        matches=re.findall(r'^verify '+re.escape(name)+r': PASS errors=00000000 max_abs_error_f32_bits=([0-9a-fA-F]{8})\s*$',text,re.M)
        if len(matches)!=1:p.error('missing/duplicate success for '+name)
        record={'case':name,'status':'PASS','max_abs_error_f32_bits':matches[0]}
        cycles=re.findall(r'^cycles '+re.escape(name)+r': ([0-9a-fA-F]+)\s*$',text,re.M)
        if cycles:record['kernel_cycles']=int(cycles[0],16)
        records.append(record)
    if not re.search(r'verify qwen3 (?:Triton )?operator suite: PASS',text) or '[nr] RA returned: PASS' not in text:
        p.error('suite/runtime did not finish successfully')
    args.output.mkdir(parents=True,exist_ok=True)
    for name in ('uart.raw.log','result.json'):
        shutil.copyfile(args.run_dir/name,args.output/name)
    shutil.copyfile(args.manifest,args.output/'manifest.json')
    report={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'source_run':args.run_dir.name,
        'fpga':result['fpga'],'image_sha256':result['sha256'],'uart_sha256':hashlib.sha256(raw).hexdigest(),
        'case_count':len(records),'all_pass':True,'cases':records}
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(records)} cases PASS; {args.output}')
if __name__=='__main__':main()
