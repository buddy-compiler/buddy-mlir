#!/usr/bin/env python3
"""Record actual linked NR placement separately from static malloc estimates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--elf',type=Path,required=True)
    p.add_argument('--graph-ir',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nm',type=Path,default=Path(__file__).resolve().parents[5]/'llvm/build-2d26/bin/llvm-nm')
    a=p.parse_args()
    raw=subprocess.check_output([str(a.nm),'--defined-only',str(a.elf)],text=True)
    symbols={s[2]:int(s[0],16) for line in raw.splitlines() if len(s:=line.split())==3}
    arenas={}
    for name,start in symbols.items():
        if name.endswith('_raw') and name[:-4]+'_end' in symbols:
            end=symbols[name[:-4]+'_end']
            arenas[name[:-4]]={'start':hex(start),'end':hex(end),'bytes':end-start}
    low_used=symbols['__heap_start']-0x80000000
    high_used=symbols['__workspace_end']-symbols['__workspace_start']
    report={'elf':str(a.elf),'elf_sha256':hashlib.sha256(a.elf.read_bytes()).hexdigest(),
      'arenas':arenas,
      'low':{'base':'0x80000000','limit':'0xb0000000','static_and_stacks_bytes':low_used,
             'remaining_scoped_heap_bytes':symbols['__heap_end']-symbols['__heap_start']},
      'high':{'base':'0xb8000000','limit':'0x100000000','persistent_bytes':high_used,
              'remaining_bytes':0x100000000-symbols['__workspace_end']},
      'static_allocation_site_estimates':[],
      'limitations':['malloc site sums are not dynamic peak proofs; loops and dynamic sizes need measured scratch_bytes',
                     'NR linker bounds are not physical DDR capacity measurements',
                     'see board/review/ddr-address-probe for sparse RA access and alias checks']}
    for path in a.graph_ir:
        text=path.read_text()
        operands=re.findall(r'call ptr @malloc\(i64 ([^,)]+)',text)
        fixed=[int(s) for s in operands if s.isdigit()]
        report['static_allocation_site_estimates'].append({
            'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'constant_sites':len(fixed),'constant_bytes_sum':sum(fixed),
            'dynamic_sites':len(operands)-len(fixed)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
