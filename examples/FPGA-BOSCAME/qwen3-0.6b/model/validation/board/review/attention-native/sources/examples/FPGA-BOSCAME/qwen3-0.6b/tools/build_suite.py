#!/usr/bin/env python3
"""Link selected, independently generated operator objects into one NR image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]

def run(cmd, **kw):
    print('+ '+shlex.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),check=True,**kw)

def configuration():
    """The enclosing case Makefile exports the resolved, authoritative values."""
    values={key[len('QWEN_CFG_'):]: value
            for key,value in os.environ.items() if key.startswith('QWEN_CFG_')}
    if not values:
        raise SystemExit('build configuration must be requested through make print-config')
    tool_names=('BUDDY_OPT','BUDDY_TRANSLATE','LLC','HOST_CC','RISCV_CC','RISCV_LD','RISCV_OBJCOPY','RISCV_OBJDUMP','PYTHON')
    identities={}
    for name in tool_names:
        command=shlex.split(values.get(name,''))
        resolved=shutil.which(command[0]) if command else None
        entry={'command':command,'executable':str(Path(resolved).resolve()) if resolved else None}
        if resolved:
            stat=Path(resolved).stat()
            entry.update(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        identities[name]=entry
    return {'directory':str(Path.cwd()),'variables':values,'tools':identities,
            'configuration_helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def configuration_helper():
    if len(sys.argv)<2:
        return False
    if sys.argv[1]=='--print-config':
        print(json.dumps(configuration(),sort_keys=True))
        return True
    if sys.argv[1]=='--config-stamp':
        if len(sys.argv)!=3:
            raise SystemExit('--config-stamp requires a filename')
        path=Path(sys.argv[2])
        contents=json.dumps(configuration(),sort_keys=True,indent=2)+'\n'
        if not path.exists() or path.read_text()!=contents:
            temporary=path.with_name(path.name+'.tmp')
            temporary.write_text(contents)
            temporary.replace(path)
        return True
    if sys.argv[1]=='--audit-lowering':
        if len(sys.argv)!=4:
            raise SystemExit('--audit-lowering requires lowered MLIR and metadata files')
        lowered=Path(sys.argv[2]).read_text()
        metadata=json.loads(Path(sys.argv[3]).read_text())
        if not re.search(r'bosc_ame\.target\s*=\s*"nr-fpga"',lowered):
            raise SystemExit('NR build rejected: buddy-opt must support and select target=nr-fpga; GEM5 and legacy FPGA artifacts cannot run in this runtime')
        if metadata.get('kind')=='matmul_i8':
            if 'bosc_ame.mqma.b.mm' not in lowered or re.search(r'\blinalg\.matmul',lowered):
                raise SystemExit('NR AME build rejected: integer matmul did not lower to bosc_ame.mqma.b.mm (CPU fallback is not an AME validation)')
        return True
    return False

def main():
    if configuration_helper():
        return
    p=argparse.ArgumentParser()
    p.add_argument('--group',choices=['all','ame','aux','rvv','scalar','smoke'],default='all')
    p.add_argument('--case',action='append',dest='cases')
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--host',action='store_true')
    args=p.parse_args()
    if args.jobs<1:
        p.error('--jobs must be positive')
    cases=[]
    for source in sorted(ROOT.glob('*/kernel.mlir')):
        name=source.parent.name
        meta=json.loads(source.with_name('metadata.json').read_text())
        kind=meta.get('kind','')
        selected=(args.group=='all' or args.group=='ame' and kind=='matmul_i8'
                  or args.group in ('rvv','scalar') and kind=='matmul_f32'
                  or args.group=='aux' and kind not in ('matmul_i8','matmul_f32')
                  or args.group=='smoke' and name in ('matmul_3x19x70','matmul_1x1024x1024','rmsnorm_1x1024','softmax_16x1x17'))
        if args.cases is not None:selected=name in args.cases
        if selected:cases.append(name)
    if not cases: p.error('no matching operator cases')
    if args.cases and set(args.cases)-set(cases):p.error('unknown --case')
    out=ROOT/'build'/('suite-'+args.group+('-host' if args.host else ''))
    out.mkdir(parents=True,exist_ok=True)
    # Keep the build log and each stage in the original sample directory.
    makevars=shlex.split(os.environ.get('QWEN_MAKE_FLAGS',''))
    case_directory=ROOT/cases[0]
    config=json.loads(subprocess.check_output(['make','--no-print-directory','-s','-C',str(case_directory),'print-config',*makevars],text=True))
    variables=config['variables']
    build_dir=Path(variables['BUILD'])
    target=str(build_dir/'host-check') if args.host else 'all'
    from concurrent.futures import ThreadPoolExecutor
    def build(name):run(['make','-C',ROOT/name,target,*makevars])
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(build,cases))
    cc=shlex.split(variables['HOST_CC' if args.host else 'RISCV_CC'])
    flags=shlex.split(variables['HOST_CFLAGS' if args.host else 'CFLAGS'])
    declarations=[];calls=[];objects=[]
    for i,name in enumerate(cases):
        symbol='launch_'+name
        declarations.append('int '+symbol+'(void);')
        calls.append(f'  nr_puts("[{i+1}/{len(cases)}] {name} BEGIN\\r\\n"); failures += {symbol}() != 0;')
        obj=out/(name+'.o')
        run([*cc,*flags,'-Dlaunch='+symbol,'-c',ROOT/name/'launch.c','-o',obj],cwd=case_directory)
        objects.append(obj)
        objects.append(ROOT/name/build_dir/('kernel.host.ll' if args.host else 'kernel.o'))
    suite=out/'suite.c'
    suite.write_text('#include "support.h"\n'+'\n'.join(declarations)+'\nint launch(void) {\n  unsigned failures=0;\n'+'\n'.join(calls)+'''\n  nr_puts("suite failed cases: "); nr_hex32(failures); nr_puts("\\r\\n");
  return print_check("qwen3 operator suite",failures,0.0f);
}
''')
    if args.host:
        image=out/'suite'
        run([*cc,*flags,suite,ROOT/'support.c',ROOT/'tools/host_main.c',*objects,'-lm','-o',image],cwd=case_directory)
    else:
        obj=out/'suite.o'
        run([*cc,*flags,'-c',suite,'-o',obj],cwd=case_directory);objects.append(obj)
        objects += [case_directory/path for path in shlex.split(variables['RUNTIME_OBJS'])]
        ld=shlex.split(variables['RISCV_LD'])
        elf=out/'suite.elf';image=out/'suite.bin'
        run([*ld,'--gc-sections','-T',str(Path(variables['NR'])/'nr.ld'),'-Map='+str(out/'suite.map'),'-o',elf,*objects],cwd=case_directory)
        run([*shlex.split(variables['PYTHON']), ROOT.parent/'tools/check_nr_elf.py', elf,
             '--objdump', variables['RISCV_OBJDUMP'], '--output', out/'elf-audit.json'],cwd=case_directory)
        run([*shlex.split(variables['RISCV_OBJCOPY']),'-O','binary',elf,image],cwd=case_directory)
    source_files={ROOT/'common.mk',ROOT/'support.h',ROOT/'support.c',Path(__file__).resolve(),ROOT/'tools/host_main.c',ROOT/'tools/vectorize_nr.py',ROOT.parent/'common/toolchain.mk'}
    for name in cases:
        source_files.update(ROOT/name/filename for filename in ('kernel.mlir','launch.c','metadata.json','makefile'))
    if not args.host:
        source_files.add(ROOT.parent/'common/uart/uart.h')
        for pattern in ('*.c','*.h','*.S','*.ld'):
            source_files.update((ROOT.parent/'common/nr').glob(pattern))
        source_files.update(ROOT.parent/'tools'/filename for filename in ('ame_to_word.py','restrict_fpga_assembly.py','nr_isa.py','check_nr_elf.py'))
    record={'group':args.group,'host':args.host,'case_count':len(cases),'cases':cases,'image':str(image.relative_to(ROOT)),
            'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
            'build_configuration':config,
            'source_files':{str(path.relative_to(REPO)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_files)},
            'sources':{n:hashlib.sha256((ROOT/n/'kernel.mlir').read_bytes()).hexdigest() for n in cases}}
    (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(image)
if __name__=='__main__':main()
