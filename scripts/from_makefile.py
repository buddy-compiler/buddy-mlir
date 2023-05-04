import os
import glob
import re
import json
import yaml
import pathlib


def find_makefiles(path):
    makefiles = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
          if dir in [
              "MLIRLinalg",
              "MLIRTensor",
              "MLIRAffine",
              "MLIRMemRef",
              "MLIRSCF",
              "MLIRMath",
              "MLIRVector",
              "VectorExpDialect",
              "RVVExperiment",
              "RVVDialect",
          ]:
            sub = os.path.join(root, dir)
            for file in os.listdir(sub):
                # If there's no mlir file in the directory, skip it
                if any(file.endswith('.mlir') for file in os.listdir(sub)):
                    if file == 'makefile':
                        makefiles.append((sub, dir))
    return makefiles


def parse_makefile(makefile):
    commands = []
    with open(makefile, 'r') as f:
        lines = f.readlines()

    loc = 0
    beg = -1
    res = ''
    title = ''
    while loc < len(lines):
        line = lines[loc].strip()
        # replace all tabs
        line = re.sub(r'\t', ' ', line)
        if len(line) == 0:
            loc += 1
            if beg == -1:
                continue
            commands.append((title, res))
            title = ''
            res = ''
            beg = -1
            continue
        if line.endswith(':'):
            beg = loc
            title = line[:-1]
            loc += 1
            continue
        if beg == -1:
            loc += 1
            continue
        if line.startswith('@'):
            line = line[1:]
        if line[-1] == '\\':
            res += line[:-1]
        else:
            res += line
        loc += 1

    return commands


# Search in the current directory and its subdirectories
path = '/Users/zircon/Works/PLCT/buddy-mlir/examples'
dirs = find_makefiles(path)

for dir, sub in dirs:
    makefile = os.path.join(dir, 'makefile')
    commands = parse_makefile(makefile)
    res = []
    for command in commands:
        cmd = command[1].split('|')
        for i in range(len(cmd)):
            cmd[i] = cmd[i].strip()
        typ = command[0].split('-')[-1]
        res.append({"filename": command[0], "type": typ, "commands": cmd})
    pathlib.Path.mkdir(pathlib.Path('./scripts/examples'), exist_ok=True)
    yaml.dump(res, open(f'./scripts/examples/{sub}.yaml', 'w+'))
