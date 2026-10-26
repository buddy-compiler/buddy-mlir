# RUN: %PYTHON %s | FileCheck %s

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from buddy.compiler.graph.transform.trace.passes import trace_insertion
from buddy.compiler.trace import TraceConfig, load_trace_config

with TemporaryDirectory() as directory:
    path = Path(directory) / "trace.toml"
    path.write_text("""[[trace.node]]
node = "conv"
id = 0
tag = "conv"
cycle = true
tensor = false

[[trace.node]]
node = "relu"
id = 1
tag = "relu"
cycle = false
tensor = true

[[trace.node]]
node = "disabled"
id = 2
tag = "disabled"
cycle = false
tensor = false
""")
    trace = TraceConfig(load_trace_config(path))
    nodes = [
        SimpleNamespace(name=name, trace_meta=None)
        for name in trace.trace_config
    ]
    trace_insertion(trace)(SimpleNamespace(body=nodes, _ops_registry={}))
    for node in nodes:
        print(node.name, node.trace_meta)
    # CHECK: conv {{.*}}'cycle': True, 'tensor': False
    # CHECK: relu {{.*}}'cycle': False, 'tensor': True
    # CHECK: disabled None

    path.write_text(path.read_text().replace("tensor = false\n", "", 1))
    try:
        load_trace_config(path)
    except ValueError as error:
        print(error)
    else:
        raise AssertionError("missing tensor choice accepted")
    # CHECK: trace tensor for conv must be a boolean

    trace.trace_config["unknown"] = {
        "id": 3,
        "tag": "unknown",
        "cycle": True,
        "tensor": False,
    }
    try:
        trace_insertion(trace)(SimpleNamespace(body=nodes, _ops_registry={}))
    except KeyError as error:
        print(error)
    else:
        raise AssertionError("unknown trace node accepted")
    # CHECK: trace nodes did not match buddy graph nodes: unknown
