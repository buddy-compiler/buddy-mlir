# Trace framework design

The trace transform follows the same split used by graph quantization:

- `passes.py` exports the public pass entry points.
- `lowering.py` contains MLIR lowering hooks used when a traced Buddy graph
  node must attach attributes to an MLIR operation or value.

`TraceInsertionPass` runs before lowering. It matches names from
`TraceConfig.trace_config` against Buddy graph nodes, normalizes the trace
metadata, stores it on the matched node, and errors if any configured node is
missing.

The Buddy graph pass marks matched graph nodes and wraps the graph lowering
registry so trace attributes are attached after each marked node lowers to
MLIR. Nodes such as `PlaceholderOp` and `GetItemOp` are rejected because they
do not lower to their own MLIR operation.
