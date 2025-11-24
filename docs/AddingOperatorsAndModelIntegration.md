## Guide: Adding Operators and Integrating Models in Buddy-MLIR

This guide targets developers who want to add frontend operator mappings (PyTorch → Buddy Graph → MLIR) and import/integrate models in Buddy-MLIR. It is based on the repository directories `examples` and `frontend/Python`. After reading, you will be able to:

- Add a Buddy Graph-level operator and lower it to MLIR (TOSA/Linalg/Math, etc.).
- Import PyTorch functions/models via TorchDynamo or the Export API into Buddy Graph and MLIR modules, and obtain/package parameters.

### Relevant Directory Overview

- `frontend/Python/frontend.py`
  - Frontend entry `DynamoCompiler`: imports FX Graph → Buddy Graph, assembles registries, and wraps execution/compilation.
- `frontend/Python/graph/`
  - `operation.py`: Buddy Graph operator classes (op types, metadata, etc.).
  - `graph.py`: Graph structure, symbol table, and MLIR module generation driver.
  - `transform/`: Graph-level transforms (e.g., `maxpool2d_simplify`).
- `frontend/Python/ops/`
  - `tosa.py` / `linalg.py` / `math.py` / `func.py`: Lowering from Buddy Graph ops to MLIR dialect ops, registered via `ops_registry`.
  - `utils.py`: Helpers for types/attributes.
- `examples/BuddyPython/`
  - `module_gen.py`: Minimal import demo (function → MLIR).
  - `bert.py`: HuggingFace BERT import demo.
- `examples/MLIRPython/`
  - `README.md`: How to build MLIR Python bindings and a TorchDynamo custom backend demo.

---

## 1. Environment Setup (MLIR Python Bindings)

Buddy's Python frontend depends on MLIR Python bindings. Build highlights (see `examples/BuddyPython/README.md` and `examples/MLIRPython/README.md`):

1) Build MLIR Python bindings under `buddy-mlir/llvm` and verify with `ninja check-mlir`.
2) Add the build output to `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd)/tools/mlir/python_packages/mlir_core
```

Then verify in Python:

```python
from mlir.ir import Context, Module
```

---

## 2. Adding an Operator (Torch → Buddy Graph → MLIR)

Adding a new operator typically involves three layers:

1) Define the operator at the Buddy Graph layer (if a suitable class does not already exist).
2) Implement the lowering function to MLIR (TOSA/Linalg/Math, etc.) and register it into `ops_registry`.
3) Map the Torch Aten/Prims symbol to the Buddy Graph operator class in the frontend import map (`DynamoCompiler._ops_map`).

### Step 1: Define the Operator in Buddy Graph

File: `frontend/Python/graph/operation.py`

If an existing class can be reused, skip this step. Otherwise, add a new class and set the operator type (impacts fusion/scheduling):

```python
class MyNewOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType  # or ReduceType/ReshapeType/...
```

For conv/pool, you can also carry layout fields (see `Conv2dOp`, `MaxPool2dOp`).

### Step 2: Implement Lowering and Register to `ops_registry`

In the target dialect file (e.g., `frontend/Python/ops/tosa.py`), implement the conversion from the Buddy op to the MLIR op:

```python
def my_new_op(node: MyNewOp, symbol_table):
    # 1) Fetch inputs (or constants) from symbol_table
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    # 2) Read output shape/dtype
    output_shape = list(node.tensor_meta["shape"])
    mlir_dtype = mlir_element_type_get(node.tensor_meta["dtype"])  # utils.py
    # 3) Build MLIR type and op
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    op = tosa.SomeOp(tensor_type, input1, ...)
    return op

ops_registry = {
    # ...
    "MyNewOp": my_new_op,
}
```

Practical tips:
- Reuse helpers from `tosa.py` (e.g., `_gen_arith_binary_op`, `_normalize_binary_operator_args`).
- Handle broadcasting, dtype alignment, and necessary `tosa.CastOp`.
- For reshape/transpose, skip no-op transforms when shapes already match (see `reshape_op` optimization).

### Step 3: Map Torch Operator to Buddy Operator

File: `frontend/Python/frontend.py`, in `DynamoCompiler.__init__` under `_ops_map`:

```python
self._ops_map = {
    # examples:
    "add.Tensor": AddOp,
    "addmm.default": AddMMOp,
    # new:
    "aten_symbol_name": MyNewOp,
}
```

Notes: Keys are the FX/Aten symbols emitted by Torch (inspectable by running the import flow and printing `gm.graph.print_tabular()`); values are Buddy Graph operator classes.

After these three steps, validate the operator path with the sample script (see next section).

---

## 3. Minimal Import and Validation (Function → MLIR)

See `examples/BuddyPython/module_gen.py`:

```python
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

def foo(x, y):
    return x * y + x

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, torch.randn(10), torch.randn(10))
graph = graphs[0]
graph.lower_to_top_level_ir()  # generate high-level MLIR (TOSA/Linalg/…)
print(graph._imported_module)
```

Key points:
- `primary_registry` controls the preferred dialect/registry (e.g., TOSA). If not found, the importer falls back to other merged registries (`math`, `linalg`, `func`, etc.).
- Setting `aot_autograd_decomposition` pre-decomposes FX graphs into ATen/Prims for easier mappings.

---

## 4. Integrating PyTorch Models (with Parameters)

Two import paths are provided:

1) Dynamo path (default): `DynamoCompiler.importer(model, *args, **kwargs)`
2) Export path (preserves input order): `DynamoCompiler.importer_by_export(module, *args, **kwargs)`

Example: `examples/BuddyPython/bert.py`:

```python
from transformers import BertModel, BertTokenizer
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp

model = BertModel.from_pretrained("bert-base-uncased").eval()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer("Replace me by any text.", return_tensors="pt")
graphs = dynamo_compiler.importer(model, **encoded)

graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
graph.lower_to_top_level_ir(do_params_pack=True)
print(graph._imported_module)
print(params)
```

Notes:
- `imported_params` returns buffers/weights for the model; `do_params_pack=True` packs them during lowering if needed.
- To preserve the original module's input argument order, use `importer_by_export` (see comments/implementation in `frontend.py`).
- For complex models, check `examples/BuddyLlama/`, `examples/BuddyResNet18/`, etc.

### Execute the Graph (Optional)

`DynamoCompiler.dynamo_run()` returns a Python callable (based on MLIR ExecutionEngine) that can be invoked with tensors:

```python
runner = dynamo_compiler.dynamo_run()
out_tensors = runner(input_tensor_0, input_tensor_1, ...)
```

Note: Ensure shared libraries like `libmlir_runner_utils`, `libmlir_c_runner_utils`, and `libomp` are visible locally (paths composed in `frontend.py` for `llvm/build/lib`).

---

## 5. Debugging and FAQs

- How to find the Torch → Buddy mapping key?
  - In `DynamoCompiler._compile_fx`, enable `verbose=True` and inspect `gm.graph.print_tabular()` (`target`/`op`/`name`) to determine keys for `_ops_map`.
- Types/broadcasting issues:
  - Reuse the binary-op wrappers in `tosa.py` (alignment, broadcasting, reshape).
- Performance/correctness:
  - Skip redundant transforms when possible during lowering (e.g., `reshape_op` directly returns on equal shapes).

---

## 6. Pre-Submission Checklist

- For the new Buddy Graph operator class, is `OpType` set appropriately?
- For the corresponding lowering function, does it:
  - Fetch inputs correctly from `symbol_table` and handle dtype/shape?
  - Use appropriate MLIR dialect(s) and ops?
  - Register into the target dialect's `ops_registry`?
- Is `DynamoCompiler._ops_map` updated with the Torch symbol mapping?
- Do minimal/model examples run and print a reasonable MLIR module?

For more background and build instructions, see `docs/PythonEnvironment.md`, `examples/*/README.md`, and the implementations in `ops/*.py`.
