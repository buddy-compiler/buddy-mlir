import mlir.ir as ir
from mlir.passmanager import *
import os


class Graph:
    def __init__(self, fx_graph, func_params, func_inputs) -> None:
        self.body = fx_graph
        self.params = func_params
        self.inputs = func_inputs
        self.device = "cpu"
        self._imported_module = None
        # self._ops_registry = {}
        # self._ops_registry.update(tosa_ops_registry)
        # self._ops_registry.update(math_ops_registry)
        # self._ops_registry.update(linalg_ops_registry)

    # def lower_to_top_level_ir(self):
    #     ctx = ir.Context()
    #     with ir.Location.unknown(ctx):
    #         fx_importer = FXGraphImporter(
    #                     self.body,
    #                     self.params,
    #                     self.inputs,
    #                     False,
    #                     self.body.__name__,
    #                     self._ops_registry,
    #                 )
    #         self._imported_module = fx_importer.import_graph()

    def lower_to_llvm_ir(self):
        print(self._imported_module)
        pm = PassManager("builtin.module")
        pm.add("func.func(tosa-to-linalg-named)")
        pm.add("func.func(tosa-to-linalg)")
        pm.add("func.func(tosa-to-tensor)")
        pm.add("func.func(tosa-to-arith)")
        pm.add("empty-tensor-to-alloc-tensor")
        pm.add("convert-elementwise-to-linalg")
        pm.add("arith-bufferize")
        pm.add("func.func(linalg-bufferize)")
        pm.add("func.func(tensor-bufferize)")
        pm.add("func-bufferize")
        pm.run(self._imported_module.operation)
        print(self._imported_module)
        print(
            "-------------------------------------------------------------------"
        )
        print("Lowering the module to LLVM dialect ...")
        pm.add("func.func(buffer-deallocation)")
        pm.add("func.func(convert-linalg-to-loops)")
        pm.add("convert-scf-to-cf")
        pm.add("convert-arith-to-llvm")
        pm.add("expand-strided-metadata")
        pm.add("finalize-memref-to-llvm")
        pm.add("convert-arith-to-llvm")
        pm.add("convert-math-to-llvm")
        pm.add("convert-math-to-libm")
        pm.add("func.func(llvm-request-c-wrappers)")
        pm.add("convert-func-to-llvm")
        pm.add("reconcile-unrealized-casts")
        pm.run(self._imported_module.operation)

    def compile(self):
        # self.lower_to_top_level_ir()
        self.lower_to_llvm_ir()
        with open("./test.mlir", "w") as f:
            f.write(str(self._imported_module))
        os.system(
            "mlir-translate ./test.mlir -mlir-to-llvmir | llvm-as | \
              llc -filetype=obj -relocation-model=pic -O3 -o test.o"
        )
