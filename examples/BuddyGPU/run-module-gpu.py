import mlir.ir as ir
import mlir.dialects.func as func
import mlir.dialects.memref as memref
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir import runtime as rt
from mlir.ir import *
import numpy as np
import ctypes
import ml_dtypes
import argparse as ap


def to_numpy(element_type: str) -> np.dtype:
    match element_type:
        case "f16":
            return np.float16
        case "f32":
            return np.float32
        case "f64":
            return np.float64
        case "i8":
            return np.int8
        case "i16":
            return np.int16
        case "i32":
            return np.int32
        case "i64":
            return np.int64
        case "bf16":
            return np.dtype("bfloat16")
        case _:
            raise ValueError(f"Unsupported type: {element_type}")


def to_mlir(dtype: np.dtype) -> ir.Type:
    match dtype:
        case np.float16:
            return ir.F16Type.get()
        case np.float32:
            return ir.F32Type.get()
        case np.float64:
            return ir.F64Type.get()
        case np.int8:
            return ir.IntegerType.get_signless(8)
        case np.int16:
            return ir.IntegerType.get_signless(16)
        case np.int32:
            return ir.IntegerType.get_signless(32)
        case np.int64:
            return ir.IntegerType.get_signless(64)
        case _:
            raise ValueError(f"Unsupported type: {dtype}")


def lower_to_llvm_cpu(module: Module) -> Module:
    pm = PassManager("builtin.module")
    pm.add("func.func(tosa-to-linalg-named)")
    pm.add("func.func(tosa-to-linalg)")
    pm.add("func.func(tosa-to-tensor)")
    pm.add("func.func(tosa-to-arith)")
    pm.add("arith-expand")
    pm.add("eliminate-empty-tensors")
    pm.add("empty-tensor-to-alloc-tensor")
    pm.add("convert-elementwise-to-linalg")
    pm.add("one-shot-bufferize")
    pm.add("func.func(convert-linalg-to-affine-loops)")
    pm.add("affine-loop-fusion")
    pm.add("func.func(affine-parallelize)")
    pm.add("lower-affine")
    pm.add("convert-scf-to-openmp")
    pm.add("func-bufferize")
    pm.add("arith-bufferize")
    pm.add("func.func(tensor-bufferize)")
    pm.add("func.func(buffer-deallocation)")
    pm.add("func.func(finalizing-bufferize)")
    pm.add("expand-strided-metadata")
    pm.add("convert-vector-to-llvm")
    pm.add("memref-expand")
    pm.add("arith-expand")
    pm.add("convert-arith-to-llvm")
    pm.add("finalize-memref-to-llvm")
    pm.add("convert-scf-to-cf")
    pm.add("func.func(llvm-request-c-wrappers)")
    pm.add("convert-openmp-to-llvm")
    pm.add("convert-math-to-llvm")
    pm.add("convert-math-to-libm")
    pm.add("convert-func-to-llvm")
    pm.add("reconcile-unrealized-casts")
    pm.run(module.operation)
    return module


def new_ranked_memref_descriptor(nparray: np.ndarray):
    if nparray.dtype == "bfloat16":
        ctp = rt.F16
    else:
        ctp = rt.as_ctype(nparray.dtype)

    if nparray.ndim == 0:
        x = rt.make_zero_d_memref_descriptor(ctp)()
        x.allocated = nparray.ctypes.data
        x.aligned = nparray.ctypes.data_as(ctypes.POINTER(ctp))
        x.offset = ctypes.c_longlong(0)
        return x

    x = rt.make_nd_memref_descriptor(nparray.ndim, ctp)()
    nbytes = nparray.nbytes
    buffer = ctypes.create_string_buffer(nbytes)
    ctypes.memmove(buffer, nparray.ctypes.data, nbytes)
    x.allocated = ctypes.cast(buffer, ctypes.c_void_p).value
    x.aligned = ctypes.cast(buffer, ctypes.POINTER(ctp))
    x.offset = ctypes.c_longlong(0)
    x.shape = nparray.ctypes.shape

    # Numpy uses byte quantities to express strides, MLIR OTOH uses the
    # torch abstraction which specifies strides in terms of elements.
    strides_ctype_t = ctypes.c_longlong * nparray.ndim
    x.strides = strides_ctype_t(
        *[x // nparray.itemsize for x in nparray.strides]
    )
    return x


def get_memref_descriptors(args: list[Type]):
    memref_ptrs = []
    for arg in args:
        elem_type = to_numpy(str(arg.element_type))
        np_arg = np.random.rand(*arg.shape).astype(elem_type)
        memref_ptrs.append(
            ctypes.pointer(ctypes.pointer(new_ranked_memref_descriptor(np_arg)))
        )
    return memref_ptrs


def test():
    with Context() as ctx:
        file = open("matmul.mlir", "r")
        module: Module = Module.parse(file.read())
        funcOp: func.FuncOp = (
            module.operation.regions[0].blocks[0].operations[0]
        )
        funcName = str(funcOp.name).replace('"', "")
        assert isinstance(funcOp, func.FuncOp)
        args_type: list[Type] = [arg.type for arg in funcOp.arguments]
        res_type = funcOp.type.results

        file = open("matmul-cubin.mlir", "r")
        # newModule = lower_to_llvm_cpu(module)
        newModule = Module.parse(file.read())
        memref_ptrs = get_memref_descriptors(res_type + args_type)

        engine = ExecutionEngine(
            newModule,
            shared_libs=[
                "/usr/lib/libomp.so",
                "/home/liam/IPRC/llvm-project/build/lib/libmlir_c_runner_utils.so",
                "/home/liam/IPRC/llvm-project/build/lib/libmlir_async_runtime.so",
                "/home/liam/IPRC/llvm-project/build/lib/libmlir_runner_utils.so",
                "/home/liam/IPRC/llvm-project/build/lib/libmlir_cuda_runtime.so",
            ],
            opt_level=3,
        )
        engine.invoke(funcName, *memref_ptrs)
        out = rt.ranked_memref_to_numpy(memref_ptrs[0][0])
        if str(res_type[0].element_type) == "bf16":
            print("Running on BF16 mode, skipping numpy comparison.")
        else:
            print(out)
            input1 = rt.ranked_memref_to_numpy(memref_ptrs[1][0])
            input2 = rt.ranked_memref_to_numpy(memref_ptrs[2][0])
            print("Skipping numpy comparison")
            numpy_out = np.matmul(input1, input2)
            print(numpy_out)
            print(
                f"MLIR equal to PyTorch? {np.allclose(out, numpy_out,rtol=1e-03, atol=1e-03)}"
            )


test()
