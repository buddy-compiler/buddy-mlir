# ===- run-module-gpu.py --------------------------------------------------===//
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ===----------------------------------------------------------------------===//
#
#  This file is a script to test whether the specified MLIR module on the GPU
#  calculates the same result as NumPy.
#
# ===----------------------------------------------------------------------===//

import mlir.dialects.func as func
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir import runtime as rt
from mlir.ir import *
import numpy as np
import ctypes
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


def test(source, target, llvm_dir):
    with Context() as ctx:
        file = open(source, "r")
        module: Module = Module.parse(file.read())
        funcOp: func.FuncOp = (
            module.operation.regions[0].blocks[0].operations[0]
        )
        funcName = str(funcOp.name).replace('"', "")
        assert isinstance(funcOp, func.FuncOp)
        args_type: list[Type] = [arg.type for arg in funcOp.arguments]
        res_type = funcOp.type.results

        file = open(target, "r")
        # newModule = lower_to_llvm_cpu(module)
        newModule = Module.parse(file.read())
        memref_ptrs = get_memref_descriptors(res_type + args_type)

        engine = ExecutionEngine(
            newModule,
            shared_libs=[
                llvm_dir + "/build/lib/libomp.so",
                llvm_dir + "/build/lib/libmlir_c_runner_utils.so",
                llvm_dir + "/build/lib/libmlir_async_runtime.so",
                llvm_dir + "/build/lib/libmlir_runner_utils.so",
                llvm_dir + "/build/lib/libmlir_cuda_runtime.so",
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
            numpy_out = np.matmul(input1, input2)
            print(numpy_out)
            print(
                f"MLIR equal to NumPy? {np.allclose(out, numpy_out,rtol=1e-03, atol=1e-03)}"
            )


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--llvm_dir", type=str, required=True)
    args = parser.parse_args()
    test(args.source, args.target, args.llvm_dir)
