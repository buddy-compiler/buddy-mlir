// RUN: mlir-opt %s -gpu-kernel-outlining \
// RUN: | mlir-opt -pass-pipeline="builtin.module(nvvm-attach-target{chip=sm_70 O=3},\
// RUN:     gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" \
// RUN: | mlir-cpu-runner -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} 
//        -shared-libs=${MLIR_CUDA_RUNTIME}

// 
// The gpu.printf dialect is a dialect for printing values from the device to the standard output, 
// as in CUDA or OpenCL, for debugging. 
// It provides an operation that takes a literal format string 
// and an arbitrary number of scalar arguments that should be printed. 
// Its purpose is to enable device-side printf, without relying on specific drivers or libraries.
// 

// 
// The gpu.printf dialect has only one operation:
//  * gpu.printf: Print a formatted string to the standard output. 
//    It requires specifying the format string, the arguments, and their types. 
//    It can also take an optional async token and a list of async dependencies, 
//    which indicate that the print should be performed asynchronously
// 
module attributes {gpu.container_module} {
    // 
    // The GPU module defines a kernel function @hello, which does the following:
    //  * Get the x coordinate of the GPU thread, and store it in %0.
    //  * Define two constants, %csti8 as 2, and %cstf32 as 3.0.
    //  * Call the gpu.printf function, print the thread’s x coordinate, and the values of the two constants.
    //  * Call the gpu.return function, end the execution of the kernel function.
    // 
    gpu.module @kernels {
        gpu.func @hello() kernel {
            %0 = gpu.thread_id x
            %csti8 = arith.constant 2 : i8
            %cstf32 = arith.constant 3.0 : f32
            // CHECK: Hello from 0, 2, 3.000000
            // CHECK: Hello from 1, 2, 3.000000
            gpu.printf "Hello from %lld, %d, %f\n" %0, %csti8, %cstf32  : index, i8, f32
            gpu.return
        }
    }
    // 
    // The function @main does the following:
    //  * Define two constants, %c2 as 2, and %c1 as 1.
    //  * Call the gpu.launch_func function, launch the GPU kernel function @kernels::@hello, 
    //    specify the number of thread blocks and threads.
    //  * Call the return function, end the execution of the function.
    // 
    func.func @main() {
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        gpu.launch_func @kernels::@hello
            blocks in (%c1, %c1, %c1)
            threads in (%c2, %c1, %c1)
        return
    }
}