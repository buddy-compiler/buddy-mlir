add_executable(llamaRun llama_run_params_pack.cc llama_linalg.o TextContainer.h)
target_link_libraries(llamaRun mlir_c_runner_utils /buddy-mlir-for-transformer/llvm/build/runtimes/runtimes-bins/openmp/runtime/src/libomp.so)
