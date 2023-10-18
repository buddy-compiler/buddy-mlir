add_executable(llamaRun llama_run_params_pack.cc llama.o)
include_directories(.)
target_link_libraries(llamaRun mlir_c_runner_utils omp)
