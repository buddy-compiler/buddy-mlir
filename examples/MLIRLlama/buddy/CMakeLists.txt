add_executable(llamaRun llama_run.cc llama.o)
#link_directories(/home/wlq/buddy-mlir/examples/MLIRLlama/buddy)
target_link_libraries(llamaRun /home/wlq/buddy-mlir/llvm/build/lib/libmlir_c_runner_utils.so)