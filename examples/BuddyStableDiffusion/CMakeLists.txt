add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/arg0_text_encoder.data
         ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/arg1_text_encoder.data
         ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/arg0_unet.data
         ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/arg0_vae.data
         ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/forward_text_encoder.mlir
         ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/subgraph0_text_encoder.mlir
         ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/forward_unet.mlir
         ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/subgraph0_unet.mlir
         ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/forward_vae.mlir
         ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/subgraph0_vae.mlir
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/import-stable-diffusion.py
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)


add_custom_command(
  OUTPUT forward_text_encoder.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/forward_text_encoder.mlir
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
            empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, \
            func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline
            "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), \
            eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
            convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, \
            convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, \
            convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O3 
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyStableDiffusion/forward_text_encoder.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/forward_text_encoder.mlir
  COMMENT "Building forward_text_encoder.o"
  VERBATIM)


add_custom_command(
  OUTPUT subgraph0_text_encoder.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/subgraph0_text_encoder.mlir 
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" |
          ${BUDDY_BINARY_DIR}/buddy-opt  
            -convert-elementwise-to-linalg
            -func-bufferize-dynamic-offset
            -arith-bufferize
            -func-bufferize
            -tensor-bufferize
            -linalg-bufferize
            -finalizing-bufferize
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3 
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyStableDiffusion/subgraph0_text_encoder.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/subgraph0_text_encoder.mlir
          buddy-opt
  COMMENT "Building subgraph0_text_encoder.o"
  VERBATIM)

add_custom_command(
  OUTPUT forward_unet.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/forward_unet.mlir
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
            empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, \
            func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline
            "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), \
            eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
            convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, \
            convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, \
            convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O3 
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyStableDiffusion/forward_unet.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/forward_unet.mlir
  COMMENT "Building forward_unet.o"
  VERBATIM)


add_custom_command(
  OUTPUT subgraph0_unet.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/subgraph0_unet.mlir 
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" |
          ${BUDDY_BINARY_DIR}/buddy-opt  
            -convert-elementwise-to-linalg
            -func-bufferize-dynamic-offset
            -arith-bufferize
            -func-bufferize
            -tensor-bufferize
            -linalg-bufferize
            -finalizing-bufferize
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3 
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyStableDiffusion/subgraph0_unet.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/subgraph0_unet.mlir
          buddy-opt
  COMMENT "Building subgraph0_unet.o"
  VERBATIM)

add_custom_command(
  OUTPUT forward_vae.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/forward_vae.mlir
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
            empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, \
            func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline
            "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), \
            eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
            convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, \
            convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, \
            convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O3 
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyStableDiffusion/forward_vae.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/forward_vae.mlir
  COMMENT "Building forward_vae.o"
  VERBATIM)


add_custom_command(
  OUTPUT subgraph0_vae.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/subgraph0_vae.mlir 
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" |
          ${BUDDY_BINARY_DIR}/buddy-opt  
            -convert-elementwise-to-linalg
            -func-bufferize-dynamic-offset
            -arith-bufferize
            -func-bufferize
            -tensor-bufferize
            -linalg-bufferize
            -finalizing-bufferize
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3 
            -o ${BUDDY_BINARY_DIR}/../examples/BuddyStableDiffusion/subgraph0_vae.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyStableDiffusion/subgraph0_vae.mlir
          buddy-opt
  COMMENT "Building subgraph0_vae.o"
  VERBATIM)

add_library(TEXTENCODER STATIC subgraph0_text_encoder.o forward_text_encoder.o)
add_library(UNET STATIC subgraph0_unet.o forward_unet.o)
add_library(VAE STATIC subgraph0_vae.o forward_vae.o)

target_compile_options(TEXTENCODER PRIVATE -ffunction-sections -fdata-sections -mcmodel=large -mlong-calls)
target_compile_options(UNET PRIVATE -ffunction-sections -fdata-sections -mcmodel=large -mlong-calls)
target_compile_options(VAE PRIVATE -ffunction-sections -fdata-sections -mcmodel=large -mlong-calls)

SET_TARGET_PROPERTIES(TEXTENCODER PROPERTIES LINKER_LANGUAGE C)
SET_TARGET_PROPERTIES(UNET PROPERTIES LINKER_LANGUAGE C)
SET_TARGET_PROPERTIES(VAE PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-stablediffusion-run buddy-stablediffusion-main.cpp)
target_link_directories(buddy-stablediffusion-run PRIVATE ${LLVM_LIBRARY_DIR})

target_link_options(buddy-stablediffusion-run PRIVATE -Wl,--no-relax)

set(BUDDY_STABLEDIFFUSION_LIBS TEXTENCODER UNET VAE mlir_c_runner_utils)

target_link_libraries(buddy-stablediffusion-run ${BUDDY_STABLEDIFFUSION_LIBS} ${JPEG_LIBRARIES} ${PNG_LIBRARIES})
include_directories(${JPEG_INCLUDE_DIRS})
include_directories(${PNG_INCLUDE_DIRS})
