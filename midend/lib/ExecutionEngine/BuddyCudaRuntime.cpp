//===- BuddyCudaRuntime.cpp - Override MLIR CUDA runtime for managed mem --===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Overrides mgpuMemAlloc/mgpuMemcpy/mgpuMemFree from MLIR's
// CudaRuntimeWrappers with managed-memory-aware versions. This static
// library is linked before mlir_cuda_runtime so that the linker picks
// these definitions first.
//
// mgpuMemAlloc  — always uses cuMemAllocManaged (unified memory),
//                 so buffers are accessible from both CPU and GPU.
// mgpuMemFree   — queries pointer type; skips cuMemFree for non-CUDA
//                 pointers (host heap, .rodata) to avoid context corruption.
// mgpuMemcpy    — auto-detects direction via cuPointerGetAttribute and
//                 dispatches to cuMemcpyHtoDAsync / cuMemcpyDtoHAsync /
//                 cuMemcpyAsync accordingly, handling pageable host memory.
//
//===----------------------------------------------------------------------===//

#include "cuda.h"
#include <cstdint>
#include <cstdio>

#define CUDA_REPORT_IF_ERROR(expr)                                             \
  [](CUresult result) {                                                        \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = nullptr;                                                \
    cuGetErrorName(result, &name);                                             \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

extern "C" void *mgpuMemAlloc(uint64_t sizeBytes, CUstream /*stream*/,
                               bool /*isHostShared*/) {
  CUdeviceptr ptr = 0;
  if (sizeBytes == 0)
    return reinterpret_cast<void *>(ptr);
  CUDA_REPORT_IF_ERROR(
      cuMemAllocManaged(&ptr, sizeBytes, CU_MEM_ATTACH_GLOBAL));
  return reinterpret_cast<void *>(ptr);
}

extern "C" void mgpuMemFree(void *ptr, CUstream /*stream*/) {
  if (!ptr)
    return;
  unsigned int memType = 0;
  cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                        reinterpret_cast<CUdeviceptr>(ptr));
  if (memType != CU_MEMORYTYPE_DEVICE && memType != CU_MEMORYTYPE_UNIFIED)
    return;
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}

extern "C" void mgpuMemcpy(void *dst, void *src, size_t sizeBytes,
                            CUstream stream) {
  unsigned int dstMemType = 0, srcMemType = 0;
  cuPointerGetAttribute(&dstMemType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                        reinterpret_cast<CUdeviceptr>(dst));
  cuPointerGetAttribute(&srcMemType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                        reinterpret_cast<CUdeviceptr>(src));

  const bool dstIsDevice =
      (dstMemType == CU_MEMORYTYPE_DEVICE || dstMemType == CU_MEMORYTYPE_UNIFIED);
  const bool srcIsDevice =
      (srcMemType == CU_MEMORYTYPE_DEVICE || srcMemType == CU_MEMORYTYPE_UNIFIED);

  if (dstIsDevice && srcIsDevice) {
    CUDA_REPORT_IF_ERROR(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                       reinterpret_cast<CUdeviceptr>(src),
                                       sizeBytes, stream));
  } else if (dstIsDevice) {
    CUDA_REPORT_IF_ERROR(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst),
                                           src, sizeBytes, stream));
  } else if (srcIsDevice) {
    CUDA_REPORT_IF_ERROR(
        cuMemcpyDtoHAsync(dst, reinterpret_cast<CUdeviceptr>(src), sizeBytes,
                          stream));
  } else {
    CUDA_REPORT_IF_ERROR(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                       reinterpret_cast<CUdeviceptr>(src),
                                       sizeBytes, stream));
  }
}
