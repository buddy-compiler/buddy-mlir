macro(CHECK_SIMD)

  include(CheckCXXSourceRuns)
  set(_CHECK_SIMD_ORIG_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")

#-------------------------------------------------------------------------------
# Check Intel SSE
#-------------------------------------------------------------------------------

  set(CMAKE_REQUIRED_FLAGS -msse)
  check_cxx_source_runs(
    "
    #include <immintrin.h>
    int main() {   
      __m128 x;
      x = _mm_set_ps(1.0f,1.0f,1.0f,1.0f);
      return 0;
    }
    " HAVE_SSE)

  if(${HAVE_SSE})
    message(STATUS "\tSSE support - yes")
  else()
    message(STATUS "\tSSE support - no")
  endif(${HAVE_SSE})

#-------------------------------------------------------------------------------
# Check Intel AVX2
#-------------------------------------------------------------------------------

  set(CMAKE_REQUIRED_FLAGS -mavx2)
  check_cxx_source_runs(
    "
    #include <immintrin.h>
    int main() {
      int data[8] = {0,0,0,0,0,0,0,0};
      __m256i a = _mm256_loadu_si256((const __m256i *)data);
      __m256i b = _mm256_bslli_epi128(a, 1);
      return 0;
    }
    " HAVE_AVX2)

  if(${HAVE_AVX2})
    message(STATUS "\tAVX2 support - yes")
  else()
    message(STATUS "\tAVX2 support - no")
  endif(${HAVE_AVX2})

#-------------------------------------------------------------------------------
# Check Intel AVX512
#-------------------------------------------------------------------------------

SET(CMAKE_REQUIRED_FLAGS -mavx512f)
check_cxx_source_runs(
  "
  #include <immintrin.h>
  int main() {
    float data[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    __m512 vector = _mm512_loadu_ps(data);
    return 0;
  }
  " HAVE_AVX512)

  if(${HAVE_AVX512})
    message(STATUS "\tAVX512 support - yes")
  else()
    message(STATUS "\tAVX512 support - no")
  endif(${HAVE_AVX512})

#-------------------------------------------------------------------------------
# Check Intel AMX
#-------------------------------------------------------------------------------

  set(CMAKE_REQUIRED_FLAGS "-mamx-tile -mamx-int8 -mamx-bf16")
  check_cxx_source_runs(
    "
    #include <immintrin.h>
    #if defined(__linux__) && defined(__x86_64__)
    #include <sys/syscall.h>
    #include <unistd.h>
    #ifndef ARCH_REQ_XCOMP_PERM
    #define ARCH_REQ_XCOMP_PERM 0x1022
    #endif
    #ifndef XFEATURE_XTILEDATA
    #define XFEATURE_XTILEDATA 18
    #endif
    #ifndef SYS_arch_prctl
    #define SYS_arch_prctl 158
    #endif
    static int request_amx_permissions() {
      long ret = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
      return ret;
    }
    #else
    static int request_amx_permissions() { return 0; }
    #endif
    alignas(64) static unsigned char cfg[64] = {0};
    int main() {
      if (request_amx_permissions() != 0)
        return 1;
      cfg[0] = 1;    // palette id
      cfg[16] = 16;  // rows for tile 0
      cfg[24] = 64;  // bytes per row (32 elements * 2 bytes)
      _tile_loadconfig(cfg);
      _tile_release();
      return 0;
    }
    "
    HAVE_AMX)

  if(HAVE_AMX)
    set(HAVE_AMX 1)
    message(STATUS "\tAMX support - yes")
  else()
    set(HAVE_AMX 0)
    message(STATUS "\tAMX support - no")
  endif()

#-------------------------------------------------------------------------------
# Check Arm Neon
#-------------------------------------------------------------------------------

  check_cxx_source_runs(
    "
    #include <arm_neon.h>
    int main() {
      float32x4_t a;
      float A[] = {1.0,2.0,3.0,4.0};
      a = vld1q_f32(A);
      return 0;
    }
    " HAVE_NEON)

  if(${HAVE_NEON})
    message(STATUS "\tArm Neon support - yes")
  else()
    message(STATUS "\tArm Neon support - no")
  endif(${HAVE_NEON})

#-------------------------------------------------------------------------------
# Check RISC-V Vector
#-------------------------------------------------------------------------------

  check_cxx_source_runs(
    "
    #include <riscv_vector.h>
    int main() {
      int avl = 70;
      int vl = vsetvl_e32m2(avl);
      return 0;
    }
    " HAVE_LOCAL_RVV)

    if(${HAVE_LOCAL_RVV})
      message(STATUS "\tRISC-V Vector local support - yes")
    else()
      message(STATUS "\tRISC-V Vector local support - no")
    endif(${HAVE_LOCAL_RVV})

  set(CMAKE_REQUIRED_FLAGS "${_CHECK_SIMD_ORIG_REQUIRED_FLAGS}")

endmacro(CHECK_SIMD)
