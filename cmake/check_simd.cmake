macro(CHECK_SIMD)

  include(CheckCXXSourceRuns)

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

endmacro(CHECK_SIMD)
