if(BUDDY_BUILD_DEEPSEEK_R1_MODEL)
  add_subdirectory(deepseek_r1)
endif()

if(BUDDY_BUILD_LLAMA31_TT_MODEL OR BUDDY_BUILD_LLAMA32_TT_MODEL)
  add_subdirectory(llama31_tt)
endif()

if(BUDDY_BUILD_WHISPER_MODEL)
  add_subdirectory(whisper)
endif()

if(BUDDY_BUILD_QWEN3_VL_MODEL)
  add_subdirectory(qwen3_vl)
endif()

if(BUDDY_BUILD_BGE_M3_MODEL)
  add_subdirectory(bge_m3)
endif()
