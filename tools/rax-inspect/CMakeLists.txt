add_executable(rax-inspect rax-inspect.cpp)

target_compile_features(rax-inspect PRIVATE cxx_std_17)
target_compile_definitions(rax-inspect PRIVATE
  BUDDY_VERSION="${BUDDY_PACKAGE_VERSION}")

target_link_libraries(rax-inspect PRIVATE
  CLI11
  buddy_rax)

install(TARGETS rax-inspect
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_tools)
