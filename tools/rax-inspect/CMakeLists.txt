add_executable(rax-inspect rax-inspect.cpp)

target_compile_features(rax-inspect PRIVATE cxx_std_17)

target_link_libraries(rax-inspect PRIVATE buddy_rax)

install(TARGETS rax-inspect
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_tools)
