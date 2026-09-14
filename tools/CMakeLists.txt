add_subdirectory(buddy-opt)
add_subdirectory(buddy-translate)
add_subdirectory(buddy-llc)
add_subdirectory(buddy-lsp-server)

# RAX tools: rax-pack packs .rhal text into a binary .rax FlatBuffer;
# rax-inspect prints the contents of a .rax file.
# flatc and buddy/runtime/rax/RAX.h are handled by runtime/rax/CMakeLists.txt.
add_subdirectory(rax-pack)
add_subdirectory(rax-inspect)

# buddy-cli: generic inference CLI — loads any .rax model and runs it.
add_subdirectory(buddy-cli)

# buddy-server: generic HTTP serving entry point. Model backends are optional
# and registered by tools/buddy-server when their targets are available.
add_subdirectory(buddy-server)
