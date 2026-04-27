include(CheckCXXSourceCompiles)

find_package(TorqTile CONFIG REQUIRED)

add_library(buddy_matmul_torq STATIC MatmulTorq.cpp)
target_include_directories(buddy_matmul_torq PUBLIC
  ${BUDDY_SOURCE_DIR}/frontend/Interfaces
  ${BUDDY_SOURCE_DIR}/runtime/include
)

target_link_libraries(buddy_matmul_torq PUBLIC TorqTile::TorqTile)
target_compile_features(buddy_matmul_torq PRIVATE cxx_std_17)
