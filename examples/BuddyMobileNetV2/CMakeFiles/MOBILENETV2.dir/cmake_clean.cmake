file(REMOVE_RECURSE
  "/BuddyMobileNetV2/arg0.data"
  "/BuddyMobileNetV2/arg1.data"
  "/BuddyMobileNetV2/forward.mlir"
  "/BuddyMobileNetV2/subgraph0.mlir"
  "forward.o"
  "libMOBILENETV2.a"
  "libMOBILENETV2.pdb"
  "subgraph0.o"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/MOBILENETV2.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
