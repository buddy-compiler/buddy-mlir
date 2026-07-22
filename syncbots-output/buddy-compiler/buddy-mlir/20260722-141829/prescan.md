## Verified affected locations (207 file locations, 4 patterns)
The patterns below were confirmed present in this repository by grep.
Fix ONLY these; all other changelog entries had 0 hits and can be ignored.

**`CastOp`**
  midend/lib/Utils/Utils.cpp:92
  midend/lib/Utils/Utils.cpp:100
  midend/lib/Utils/GPUUtils.cpp:128
  midend/lib/Utils/GPUUtils.cpp:146
  midend/lib/Utils/AffineTransformUtils.cpp:242
  midend/lib/Utils/AffineTransformUtils.cpp:245
  midend/lib/Utils/AffineTransformUtils.cpp:321
  midend/lib/Utils/AffineTransformUtils.cpp:324
  midend/lib/Utils/AffineTransformUtils.cpp:422
  midend/lib/Utils/AffineTransformUtils.cpp:425
  midend/lib/Dialect/AME/Transforms/LegalizeForLLVMExport.cpp:60
  midend/lib/Conversion/LowerLinalgToBOSCAME/LowerLinalgToBOSCAME.cpp:147
  midend/lib/Conversion/LowerLinalgToBOSCAME/LowerLinalgToBOSCAME.cpp:149
  midend/lib/Conversion/LowerLinalgToBOSCAME/LowerLinalgToBOSCAME.cpp:151
  midend/lib/Conversion/LowerLinalgToBOSCAME/LowerLinalgToBOSCAME.cpp:178
  ... and 86 more locations

**`TargetOp`**
  midend/lib/Dialect/RVV/Transforms/LegalizeForLLVMExport.cpp:36
  midend/lib/Dialect/RVV/Transforms/LegalizeForLLVMExport.cpp:73

**`tosa.matmul`**
  tools/buddy-opt/buddy-opt.cpp:210
  midend/lib/Conversion/GraphRedundancyElimination/SimplifyTosaMatmulScalar.cpp:16
  midend/lib/Conversion/GraphRedundancyElimination/SimplifyTosaMatmulScalar.cpp:41
  midend/lib/Conversion/GraphRedundancyElimination/SimplifyTosaMatmulScalar.cpp:87
  midend/lib/Conversion/GraphRedundancyElimination/SimplifyTosaMatmulScalar.cpp:120

**`arith::AddFOp`**
  midend/lib/Utils/Utils.cpp:203
  midend/lib/Utils/Utils.cpp:204
  midend/lib/Utils/Utils.cpp:227
  midend/lib/Utils/DIPUtils.cpp:306
  midend/lib/Utils/DIPUtils.cpp:327
  midend/lib/Utils/DIPUtils.cpp:340
  midend/lib/Utils/DIPUtils.cpp:506
  midend/lib/Utils/DIPUtils.cpp:508
  midend/lib/Utils/DIPUtils.cpp:541
  midend/lib/Utils/DIPUtils.cpp:584
  midend/lib/Utils/DIPUtils.cpp:589
  midend/lib/Utils/DIPUtils.cpp:672
  midend/lib/Utils/DIPUtils.cpp:676
  midend/lib/Utils/DIPUtils.cpp:678
  midend/lib/Utils/DIPUtils.cpp:858
  ... and 84 more locations
