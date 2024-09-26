module attributes {gpu.container_module} {
  func.func @subgraph1(%arg0: tensor<1x1x28x28xf32>, %arg1: tensor<6x1x5x5xf32>, %arg2: tensor<6xf32>, %arg3: tensor<16x6x5x5xf32>, %arg4: tensor<16xf32>, %arg5: tensor<120xf32>, %arg6: tensor<256x120xf32>, %arg7: tensor<84xf32>, %arg8: tensor<120x84xf32>, %arg9: tensor<10xf32>, %arg10: tensor<84x10xf32>) -> tensor<1x10xf32> {
    %0 = bufferization.to_memref %arg0 : memref<1x1x28x28xf32>
    %1 = bufferization.to_memref %arg1 : memref<6x1x5x5xf32>
    %2 = bufferization.to_memref %arg2 : memref<6xf32>
    %3 = bufferization.to_memref %arg3 : memref<16x6x5x5xf32>
    %4 = bufferization.to_memref %arg4 : memref<16xf32>
    %5 = bufferization.to_memref %arg5 : memref<120xf32>
    %6 = bufferization.to_memref %arg6 : memref<256x120xf32>
    %7 = bufferization.to_memref %arg7 : memref<84xf32>
    %8 = bufferization.to_memref %arg8 : memref<120x84xf32>
    %9 = bufferization.to_memref %arg9 : memref<10xf32>
    %10 = bufferization.to_memref %arg10 : memref<84x10xf32>
    %alloc = memref.alloc() : memref<1x6x24x24xf32>
    %cast = memref.cast %0 : memref<1x1x28x28xf32> to memref<*xf32>
    %cast_0 = memref.cast %1 : memref<6x1x5x5xf32> to memref<*xf32>
    %cast_1 = memref.cast %2 : memref<6xf32> to memref<*xf32>
    %cast_2 = memref.cast %alloc : memref<1x6x24x24xf32> to memref<*xf32>
    gpu.host_register %cast : memref<*xf32>
    gpu.host_register %cast_0 : memref<*xf32>
    gpu.host_register %cast_1 : memref<*xf32>
    gpu.host_register %cast_2 : memref<*xf32>
    %c1 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c5 = arith.constant 5 : index
    %c5_4 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1_5 = arith.constant 1 : index
    gpu.launch blocks(%arg11, %arg12, %arg13) in (%arg17 = %c1, %arg18 = %c6, %arg19 = %c4) threads(%arg14, %arg15, %arg16) in (%arg20 = %c16, %arg21 = %c16, %arg22 = %c1_5) {
      %c2 = arith.constant 2 : index
      %12 = arith.divui %arg13, %c2 : index
      %13 = arith.muli %12, %c16 : index
      %14 = arith.addi %13, %arg14 : index
      %15 = arith.remui %arg13, %c2 : index
      %16 = arith.muli %15, %c16 : index
      %17 = arith.addi %16, %arg15 : index
      %c1_140 = arith.constant 1 : index
      %c1_141 = arith.constant 1 : index
      %18 = arith.muli %14, %c1_140 : index
      %19 = arith.muli %17, %c1_141 : index
      %c24 = arith.constant 24 : index
      %c24_142 = arith.constant 24 : index
      %20 = arith.cmpi ult, %14, %c24 : index
      %21 = arith.cmpi ult, %17, %c24_142 : index
      %22 = arith.andi %20, %21 : i1
      %cst = arith.constant 0.000000e+00 : f32
      scf.if %22 {
        %23 = scf.for %arg23 = %c0 to %c1_3 step %c1_5 iter_args(%arg24 = %cst) -> (f32) {
          %26 = scf.for %arg25 = %c0 to %c5 step %c1_5 iter_args(%arg26 = %cst) -> (f32) {
            %28 = scf.for %arg27 = %c0 to %c5_4 step %c1_5 iter_args(%arg28 = %cst) -> (f32) {
              %30 = arith.addi %18, %arg25 : index
              %31 = arith.addi %19, %arg27 : index
              %32 = memref.load %0[%arg11, %arg23, %30, %31] : memref<1x1x28x28xf32>
              %33 = memref.load %1[%arg12, %arg23, %arg25, %arg27] : memref<6x1x5x5xf32>
              %34 = arith.mulf %32, %33 : f32
              %35 = arith.addf %arg28, %34 : f32
              scf.yield %35 : f32
            }
            %29 = arith.addf %28, %arg26 : f32
            scf.yield %29 : f32
          }
          %27 = arith.addf %26, %arg24 : f32
          scf.yield %27 : f32
        }
        %24 = memref.load %2[%arg12] : memref<6xf32>
        %25 = arith.addf %23, %24 : f32
        memref.store %25, %alloc[%arg11, %arg12, %14, %17] : memref<1x6x24x24xf32>
      }
      gpu.terminator
    }
    gpu.host_unregister %cast : memref<*xf32>
    gpu.host_unregister %cast_0 : memref<*xf32>
    gpu.host_unregister %cast_1 : memref<*xf32>
    gpu.host_unregister %cast_2 : memref<*xf32>
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c3456 = arith.constant 3456 : index
    %alloc_8 = memref.alloc() : memref<1xindex>
    memref.store %c3456, %alloc_8[%c0_6] : memref<1xindex>
    %reshape = memref.reshape %alloc(%alloc_8) : (memref<1x6x24x24xf32>, memref<1xindex>) -> memref<3456xf32>
    %cast_9 = memref.cast %alloc : memref<1x6x24x24xf32> to memref<*xf32>
    gpu.host_register %cast_9 : memref<*xf32>
    gpu.launch blocks(%arg11, %arg12, %arg13) in (%arg17 = %c1_7, %arg18 = %c1_7, %arg19 = %c1_7) threads(%arg14, %arg15, %arg16) in (%arg20 = %c512, %arg21 = %c1_7, %arg22 = %c1_7) {
      %cst = arith.constant 0.000000e+00 : f32
      scf.for %arg23 = %arg14 to %c3456 step %c512 {
        %12 = memref.load %reshape[%arg23] : memref<3456xf32>
        %13 = arith.maxnumf %12, %cst : f32
        memref.store %13, %reshape[%arg23] : memref<3456xf32>
      }
      gpu.terminator
    }
    gpu.host_unregister %cast_9 : memref<*xf32>
    %alloc_10 = memref.alloc() : memref<1x6x24x24xf32>
    memref.copy %alloc, %alloc_10 : memref<1x6x24x24xf32> to memref<1x6x24x24xf32>
    %alloc_11 = memref.alloc() : memref<1x6x12x12xf32>
    %cast_12 = memref.cast %alloc_10 : memref<1x6x24x24xf32> to memref<*xf32>
    %cast_13 = memref.cast %alloc_11 : memref<1x6x12x12xf32> to memref<*xf32>
    gpu.host_register %cast_12 : memref<*xf32>
    gpu.host_register %cast_13 : memref<*xf32>
    %c1_14 = arith.constant 1 : index
    %c6_15 = arith.constant 6 : index
    %c1_16 = arith.constant 1 : index
    %c16_17 = arith.constant 16 : index
    %c0_18 = arith.constant 0 : index
    %c1_19 = arith.constant 1 : index
    gpu.launch blocks(%arg11, %arg12, %arg13) in (%arg17 = %c1_14, %arg18 = %c6_15, %arg19 = %c1_16) threads(%arg14, %arg15, %arg16) in (%arg20 = %c16_17, %arg21 = %c16_17, %arg22 = %c1_19) {
      %c1_140 = arith.constant 1 : index
      %12 = arith.divui %arg13, %c1_140 : index
      %13 = arith.muli %12, %c16_17 : index
      %14 = arith.addi %13, %arg14 : index
      %15 = arith.remui %arg13, %c1_140 : index
      %16 = arith.muli %15, %c16_17 : index
      %17 = arith.addi %16, %arg15 : index
      %c2 = arith.constant 2 : index
      %c2_141 = arith.constant 2 : index
      %c2_142 = arith.constant 2 : index
      %c2_143 = arith.constant 2 : index
      %18 = arith.muli %14, %c2_142 : index
      %19 = arith.muli %17, %c2_143 : index
      %c12 = arith.constant 12 : index
      %c12_144 = arith.constant 12 : index
      %20 = arith.cmpi ult, %14, %c12 : index
      %21 = arith.cmpi ult, %17, %c12_144 : index
      %22 = arith.andi %20, %21 : i1
      scf.if %22 {
        %23 = memref.load %alloc_10[%arg11, %arg12, %18, %19] : memref<1x6x24x24xf32>
        %24 = scf.for %arg23 = %c0_18 to %c2 step %c1_19 iter_args(%arg24 = %23) -> (f32) {
          %25 = scf.for %arg25 = %c0_18 to %c2_141 step %c1_19 iter_args(%arg26 = %23) -> (f32) {
            %27 = arith.addi %18, %arg23 : index
            %28 = arith.addi %19, %arg25 : index
            %29 = memref.load %alloc_10[%arg11, %arg12, %27, %28] : memref<1x6x24x24xf32>
            %30 = arith.maxnumf %arg26, %29 : f32
            scf.yield %30 : f32
          }
          %26 = arith.maxnumf %25, %arg24 : f32
          scf.yield %26 : f32
        }
        memref.store %24, %alloc_11[%arg11, %arg12, %14, %17] : memref<1x6x12x12xf32>
      }
      gpu.terminator
    }
    gpu.host_unregister %cast_12 : memref<*xf32>
    gpu.host_unregister %cast_13 : memref<*xf32>
    %alloc_20 = memref.alloc() : memref<1x16x8x8xf32>
    %cast_21 = memref.cast %alloc_11 : memref<1x6x12x12xf32> to memref<*xf32>
    %cast_22 = memref.cast %3 : memref<16x6x5x5xf32> to memref<*xf32>
    %cast_23 = memref.cast %4 : memref<16xf32> to memref<*xf32>
    %cast_24 = memref.cast %alloc_20 : memref<1x16x8x8xf32> to memref<*xf32>
    gpu.host_register %cast_21 : memref<*xf32>
    gpu.host_register %cast_22 : memref<*xf32>
    gpu.host_register %cast_23 : memref<*xf32>
    gpu.host_register %cast_24 : memref<*xf32>
    %c1_25 = arith.constant 1 : index
    %c6_26 = arith.constant 6 : index
    %c16_27 = arith.constant 16 : index
    %c1_28 = arith.constant 1 : index
    %c16_29 = arith.constant 16 : index
    %c5_30 = arith.constant 5 : index
    %c5_31 = arith.constant 5 : index
    %c0_32 = arith.constant 0 : index
    %c1_33 = arith.constant 1 : index
    gpu.launch blocks(%arg11, %arg12, %arg13) in (%arg17 = %c1_25, %arg18 = %c16_27, %arg19 = %c1_28) threads(%arg14, %arg15, %arg16) in (%arg20 = %c16_29, %arg21 = %c16_29, %arg22 = %c1_33) {
      %c1_140 = arith.constant 1 : index
      %12 = arith.divui %arg13, %c1_140 : index
      %13 = arith.muli %12, %c16_29 : index
      %14 = arith.addi %13, %arg14 : index
      %15 = arith.remui %arg13, %c1_140 : index
      %16 = arith.muli %15, %c16_29 : index
      %17 = arith.addi %16, %arg15 : index
      %c1_141 = arith.constant 1 : index
      %c1_142 = arith.constant 1 : index
      %18 = arith.muli %14, %c1_141 : index
      %19 = arith.muli %17, %c1_142 : index
      %c8 = arith.constant 8 : index
      %c8_143 = arith.constant 8 : index
      %20 = arith.cmpi ult, %14, %c8 : index
      %21 = arith.cmpi ult, %17, %c8_143 : index
      %22 = arith.andi %20, %21 : i1
      %cst = arith.constant 0.000000e+00 : f32
      scf.if %22 {
        %23 = scf.for %arg23 = %c0_32 to %c6_26 step %c1_33 iter_args(%arg24 = %cst) -> (f32) {
          %26 = scf.for %arg25 = %c0_32 to %c5_30 step %c1_33 iter_args(%arg26 = %cst) -> (f32) {
            %28 = scf.for %arg27 = %c0_32 to %c5_31 step %c1_33 iter_args(%arg28 = %cst) -> (f32) {
              %30 = arith.addi %18, %arg25 : index
              %31 = arith.addi %19, %arg27 : index
              %32 = memref.load %alloc_11[%arg11, %arg23, %30, %31] : memref<1x6x12x12xf32>
              %33 = memref.load %3[%arg12, %arg23, %arg25, %arg27] : memref<16x6x5x5xf32>
              %34 = arith.mulf %32, %33 : f32
              %35 = arith.addf %arg28, %34 : f32
              scf.yield %35 : f32
            }
            %29 = arith.addf %28, %arg26 : f32
            scf.yield %29 : f32
          }
          %27 = arith.addf %26, %arg24 : f32
          scf.yield %27 : f32
        }
        %24 = memref.load %4[%arg12] : memref<16xf32>
        %25 = arith.addf %23, %24 : f32
        memref.store %25, %alloc_20[%arg11, %arg12, %14, %17] : memref<1x16x8x8xf32>
      }
      gpu.terminator
    }
    gpu.host_unregister %cast_21 : memref<*xf32>
    gpu.host_unregister %cast_22 : memref<*xf32>
    gpu.host_unregister %cast_23 : memref<*xf32>
    gpu.host_unregister %cast_24 : memref<*xf32>
    %c0_34 = arith.constant 0 : index
    %c1_35 = arith.constant 1 : index
    %c512_36 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %alloc_37 = memref.alloc() : memref<1xindex>
    memref.store %c1024, %alloc_37[%c0_34] : memref<1xindex>
    %reshape_38 = memref.reshape %alloc_20(%alloc_37) : (memref<1x16x8x8xf32>, memref<1xindex>) -> memref<1024xf32>
    %cast_39 = memref.cast %alloc_20 : memref<1x16x8x8xf32> to memref<*xf32>
    gpu.host_register %cast_39 : memref<*xf32>
    gpu.launch blocks(%arg11, %arg12, %arg13) in (%arg17 = %c1_35, %arg18 = %c1_35, %arg19 = %c1_35) threads(%arg14, %arg15, %arg16) in (%arg20 = %c512_36, %arg21 = %c1_35, %arg22 = %c1_35) {
      %cst = arith.constant 0.000000e+00 : f32
      scf.for %arg23 = %arg14 to %c1024 step %c512_36 {
        %12 = memref.load %reshape_38[%arg23] : memref<1024xf32>
        %13 = arith.maxnumf %12, %cst : f32
        memref.store %13, %reshape_38[%arg23] : memref<1024xf32>
      }
      gpu.terminator
    }
    gpu.host_unregister %cast_39 : memref<*xf32>
    %alloc_40 = memref.alloc() : memref<1x16x8x8xf32>
    memref.copy %alloc_20, %alloc_40 : memref<1x16x8x8xf32> to memref<1x16x8x8xf32>
    %alloc_41 = memref.alloc() : memref<1x16x4x4xf32>
    %cast_42 = memref.cast %alloc_40 : memref<1x16x8x8xf32> to memref<*xf32>
    %cast_43 = memref.cast %alloc_41 : memref<1x16x4x4xf32> to memref<*xf32>
    gpu.host_register %cast_42 : memref<*xf32>
    gpu.host_register %cast_43 : memref<*xf32>
    %c1_44 = arith.constant 1 : index
    %c16_45 = arith.constant 16 : index
    %c1_46 = arith.constant 1 : index
    %c16_47 = arith.constant 16 : index
    %c0_48 = arith.constant 0 : index
    %c1_49 = arith.constant 1 : index
    gpu.launch blocks(%arg11, %arg12, %arg13) in (%arg17 = %c1_44, %arg18 = %c16_45, %arg19 = %c1_46) threads(%arg14, %arg15, %arg16) in (%arg20 = %c16_47, %arg21 = %c16_47, %arg22 = %c1_49) {
      %c1_140 = arith.constant 1 : index
      %12 = arith.divui %arg13, %c1_140 : index
      %13 = arith.muli %12, %c16_47 : index
      %14 = arith.addi %13, %arg14 : index
      %15 = arith.remui %arg13, %c1_140 : index
      %16 = arith.muli %15, %c16_47 : index
      %17 = arith.addi %16, %arg15 : index
      %c2 = arith.constant 2 : index
      %c2_141 = arith.constant 2 : index
      %c2_142 = arith.constant 2 : index
      %c2_143 = arith.constant 2 : index
      %18 = arith.muli %14, %c2_142 : index
      %19 = arith.muli %17, %c2_143 : index
      %c4_144 = arith.constant 4 : index
      %c4_145 = arith.constant 4 : index
      %20 = arith.cmpi ult, %14, %c4_144 : index
      %21 = arith.cmpi ult, %17, %c4_145 : index
      %22 = arith.andi %20, %21 : i1
      scf.if %22 {
        %23 = memref.load %alloc_40[%arg11, %arg12, %18, %19] : memref<1x16x8x8xf32>
        %24 = scf.for %arg23 = %c0_48 to %c2 step %c1_49 iter_args(%arg24 = %23) -> (f32) {
          %25 = scf.for %arg25 = %c0_48 to %c2_141 step %c1_49 iter_args(%arg26 = %23) -> (f32) {
            %27 = arith.addi %18, %arg23 : index
            %28 = arith.addi %19, %arg25 : index
            %29 = memref.load %alloc_40[%arg11, %arg12, %27, %28] : memref<1x16x8x8xf32>
            %30 = arith.maxnumf %arg26, %29 : f32
            scf.yield %30 : f32
          }
          %26 = arith.maxnumf %25, %arg24 : f32
          scf.yield %26 : f32
        }
        memref.store %24, %alloc_41[%arg11, %arg12, %14, %17] : memref<1x16x4x4xf32>
      }
      gpu.terminator
    }
    gpu.host_unregister %cast_42 : memref<*xf32>
    gpu.host_unregister %cast_43 : memref<*xf32>
    %alloc_50 = memref.alloc() : memref<2xindex>
    %c0_51 = arith.constant 0 : index
    %c1_52 = arith.constant 1 : index
    memref.store %c1_52, %alloc_50[%c0_51] : memref<2xindex>
    %c1_53 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    memref.store %c256, %alloc_50[%c1_53] : memref<2xindex>
    %reshape_54 = memref.reshape %alloc_41(%alloc_50) : (memref<1x16x4x4xf32>, memref<2xindex>) -> memref<1x256xf32>
    %c0_55 = arith.constant 0 : index
    %c1_56 = arith.constant 1 : index
    %c512_57 = arith.constant 512 : index
    %c256_58 = arith.constant 256 : index
    %c30720 = arith.constant 30720 : index
    %c120 = arith.constant 120 : index
    %alloc_59 = memref.alloc() : memref<1xindex>
    %alloc_60 = memref.alloc() : memref<1xindex>
    %alloc_61 = memref.alloc() : memref<1xindex>
    memref.store %c256_58, %alloc_59[%c0_55] : memref<1xindex>
    memref.store %c30720, %alloc_60[%c0_55] : memref<1xindex>
    memref.store %c120, %alloc_61[%c0_55] : memref<1xindex>
    %reshape_62 = memref.reshape %reshape_54(%alloc_59) : (memref<1x256xf32>, memref<1xindex>) -> memref<256xf32>
    %reshape_63 = memref.reshape %6(%alloc_60) : (memref<256x120xf32>, memref<1xindex>) -> memref<30720xf32>
    %reshape_64 = memref.reshape %5(%alloc_61) : (memref<120xf32>, memref<1xindex>) -> memref<120xf32>
    %cast_65 = memref.cast %reshape_62 : memref<256xf32> to memref<*xf32>
    gpu.host_register %cast_65 : memref<*xf32>
    %cast_66 = memref.cast %reshape_63 : memref<30720xf32> to memref<*xf32>
    gpu.host_register %cast_66 : memref<*xf32>
    %cast_67 = memref.cast %reshape_64 : memref<120xf32> to memref<*xf32>
    gpu.host_register %cast_67 : memref<*xf32>
    %c1_68 = arith.constant 1 : index
    %c120_69 = arith.constant 120 : index
    %c256_70 = arith.constant 256 : index
    gpu.launch blocks(%arg11, %arg12, %arg13) in (%arg17 = %c1_56, %arg18 = %c1_56, %arg19 = %c1_56) threads(%arg14, %arg15, %arg16) in (%arg20 = %c512_57, %arg21 = %c1_56, %arg22 = %c1_56) {
      scf.for %arg23 = %arg14 to %c1_68 step %c512_57 {
        scf.for %arg24 = %arg15 to %c120_69 step %c1_56 {
          %cst = arith.constant 0.000000e+00 : f32
          %12 = scf.for %arg25 = %c0_55 to %c256_70 step %c1_56 iter_args(%arg26 = %cst) -> (f32) {
            %19 = arith.muli %arg23, %c256_70 : index
            %20 = arith.addi %19, %arg25 : index
            %21 = memref.load %reshape_62[%20] : memref<256xf32>
            %22 = arith.muli %arg25, %c120_69 : index
            %23 = arith.addi %22, %arg24 : index
            %24 = memref.load %reshape_63[%23] : memref<30720xf32>
            %25 = arith.mulf %21, %24 : f32
            %26 = arith.addf %arg26, %25 : f32
            scf.yield %26 : f32
          }
          %13 = arith.muli %arg23, %c120_69 : index
          %14 = arith.addi %13, %arg24 : index
          %15 = memref.load %reshape_64[%14] : memref<120xf32>
          %16 = arith.addf %12, %15 : f32
          %17 = arith.muli %arg23, %c120_69 : index
          %18 = arith.addi %17, %arg24 : index
          memref.store %16, %reshape_64[%18] : memref<120xf32>
        }
      }
      gpu.terminator
    }
    %alloc_71 = memref.alloc() : memref<1x120xf32>
    %alloc_72 = memref.alloc() : memref<2xindex>
    %c1_73 = arith.constant 1 : index
    %c0_74 = arith.constant 0 : index
    memref.store %c1_73, %alloc_72[%c0_74] : memref<2xindex>
    %c120_75 = arith.constant 120 : index
    %c1_76 = arith.constant 1 : index
    memref.store %c120_75, %alloc_72[%c1_76] : memref<2xindex>
    %reshape_77 = memref.reshape %5(%alloc_72) : (memref<120xf32>, memref<2xindex>) -> memref<1x120xf32>
    memref.copy %reshape_77, %alloc_71 : memref<1x120xf32> to memref<1x120xf32>
    %c0_78 = arith.constant 0 : index
    %c1_79 = arith.constant 1 : index
    %c512_80 = arith.constant 512 : index
    %c120_81 = arith.constant 120 : index
    %alloc_82 = memref.alloc() : memref<1xindex>
    memref.store %c120_81, %alloc_82[%c0_78] : memref<1xindex>
    %reshape_83 = memref.reshape %alloc_71(%alloc_82) : (memref<1x120xf32>, memref<1xindex>) -> memref<120xf32>
    %cast_84 = memref.cast %alloc_71 : memref<1x120xf32> to memref<*xf32>
    gpu.host_register %cast_84 : memref<*xf32>
    gpu.launch blocks(%arg11, %arg12, %arg13) in (%arg17 = %c1_79, %arg18 = %c1_79, %arg19 = %c1_79) threads(%arg14, %arg15, %arg16) in (%arg20 = %c512_80, %arg21 = %c1_79, %arg22 = %c1_79) {
      %cst = arith.constant 0.000000e+00 : f32
      scf.for %arg23 = %arg14 to %c120_81 step %c512_80 {
        %12 = memref.load %reshape_83[%arg23] : memref<120xf32>
        %13 = arith.maxnumf %12, %cst : f32
        memref.store %13, %reshape_83[%arg23] : memref<120xf32>
      }
      gpu.terminator
    }
    gpu.host_unregister %cast_84 : memref<*xf32>
    %alloc_85 = memref.alloc() : memref<1x120xf32>
    memref.copy %alloc_71, %alloc_85 : memref<1x120xf32> to memref<1x120xf32>
    %c0_86 = arith.constant 0 : index
    %c1_87 = arith.constant 1 : index
    %c512_88 = arith.constant 512 : index
    %c120_89 = arith.constant 120 : index
    %c10080 = arith.constant 10080 : index
    %c84 = arith.constant 84 : index
    %alloc_90 = memref.alloc() : memref<1xindex>
    %alloc_91 = memref.alloc() : memref<1xindex>
    %alloc_92 = memref.alloc() : memref<1xindex>
    memref.store %c120_89, %alloc_90[%c0_86] : memref<1xindex>
    memref.store %c10080, %alloc_91[%c0_86] : memref<1xindex>
    memref.store %c84, %alloc_92[%c0_86] : memref<1xindex>
    %reshape_93 = memref.reshape %alloc_85(%alloc_90) : (memref<1x120xf32>, memref<1xindex>) -> memref<120xf32>
    %reshape_94 = memref.reshape %8(%alloc_91) : (memref<120x84xf32>, memref<1xindex>) -> memref<10080xf32>
    %reshape_95 = memref.reshape %7(%alloc_92) : (memref<84xf32>, memref<1xindex>) -> memref<84xf32>
    %cast_96 = memref.cast %reshape_93 : memref<120xf32> to memref<*xf32>
    gpu.host_register %cast_96 : memref<*xf32>
    %cast_97 = memref.cast %reshape_94 : memref<10080xf32> to memref<*xf32>
    gpu.host_register %cast_97 : memref<*xf32>
    %cast_98 = memref.cast %reshape_95 : memref<84xf32> to memref<*xf32>
    gpu.host_register %cast_98 : memref<*xf32>
    %c1_99 = arith.constant 1 : index
    %c84_100 = arith.constant 84 : index
    %c120_101 = arith.constant 120 : index
    gpu.launch blocks(%arg11, %arg12, %arg13) in (%arg17 = %c1_87, %arg18 = %c1_87, %arg19 = %c1_87) threads(%arg14, %arg15, %arg16) in (%arg20 = %c512_88, %arg21 = %c1_87, %arg22 = %c1_87) {
      scf.for %arg23 = %arg14 to %c1_99 step %c512_88 {
        scf.for %arg24 = %arg15 to %c84_100 step %c1_87 {
          %cst = arith.constant 0.000000e+00 : f32
          %12 = scf.for %arg25 = %c0_86 to %c120_101 step %c1_87 iter_args(%arg26 = %cst) -> (f32) {
            %19 = arith.muli %arg23, %c120_101 : index
            %20 = arith.addi %19, %arg25 : index
            %21 = memref.load %reshape_93[%20] : memref<120xf32>
            %22 = arith.muli %arg25, %c84_100 : index
            %23 = arith.addi %22, %arg24 : index
            %24 = memref.load %reshape_94[%23] : memref<10080xf32>
            %25 = arith.mulf %21, %24 : f32
            %26 = arith.addf %arg26, %25 : f32
            scf.yield %26 : f32
          }
          %13 = arith.muli %arg23, %c84_100 : index
          %14 = arith.addi %13, %arg24 : index
          %15 = memref.load %reshape_95[%14] : memref<84xf32>
          %16 = arith.addf %12, %15 : f32
          %17 = arith.muli %arg23, %c84_100 : index
          %18 = arith.addi %17, %arg24 : index
          memref.store %16, %reshape_95[%18] : memref<84xf32>
        }
      }
      gpu.terminator
    }
    %alloc_102 = memref.alloc() : memref<1x84xf32>
    %alloc_103 = memref.alloc() : memref<2xindex>
    %c1_104 = arith.constant 1 : index
    %c0_105 = arith.constant 0 : index
    memref.store %c1_104, %alloc_103[%c0_105] : memref<2xindex>
    %c84_106 = arith.constant 84 : index
    %c1_107 = arith.constant 1 : index
    memref.store %c84_106, %alloc_103[%c1_107] : memref<2xindex>
    %reshape_108 = memref.reshape %7(%alloc_103) : (memref<84xf32>, memref<2xindex>) -> memref<1x84xf32>
    memref.copy %reshape_108, %alloc_102 : memref<1x84xf32> to memref<1x84xf32>
    %c0_109 = arith.constant 0 : index
    %c1_110 = arith.constant 1 : index
    %c512_111 = arith.constant 512 : index
    %c84_112 = arith.constant 84 : index
    %alloc_113 = memref.alloc() : memref<1xindex>
    memref.store %c84_112, %alloc_113[%c0_109] : memref<1xindex>
    %reshape_114 = memref.reshape %alloc_102(%alloc_113) : (memref<1x84xf32>, memref<1xindex>) -> memref<84xf32>
    %cast_115 = memref.cast %alloc_102 : memref<1x84xf32> to memref<*xf32>
    gpu.host_register %cast_115 : memref<*xf32>
    gpu.launch blocks(%arg11, %arg12, %arg13) in (%arg17 = %c1_110, %arg18 = %c1_110, %arg19 = %c1_110) threads(%arg14, %arg15, %arg16) in (%arg20 = %c512_111, %arg21 = %c1_110, %arg22 = %c1_110) {
      %cst = arith.constant 0.000000e+00 : f32
      scf.for %arg23 = %arg14 to %c84_112 step %c512_111 {
        %12 = memref.load %reshape_114[%arg23] : memref<84xf32>
        %13 = arith.maxnumf %12, %cst : f32
        memref.store %13, %reshape_114[%arg23] : memref<84xf32>
      }
      gpu.terminator
    }
    gpu.host_unregister %cast_115 : memref<*xf32>
    %alloc_116 = memref.alloc() : memref<1x84xf32>
    memref.copy %alloc_102, %alloc_116 : memref<1x84xf32> to memref<1x84xf32>
    %c0_117 = arith.constant 0 : index
    %c1_118 = arith.constant 1 : index
    %c512_119 = arith.constant 512 : index
    %c84_120 = arith.constant 84 : index
    %c840 = arith.constant 840 : index
    %c10 = arith.constant 10 : index
    %alloc_121 = memref.alloc() : memref<1xindex>
    %alloc_122 = memref.alloc() : memref<1xindex>
    %alloc_123 = memref.alloc() : memref<1xindex>
    memref.store %c84_120, %alloc_121[%c0_117] : memref<1xindex>
    memref.store %c840, %alloc_122[%c0_117] : memref<1xindex>
    memref.store %c10, %alloc_123[%c0_117] : memref<1xindex>
    %reshape_124 = memref.reshape %alloc_116(%alloc_121) : (memref<1x84xf32>, memref<1xindex>) -> memref<84xf32>
    %reshape_125 = memref.reshape %10(%alloc_122) : (memref<84x10xf32>, memref<1xindex>) -> memref<840xf32>
    %reshape_126 = memref.reshape %9(%alloc_123) : (memref<10xf32>, memref<1xindex>) -> memref<10xf32>
    %cast_127 = memref.cast %reshape_124 : memref<84xf32> to memref<*xf32>
    gpu.host_register %cast_127 : memref<*xf32>
    %cast_128 = memref.cast %reshape_125 : memref<840xf32> to memref<*xf32>
    gpu.host_register %cast_128 : memref<*xf32>
    %cast_129 = memref.cast %reshape_126 : memref<10xf32> to memref<*xf32>
    gpu.host_register %cast_129 : memref<*xf32>
    %c1_130 = arith.constant 1 : index
    %c10_131 = arith.constant 10 : index
    %c84_132 = arith.constant 84 : index
    gpu.launch blocks(%arg11, %arg12, %arg13) in (%arg17 = %c1_118, %arg18 = %c1_118, %arg19 = %c1_118) threads(%arg14, %arg15, %arg16) in (%arg20 = %c512_119, %arg21 = %c1_118, %arg22 = %c1_118) {
      scf.for %arg23 = %arg14 to %c1_130 step %c512_119 {
        scf.for %arg24 = %arg15 to %c10_131 step %c1_118 {
          %cst = arith.constant 0.000000e+00 : f32
          %12 = scf.for %arg25 = %c0_117 to %c84_132 step %c1_118 iter_args(%arg26 = %cst) -> (f32) {
            %19 = arith.muli %arg23, %c84_132 : index
            %20 = arith.addi %19, %arg25 : index
            %21 = memref.load %reshape_124[%20] : memref<84xf32>
            %22 = arith.muli %arg25, %c10_131 : index
            %23 = arith.addi %22, %arg24 : index
            %24 = memref.load %reshape_125[%23] : memref<840xf32>
            %25 = arith.mulf %21, %24 : f32
            %26 = arith.addf %arg26, %25 : f32
            scf.yield %26 : f32
          }
          %13 = arith.muli %arg23, %c10_131 : index
          %14 = arith.addi %13, %arg24 : index
          %15 = memref.load %reshape_126[%14] : memref<10xf32>
          %16 = arith.addf %12, %15 : f32
          %17 = arith.muli %arg23, %c10_131 : index
          %18 = arith.addi %17, %arg24 : index
          memref.store %16, %reshape_126[%18] : memref<10xf32>
        }
      }
      gpu.terminator
    }
    %alloc_133 = memref.alloc() : memref<1x10xf32>
    %alloc_134 = memref.alloc() : memref<2xindex>
    %c1_135 = arith.constant 1 : index
    %c0_136 = arith.constant 0 : index
    memref.store %c1_135, %alloc_134[%c0_136] : memref<2xindex>
    %c10_137 = arith.constant 10 : index
    %c1_138 = arith.constant 1 : index
    memref.store %c10_137, %alloc_134[%c1_138] : memref<2xindex>
    %reshape_139 = memref.reshape %9(%alloc_134) : (memref<10xf32>, memref<2xindex>) -> memref<1x10xf32>
    memref.copy %reshape_139, %alloc_133 : memref<1x10xf32> to memref<1x10xf32>
    %11 = bufferization.to_tensor %alloc_133 : memref<1x10xf32>
    return %11 : tensor<1x10xf32>
  }
}

