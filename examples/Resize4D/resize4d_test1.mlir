#map = affine_map<(d0) -> (d0)>
module {
  func.func @resize_2d_nearest_neighbour_interpolation(%arg0: memref<?x?xf32>, %arg1: f32, %arg2: f32, %arg3: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %c0 : index to i32
    %1 = arith.sitofp %0 : i32 to f32
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg3, %c0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg3, %c1 : memref<?x?xf32>
    %2 = arith.divui %dim_2, %c32 : index
    %3 = arith.muli %c32, %2 : index
    %4 = arith.subi %dim_2, %3 : index
    %5 = vector.splat %arg1 : vector<32xf32>
    %6 = vector.splat %arg2 : vector<32xf32>
    %7 = arith.subi %dim, %c1 : index
    %8 = arith.index_cast %7 : index to i32
    %9 = arith.sitofp %8 : i32 to f32
    %10 = arith.subi %dim_0, %c1 : index
    %11 = arith.index_cast %10 : index to i32
    %12 = arith.sitofp %11 : i32 to f32
    %13 = arith.subi %dim_1, %c1 : index
    %14 = arith.index_cast %13 : index to i32
    %15 = arith.sitofp %14 : i32 to f32
    %16 = arith.subi %dim_2, %c1 : index
    %17 = arith.index_cast %16 : index to i32
    %18 = arith.sitofp %17 : i32 to f32
    affine.for %arg4 = #map(%c0) to #map(%dim_1) {
      affine.for %arg5 = #map(%c0) to #map(%3) step 32 {
        %19 = arith.index_cast %arg4 : index to i32
        %20 = arith.sitofp %19 : i32 to f32
        %21 = vector.splat %20 : vector<32xf32>
        %alloc = memref.alloc() : memref<32xf32>
        affine.for %arg6 = #map(%c0) to #map(%c32) {
          %37 = arith.addi %arg6, %arg5 : index
          %38 = arith.index_cast %37 : index to i32
          %39 = arith.sitofp %38 : i32 to f32
          memref.store %39, %alloc[%arg6] : memref<32xf32>
        }
        %22 = vector.load %alloc[%c0] : memref<32xf32>, vector<32xf32>
        %23 = arith.mulf %22, %5 : vector<32xf32>
        %24 = arith.mulf %21, %6 : vector<32xf32>
        %25 = math.ceil %23 : vector<32xf32>
        %26 = math.floor %23 : vector<32xf32>
        %27 = arith.subf %25, %23 : vector<32xf32>
        %28 = arith.subf %23, %26 : vector<32xf32>
        %29 = arith.cmpf ogt, %27, %28 : vector<32xf32>
        %30 = arith.select %29, %26, %25 : vector<32xi1>, vector<32xf32>
        %31 = math.ceil %24 : vector<32xf32>
        %32 = math.floor %24 : vector<32xf32>
        %33 = arith.subf %31, %24 : vector<32xf32>
        %34 = arith.subf %24, %32 : vector<32xf32>
        %35 = arith.cmpf ogt, %33, %34 : vector<32xf32>
        %36 = arith.select %35, %32, %31 : vector<32xi1>, vector<32xf32>
        affine.for %arg6 = #map(%c0) to #map(%c32) {
          %37 = vector.extractelement %30[%arg6 : index] : vector<32xf32>
          %38 = vector.extractelement %36[%arg6 : index] : vector<32xf32>
          %39 = arith.maxf %37, %1 : f32
          %40 = arith.minf %39, %12 : f32
          %41 = arith.maxf %38, %1 : f32
          %42 = arith.minf %41, %9 : f32
          %43 = arith.fptoui %40 : f32 to i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.fptoui %42 : f32 to i32
          %46 = arith.index_cast %45 : i32 to index
          %47 = vector.extractelement %22[%arg6 : index] : vector<32xf32>
          %48 = vector.extractelement %21[%arg6 : index] : vector<32xf32>
          %49 = arith.maxf %47, %1 : f32
          %50 = arith.minf %49, %18 : f32
          %51 = arith.maxf %48, %1 : f32
          %52 = arith.minf %51, %15 : f32
          %53 = arith.fptoui %50 : f32 to i32
          %54 = arith.index_cast %53 : i32 to index
          %55 = arith.fptoui %52 : f32 to i32
          %56 = arith.index_cast %55 : i32 to index
          %57 = memref.load %arg0[%46, %44] : memref<?x?xf32>
          memref.store %57, %arg3[%56, %54] : memref<?x?xf32>
        }
      }
    }
    affine.for %arg4 = #map(%c0) to #map(%dim_1) {
      affine.for %arg5 = #map(%3) to #map(%dim_2) step 32 {
        %19 = arith.index_cast %arg4 : index to i32
        %20 = arith.sitofp %19 : i32 to f32
        %21 = vector.splat %20 : vector<32xf32>
        %alloc = memref.alloc() : memref<32xf32>
        affine.for %arg6 = #map(%c0) to #map(%4) {
          %37 = arith.addi %arg6, %arg5 : index
          %38 = arith.index_cast %37 : index to i32
          %39 = arith.sitofp %38 : i32 to f32
          memref.store %39, %alloc[%arg6] : memref<32xf32>
        }
        %22 = vector.load %alloc[%c0] : memref<32xf32>, vector<32xf32>
        %23 = arith.mulf %22, %5 : vector<32xf32>
        %24 = arith.mulf %21, %6 : vector<32xf32>
        %25 = math.ceil %23 : vector<32xf32>
        %26 = math.floor %23 : vector<32xf32>
        %27 = arith.subf %25, %23 : vector<32xf32>
        %28 = arith.subf %23, %26 : vector<32xf32>
        %29 = arith.cmpf ogt, %27, %28 : vector<32xf32>
        %30 = arith.select %29, %26, %25 : vector<32xi1>, vector<32xf32>
        %31 = math.ceil %24 : vector<32xf32>
        %32 = math.floor %24 : vector<32xf32>
        %33 = arith.subf %31, %24 : vector<32xf32>
        %34 = arith.subf %24, %32 : vector<32xf32>
        %35 = arith.cmpf ogt, %33, %34 : vector<32xf32>
        %36 = arith.select %35, %32, %31 : vector<32xi1>, vector<32xf32>
        affine.for %arg6 = #map(%c0) to #map(%4) {
          %37 = vector.extractelement %30[%arg6 : index] : vector<32xf32>
          %38 = vector.extractelement %36[%arg6 : index] : vector<32xf32>
          %39 = arith.maxf %37, %1 : f32
          %40 = arith.minf %39, %12 : f32
          %41 = arith.maxf %38, %1 : f32
          %42 = arith.minf %41, %9 : f32
          %43 = arith.fptoui %40 : f32 to i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.fptoui %42 : f32 to i32
          %46 = arith.index_cast %45 : i32 to index
          %47 = vector.extractelement %22[%arg6 : index] : vector<32xf32>
          %48 = vector.extractelement %21[%arg6 : index] : vector<32xf32>
          %49 = arith.maxf %47, %1 : f32
          %50 = arith.minf %49, %18 : f32
          %51 = arith.maxf %48, %1 : f32
          %52 = arith.minf %51, %15 : f32
          %53 = arith.fptoui %50 : f32 to i32
          %54 = arith.index_cast %53 : i32 to index
          %55 = arith.fptoui %52 : f32 to i32
          %56 = arith.index_cast %55 : i32 to index
          %57 = memref.load %arg0[%46, %44] : memref<?x?xf32>
          memref.store %57, %arg3[%56, %54] : memref<?x?xf32>
        }
      }
    }
    return
  }
  func.func @resize_4d_nearest_neighbour_interpolation(%arg0: memref<?x?x?x?xf32>, %arg1: f32, %arg2: f32, %arg3: memref<?x?x?x?xf32>) attributes {llvm.emit_c_interface} {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = arith.index_cast %c0 : index to i32
    %1 = arith.sitofp %0 : i32 to f32
    %dim = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %dim_0 = memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
    %dim_batch1 = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
    %dim_color1 = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>
    %dim_1 = memref.dim %arg3, %c1 : memref<?x?x?x?xf32>
    %dim_2 = memref.dim %arg3, %c2 : memref<?x?x?x?xf32>
    %dim_batch2 = memref.dim %arg3, %c0 : memref<?x?x?x?xf32>
    %dim_color2 = memref.dim %arg3, %c3 : memref<?x?x?x?xf32>
    %2 = arith.divui %dim_2, %c32 : index
    %3 = arith.muli %c32, %2 : index
    %4 = arith.subi %dim_2, %3 : index
    %5 = vector.splat %arg1 : vector<32xf32>
    %6 = vector.splat %arg2 : vector<32xf32>
    %7 = arith.subi %dim, %c1 : index
    %8 = arith.index_cast %7 : index to i32
    %9 = arith.sitofp %8 : i32 to f32
    %10 = arith.subi %dim_0, %c1 : index
    %11 = arith.index_cast %10 : index to i32
    %12 = arith.sitofp %11 : i32 to f32
    %13 = arith.subi %dim_1, %c1 : index
    %14 = arith.index_cast %13 : index to i32
    %15 = arith.sitofp %14 : i32 to f32
    %16 = arith.subi %dim_2, %c1 : index
    %17 = arith.index_cast %16 : index to i32
    %18 = arith.sitofp %17 : i32 to f32
    affine.for %batch = #map(%c0) to #map(%c1) {
      affine.for %color = #map(%c0) to #map(%3) {     
        affine.for %arg4 = #map(%c0) to #map(%dim_1) {
          affine.for %arg5 = #map(%c0) to #map(%3) step 32 {
            %19 = arith.index_cast %arg4 : index to i32
            %20 = arith.sitofp %19 : i32 to f32
            %21 = vector.splat %20 : vector<32xf32>
            %alloc = memref.alloc() : memref<32xf32>
            affine.for %arg6 = #map(%c0) to #map(%c32) {
              %37 = arith.addi %arg6, %arg5 : index
              %38 = arith.index_cast %37 : index to i32
              %39 = arith.sitofp %38 : i32 to f32
              memref.store %39, %alloc[%arg6] : memref<32xf32>
            }
            %22 = vector.load %alloc[%c0] : memref<32xf32>, vector<32xf32>
            %23 = arith.mulf %22, %5 : vector<32xf32>
            %24 = arith.mulf %21, %6 : vector<32xf32>
            %25 = math.ceil %23 : vector<32xf32>
            %26 = math.floor %23 : vector<32xf32>
            %27 = arith.subf %25, %23 : vector<32xf32>
            %28 = arith.subf %23, %26 : vector<32xf32>
            %29 = arith.cmpf ogt, %27, %28 : vector<32xf32>
            %30 = arith.select %29, %26, %25 : vector<32xi1>, vector<32xf32>
            %31 = math.ceil %24 : vector<32xf32>
            %32 = math.floor %24 : vector<32xf32>
            %33 = arith.subf %31, %24 : vector<32xf32>
            %34 = arith.subf %24, %32 : vector<32xf32>
            %35 = arith.cmpf ogt, %33, %34 : vector<32xf32>
            %36 = arith.select %35, %32, %31 : vector<32xi1>, vector<32xf32>
            affine.for %arg6 = #map(%c0) to #map(%c32) {
              %37 = vector.extractelement %30[%arg6 : index] : vector<32xf32>
              %38 = vector.extractelement %36[%arg6 : index] : vector<32xf32>
              %39 = arith.maxf %37, %1 : f32
              %40 = arith.minf %39, %12 : f32
              %41 = arith.maxf %38, %1 : f32
              %42 = arith.minf %41, %9 : f32
              %43 = arith.fptoui %40 : f32 to i32
              %44 = arith.index_cast %43 : i32 to index
              %45 = arith.fptoui %42 : f32 to i32
              %46 = arith.index_cast %45 : i32 to index
              %47 = vector.extractelement %22[%arg6 : index] : vector<32xf32>
              %48 = vector.extractelement %21[%arg6 : index] : vector<32xf32>
              %49 = arith.maxf %47, %1 : f32
              %50 = arith.minf %49, %18 : f32
              %51 = arith.maxf %48, %1 : f32
              %52 = arith.minf %51, %15 : f32
              %53 = arith.fptoui %50 : f32 to i32
              %54 = arith.index_cast %53 : i32 to index
              %55 = arith.fptoui %52 : f32 to i32
              %56 = arith.index_cast %55 : i32 to index
              %57 = memref.load %arg0[%batch, %46, %44, %color] : memref<?x?x?x?xf32>
              memref.store %57, %arg3[%batch, %56, %54, %color] : memref<?x?x?x?xf32>
            }
          }
        }
      }
    }
    affine.for %batch1 = #map(%c0) to #map(%c1) {
      affine.for %color1 = #map(%c0) to #map(%3) {   
        affine.for %arg4 = #map(%c0) to #map(%dim_1) {
          affine.for %arg5 = #map(%3) to #map(%dim_2) step 32 {
            %19 = arith.index_cast %arg4 : index to i32
            %20 = arith.sitofp %19 : i32 to f32
            %21 = vector.splat %20 : vector<32xf32>
            %alloc = memref.alloc() : memref<32xf32>
            affine.for %arg6 = #map(%c0) to #map(%4) {
              %37 = arith.addi %arg6, %arg5 : index
              %38 = arith.index_cast %37 : index to i32
              %39 = arith.sitofp %38 : i32 to f32
              memref.store %39, %alloc[%arg6] : memref<32xf32>
            }
            %22 = vector.load %alloc[%c0] : memref<32xf32>, vector<32xf32>
            %23 = arith.mulf %22, %5 : vector<32xf32>
            %24 = arith.mulf %21, %6 : vector<32xf32>
            %25 = math.ceil %23 : vector<32xf32>
            %26 = math.floor %23 : vector<32xf32>
            %27 = arith.subf %25, %23 : vector<32xf32>
            %28 = arith.subf %23, %26 : vector<32xf32>
            %29 = arith.cmpf ogt, %27, %28 : vector<32xf32>
            %30 = arith.select %29, %26, %25 : vector<32xi1>, vector<32xf32>
            %31 = math.ceil %24 : vector<32xf32>
            %32 = math.floor %24 : vector<32xf32>
            %33 = arith.subf %31, %24 : vector<32xf32>
            %34 = arith.subf %24, %32 : vector<32xf32>
            %35 = arith.cmpf ogt, %33, %34 : vector<32xf32>
            %36 = arith.select %35, %32, %31 : vector<32xi1>, vector<32xf32>
            affine.for %arg6 = #map(%c0) to #map(%4) {
              %37 = vector.extractelement %30[%arg6 : index] : vector<32xf32>
              %38 = vector.extractelement %36[%arg6 : index] : vector<32xf32>
              %39 = arith.maxf %37, %1 : f32
              %40 = arith.minf %39, %12 : f32
              %41 = arith.maxf %38, %1 : f32
              %42 = arith.minf %41, %9 : f32
              %43 = arith.fptoui %40 : f32 to i32
              %44 = arith.index_cast %43 : i32 to index
              %45 = arith.fptoui %42 : f32 to i32
              %46 = arith.index_cast %45 : i32 to index
              %47 = vector.extractelement %22[%arg6 : index] : vector<32xf32>
              %48 = vector.extractelement %21[%arg6 : index] : vector<32xf32>
              %49 = arith.maxf %47, %1 : f32
              %50 = arith.minf %49, %18 : f32
              %51 = arith.maxf %48, %1 : f32
              %52 = arith.minf %51, %15 : f32
              %53 = arith.fptoui %50 : f32 to i32
              %54 = arith.index_cast %53 : i32 to index
              %55 = arith.fptoui %52 : f32 to i32
              %56 = arith.index_cast %55 : i32 to index
              %57 = memref.load %arg0[%batch1, %46, %44, %color1] : memref<?x?x?x?xf32>
              memref.store %57, %arg3[%batch1, %56, %54, %color1] : memref<?x?x?x?xf32>
            }
          }
        }
      }
    }
    return
  }
  func.func @resize_2d_bilinear_interpolation(%arg0: memref<?x?xf32>, %arg1: f32, %arg2: f32, %arg3: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %c0 : index to i32
    %1 = arith.sitofp %0 : i32 to f32
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg3, %c0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg3, %c1 : memref<?x?xf32>
    %2 = arith.divui %dim_2, %c32 : index
    %3 = arith.muli %c32, %2 : index
    %4 = arith.subi %dim_2, %3 : index
    %5 = vector.splat %arg1 : vector<32xf32>
    %6 = vector.splat %arg2 : vector<32xf32>
    %7 = arith.subi %dim, %c1 : index
    %8 = arith.index_cast %7 : index to i32
    %9 = arith.sitofp %8 : i32 to f32
    %10 = arith.subi %dim_0, %c1 : index
    %11 = arith.index_cast %10 : index to i32
    %12 = arith.sitofp %11 : i32 to f32
    %13 = arith.subi %dim_1, %c1 : index
    %14 = arith.index_cast %13 : index to i32
    %15 = arith.sitofp %14 : i32 to f32
    %16 = arith.subi %dim_2, %c1 : index
    %17 = arith.index_cast %16 : index to i32
    %18 = arith.sitofp %17 : i32 to f32
    %19 = arith.index_cast %c1 : index to i32
    %20 = arith.sitofp %19 : i32 to f32
    affine.for %arg4 = #map(%c0) to #map(%dim_1) {
      affine.for %arg5 = #map(%c0) to #map(%3) step 32 {
        %21 = arith.index_cast %arg4 : index to i32
        %22 = arith.sitofp %21 : i32 to f32
        %23 = vector.splat %22 : vector<32xf32>
        %alloc = memref.alloc() : memref<32xf32>
        affine.for %arg6 = #map(%c0) to #map(%c32) {
          %33 = arith.addi %arg6, %arg5 : index
          %34 = arith.index_cast %33 : index to i32
          %35 = arith.sitofp %34 : i32 to f32
          memref.store %35, %alloc[%arg6] : memref<32xf32>
        }
        %24 = vector.load %alloc[%c0] : memref<32xf32>, vector<32xf32>
        %25 = arith.mulf %24, %5 : vector<32xf32>
        %26 = arith.mulf %23, %6 : vector<32xf32>
        %27 = math.floor %25 : vector<32xf32>
        %28 = math.ceil %25 : vector<32xf32>
        %29 = math.floor %26 : vector<32xf32>
        %30 = math.ceil %26 : vector<32xf32>
        %31 = arith.subf %25, %27 : vector<32xf32>
        %32 = arith.subf %26, %29 : vector<32xf32>
        affine.for %arg6 = #map(%c0) to #map(%c32) {
          %33 = vector.extractelement %24[%arg6 : index] : vector<32xf32>
          %34 = vector.extractelement %23[%arg6 : index] : vector<32xf32>
          %35 = arith.maxf %33, %1 : f32
          %36 = arith.minf %35, %18 : f32
          %37 = arith.maxf %34, %1 : f32
          %38 = arith.minf %37, %15 : f32
          %39 = arith.fptoui %36 : f32 to i32
          %40 = arith.index_cast %39 : i32 to index
          %41 = arith.fptoui %38 : f32 to i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = vector.extractelement %27[%arg6 : index] : vector<32xf32>
          %44 = vector.extractelement %29[%arg6 : index] : vector<32xf32>
          %45 = arith.maxf %43, %1 : f32
          %46 = arith.minf %45, %12 : f32
          %47 = arith.maxf %44, %1 : f32
          %48 = arith.minf %47, %9 : f32
          %49 = arith.fptoui %46 : f32 to i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = arith.fptoui %48 : f32 to i32
          %52 = arith.index_cast %51 : i32 to index
          %53 = vector.extractelement %28[%arg6 : index] : vector<32xf32>
          %54 = vector.extractelement %30[%arg6 : index] : vector<32xf32>
          %55 = arith.maxf %53, %1 : f32
          %56 = arith.minf %55, %12 : f32
          %57 = arith.maxf %54, %1 : f32
          %58 = arith.minf %57, %9 : f32
          %59 = arith.fptoui %56 : f32 to i32
          %60 = arith.index_cast %59 : i32 to index
          %61 = arith.fptoui %58 : f32 to i32
          %62 = arith.index_cast %61 : i32 to index
          %63 = vector.extractelement %31[%arg6 : index] : vector<32xf32>
          %64 = vector.extractelement %32[%arg6 : index] : vector<32xf32>
          %65 = arith.maxf %63, %1 : f32
          %66 = arith.minf %65, %12 : f32
          %67 = arith.maxf %64, %1 : f32
          %68 = arith.minf %67, %9 : f32
          %69 = arith.subf %20, %66 : f32
          %70 = arith.subf %20, %68 : f32
          %71 = memref.load %arg0[%52, %50] : memref<?x?xf32>
          %72 = memref.load %arg0[%62, %50] : memref<?x?xf32>
          %73 = memref.load %arg0[%52, %60] : memref<?x?xf32>
          %74 = memref.load %arg0[%62, %60] : memref<?x?xf32>
          %75 = arith.mulf %69, %70 : f32
          %76 = arith.mulf %66, %70 : f32
          %77 = arith.mulf %68, %69 : f32
          %78 = arith.mulf %66, %68 : f32
          %79 = arith.mulf %71, %75 : f32
          %80 = arith.mulf %72, %76 : f32
          %81 = arith.mulf %73, %77 : f32
          %82 = arith.mulf %74, %78 : f32
          %83 = arith.addf %79, %80 : f32
          %84 = arith.addf %81, %82 : f32
          %85 = arith.addf %83, %84 : f32
          %86 = math.ceil %85 : f32
          %87 = math.floor %85 : f32
          %88 = arith.subf %86, %85 : f32
          %89 = arith.subf %85, %87 : f32
          %90 = arith.cmpf ogt, %88, %89 : f32
          %91 = arith.select %90, %87, %86 : f32
          memref.store %91, %arg3[%42, %40] : memref<?x?xf32>
        }
      }
    }
    affine.for %arg4 = #map(%c0) to #map(%dim_1) {
      affine.for %arg5 = #map(%3) to #map(%dim_2) step 32 {
        %21 = arith.index_cast %arg4 : index to i32
        %22 = arith.sitofp %21 : i32 to f32
        %23 = vector.splat %22 : vector<32xf32>
        %alloc = memref.alloc() : memref<32xf32>
        affine.for %arg6 = #map(%c0) to #map(%4) {
          %33 = arith.addi %arg6, %arg5 : index
          %34 = arith.index_cast %33 : index to i32
          %35 = arith.sitofp %34 : i32 to f32
          memref.store %35, %alloc[%arg6] : memref<32xf32>
        }
        %24 = vector.load %alloc[%c0] : memref<32xf32>, vector<32xf32>
        %25 = arith.mulf %24, %5 : vector<32xf32>
        %26 = arith.mulf %23, %6 : vector<32xf32>
        %27 = math.floor %25 : vector<32xf32>
        %28 = math.ceil %25 : vector<32xf32>
        %29 = math.floor %26 : vector<32xf32>
        %30 = math.ceil %26 : vector<32xf32>
        %31 = arith.subf %25, %27 : vector<32xf32>
        %32 = arith.subf %26, %29 : vector<32xf32>
        affine.for %arg6 = #map(%c0) to #map(%4) {
          %33 = vector.extractelement %24[%arg6 : index] : vector<32xf32>
          %34 = vector.extractelement %23[%arg6 : index] : vector<32xf32>
          %35 = arith.maxf %33, %1 : f32
          %36 = arith.minf %35, %18 : f32
          %37 = arith.maxf %34, %1 : f32
          %38 = arith.minf %37, %15 : f32
          %39 = arith.fptoui %36 : f32 to i32
          %40 = arith.index_cast %39 : i32 to index
          %41 = arith.fptoui %38 : f32 to i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = vector.extractelement %27[%arg6 : index] : vector<32xf32>
          %44 = vector.extractelement %29[%arg6 : index] : vector<32xf32>
          %45 = arith.maxf %43, %1 : f32
          %46 = arith.minf %45, %12 : f32
          %47 = arith.maxf %44, %1 : f32
          %48 = arith.minf %47, %9 : f32
          %49 = arith.fptoui %46 : f32 to i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = arith.fptoui %48 : f32 to i32
          %52 = arith.index_cast %51 : i32 to index
          %53 = vector.extractelement %28[%arg6 : index] : vector<32xf32>
          %54 = vector.extractelement %30[%arg6 : index] : vector<32xf32>
          %55 = arith.maxf %53, %1 : f32
          %56 = arith.minf %55, %12 : f32
          %57 = arith.maxf %54, %1 : f32
          %58 = arith.minf %57, %9 : f32
          %59 = arith.fptoui %56 : f32 to i32
          %60 = arith.index_cast %59 : i32 to index
          %61 = arith.fptoui %58 : f32 to i32
          %62 = arith.index_cast %61 : i32 to index
          %63 = vector.extractelement %31[%arg6 : index] : vector<32xf32>
          %64 = vector.extractelement %32[%arg6 : index] : vector<32xf32>
          %65 = arith.maxf %63, %1 : f32
          %66 = arith.minf %65, %12 : f32
          %67 = arith.maxf %64, %1 : f32
          %68 = arith.minf %67, %9 : f32
          %69 = arith.subf %20, %66 : f32
          %70 = arith.subf %20, %68 : f32
          %71 = memref.load %arg0[%52, %50] : memref<?x?xf32>
          %72 = memref.load %arg0[%62, %50] : memref<?x?xf32>
          %73 = memref.load %arg0[%52, %60] : memref<?x?xf32>
          %74 = memref.load %arg0[%62, %60] : memref<?x?xf32>
          %75 = arith.mulf %69, %70 : f32
          %76 = arith.mulf %66, %70 : f32
          %77 = arith.mulf %68, %69 : f32
          %78 = arith.mulf %66, %68 : f32
          %79 = arith.mulf %71, %75 : f32
          %80 = arith.mulf %72, %76 : f32
          %81 = arith.mulf %73, %77 : f32
          %82 = arith.mulf %74, %78 : f32
          %83 = arith.addf %79, %80 : f32
          %84 = arith.addf %81, %82 : f32
          %85 = arith.addf %83, %84 : f32
          %86 = math.ceil %85 : f32
          %87 = math.floor %85 : f32
          %88 = arith.subf %86, %85 : f32
          %89 = arith.subf %85, %87 : f32
          %90 = arith.cmpf ogt, %88, %89 : f32
          %91 = arith.select %90, %87, %86 : f32
          memref.store %91, %arg3[%42, %40] : memref<?x?xf32>
        }
      }
    }
    return
  }
}

