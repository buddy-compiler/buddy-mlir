// RUN: buddy-opt %s \
// RUN:     --lower-gemmini | \
// RUN: FileCheck %s

// n x h x w x c
// batchSize = 1 inputDim = 5 inChannels = 2
memref.global "private" @input : memref<1x5x5x2xi8> = dense<[[[[1, 2], [0, 0], [-1, -1], [0, 0], [1, 1]],
                                                              [[1, 2], [0, 0], [-1, -1], [0, 0], [1, 1]],
                                                              [[1, 2], [0, 0], [-1, -1], [0, 0], [1, 1]],
                                                              [[1, 2], [0, 0], [-1, -1], [0, 0], [1, 1]],
                                                              [[1, 2], [0, 0], [-1, -1], [0, 0], [1, 1]]]]>

// chw x f
// outChannels = 2 kernelDim = 3 inChannels = 2
memref.global "private" @weight : memref<9x1xi8> = dense<[[1], [1], [1],
                                                          [1], [1], [1],
                                                          [1], [1], [1]]>

// outChannels = 1
memref.global "private" @bias : memref<1xi32> = dense<[1]>

func.func @main() -> i64 {
  %0 = arith.constant 0 : i64
  %3 = arith.constant 3 : i64
  %input = memref.get_global @input : memref<1x5x5x2xi8>
  %weight = memref.get_global @weight : memref<9x1xi8>
  %bias = memref.get_global @bias : memref<1xi32>
  %output0 = memref.alloc() : memref<9x1xi8>
  %output1 = memref.alloc() : memref<9x1xi8>
  %output = memref.alloc() : memref<9x2xi8>

  %subview0 = memref.alloc() : memref<1x5x5x1xi8>
  %subview1 = memref.alloc() : memref<1x5x5x1xi8>

  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3*2)>, 
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> 
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  }
  ins (%input : memref<1x5x5x2xi8>)
  outs (%subview0 : memref<1x5x5x1xi8>) {
    ^bb0(%in : i8, %out : i8):
      linalg.yield %in : i8
  }

  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3*2 + 1)>, 
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> 
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  }
  ins (%input : memref<1x5x5x2xi8>)
  outs (%subview1 : memref<1x5x5x1xi8>) {
    ^bb0(%in : i8, %out : i8):
      linalg.yield %in : i8
  }

  //gemmini.print %subview0 : memref<1x5x5x1xi8>
  //gemmini.print %subview1 : memref<1x5x5x1xi8>
  
  gemmini.tile_conv %subview0 %weight %bias %output0 %3 %3 %3 {stride = 1}:
     memref<1x5x5x1xi8> memref<9x1xi8> memref<1xi32> memref<9x1xi8> i64 i64 i64
  // gemmini.print %output0 : memref<9x1xi8>

  gemmini.tile_conv %subview1 %weight %bias %output1 %3 %3 %3 {stride = 1}:
     memref<1x5x5x1xi8> memref<9x1xi8> memref<1xi32> memref<9x1xi8> i64 i64 i64
  // gemmini.print %output1 : memref<9x1xi8>

  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,        
      affine_map<(d0, d1) -> (d0, d1 * 2)>     
    ],
    iterator_types = ["parallel", "parallel"]
  }
  ins(%output0 : memref<9x1xi8>) 
  outs(%output : memref<9x2xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
  }

  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,        
      affine_map<(d0, d1) -> (d0, d1 * 2 + 1)>
    ],
    iterator_types = ["parallel", "parallel"]
  }
  ins(%output1 : memref<9x1xi8>) 
  outs(%output : memref<9x2xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
  }

  gemmini.print %output : memref<9x2xi8>

  return %0 : i64
}
