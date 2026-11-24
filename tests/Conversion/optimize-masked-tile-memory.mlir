// RUN: buddy-opt %s --vectorize-masked-stores --lower-strided-copies --stack-temporaries | FileCheck %s

#id = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @guarded_store
// CHECK: arith.cmpi sle
// CHECK: scf.if
// CHECK: vector.maskedstore
// CHECK: } else {
// CHECK: memref.store
func.func @guarded_store(%base: i32, %output: memref<?xi32>, %values: tensor<13xi32>, %mask: tensor<13xi1>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %length = arith.constant 13 : index
  %empty = tensor.empty() : tensor<13xi32>
  %iota = linalg.generic {indexing_maps = [#id], iterator_types = ["parallel"]} outs(%empty : tensor<13xi32>) {
  ^bb0(%unused: i32):
    %pos = linalg.index 0 : index
    %pos32 = arith.index_cast %pos : index to i32
    linalg.yield %pos32 : i32
  } -> tensor<13xi32>
  %bases = linalg.fill ins(%base : i32) outs(%empty : tensor<13xi32>) -> tensor<13xi32>
  %offsets = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]} ins(%iota, %bases : tensor<13xi32>, tensor<13xi32>) outs(%empty : tensor<13xi32>) {
  ^bb0(%pos: i32, %start: i32, %unused: i32):
    %sum = arith.addi %start, %pos : i32
    linalg.yield %sum : i32
  } -> tensor<13xi32>
  scf.for %pos = %zero to %length step %one {
    %offset = tensor.extract %offsets[%pos] : tensor<13xi32>
    %index = arith.index_cast %offset : i32 to index
    %val = tensor.extract %values[%pos] : tensor<13xi32>
    %pred = tensor.extract %mask[%pos] : tensor<13xi1>
    scf.if %pred {
      memref.store %val, %output[%index] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func.func @scatter_dynamic
// CHECK-NOT: vector.maskedstore
// CHECK: memref.store
func.func @scatter_dynamic(%output: memref<?x?xi32>, %indices: tensor<?xindex>, %mask: tensor<?xi1>, %value: i32, %size: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %pos = %zero to %size step %one {
    %offset = tensor.extract %indices[%pos] : tensor<?xindex>
    %pred = tensor.extract %mask[%pos] : tensor<?xi1>
    scf.if %pred {
      memref.store %value, %output[%offset, %pos] : memref<?x?xi32>
    }
  }
  return
}

// CHECK-LABEL: func.func @copy_rows
// CHECK: memref.dim
// CHECK: vector.load
// CHECK: vector.store
// CHECK-NOT: memref.copy
func.func @copy_rows(%input: memref<?x?xf32, strided<[64, 1], offset: ?>>, %rows: index, %cols: index) -> memref<?x?xf32> {
  %local = memref.alloc(%rows, %cols) : memref<?x?xf32>
  memref.copy %input, %local : memref<?x?xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32>
  return %local : memref<?x?xf32>
}

// CHECK-LABEL: func.func @copy_alias_unknown
// CHECK: memref.copy
func.func @copy_alias_unknown(%input: memref<32xi32, strided<[2], offset: ?>>, %output: memref<32xi32>) {
  memref.copy %input, %output : memref<32xi32, strided<[2], offset: ?>> to memref<32xi32>
  return
}

// CHECK-LABEL: func.func @copy_columns
// CHECK: memref.transpose
// CHECK: memref.transpose
// CHECK: vector.load
// CHECK: vector.store
// CHECK-NOT: memref.copy
func.func @copy_columns(%input: memref<16x8xi32, strided<[1, 24]>>) {
  %local = memref.alloc() : memref<8x16xi32>
  %col_major = memref.transpose %local (d0, d1) -> (d1, d0) : memref<8x16xi32> to memref<16x8xi32, strided<[1, 16]>>
  memref.copy %input, %col_major : memref<16x8xi32, strided<[1, 24]>> to memref<16x8xi32, strided<[1, 16]>>
  memref.dealloc %local : memref<8x16xi32>
  return
}

// CHECK-LABEL: func.func @stack_reuse
// CHECK: %[[SLOT:.*]] = memref.alloca() : memref<8x8xi32>
// CHECK-NOT: memref.alloc
// CHECK: memref.load %[[SLOT]]
// CHECK: memref.load %[[SLOT]]
// CHECK-NOT: memref.dealloc
func.func @stack_reuse(%x: i32, %pos: index) -> i32 {
  %first = memref.alloc() : memref<8x8xi32>
  memref.store %x, %first[%pos, %pos] : memref<8x8xi32>
  %val = memref.load %first[%pos, %pos] : memref<8x8xi32>
  memref.dealloc %first : memref<8x8xi32>
  %second = memref.alloc() : memref<8x8xi32>
  memref.store %val, %second[%pos, %pos] : memref<8x8xi32>
  %result = memref.load %second[%pos, %pos] : memref<8x8xi32>
  memref.dealloc %second : memref<8x8xi32>
  return %result : i32
}

// CHECK-LABEL: func.func @stack_escape
// CHECK: memref.alloc()
// CHECK: memref.alloc(%arg0)
func.func @stack_escape(%size: index) -> (memref<8xf32>, memref<?xf32>) {
  %escaped = memref.alloc() : memref<8xf32>
  %dynamic = memref.alloc(%size) : memref<?xf32>
  return %escaped, %dynamic : memref<8xf32>, memref<?xf32>
}
