func.func @testloopop(%a : memref<10xf32>) -> () {
  %i64Value = arith.constant 1 : i64
  %i32Value = arith.constant 128 : i32
  %idxValue = arith.constant 8 : index

  acc.loop gang worker vector {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(num=%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(static=%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop worker(%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop worker(%i32Value: i32) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop worker(%idxValue: index) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop vector(%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop vector(%i32Value: i32) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop vector(%idxValue: index) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(num=%i64Value: i64) worker vector {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(num=%i64Value: i64, static=%i64Value: i64) worker(%i64Value: i64) vector(%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(num=%i32Value: i32, static=%idxValue: index) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop tile(%i64Value, %i64Value : i64, i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop tile(%i32Value, %i32Value : i32, i32) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(static=%i64Value: i64, num=%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(dim=%i64Value : i64, static=%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  %b = acc.cache varPtr(%a : memref<10xf32>) -> memref<10xf32>
  acc.loop cache(%b : memref<10xf32>) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  return
}

// CHECK:      [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK-NEXT: [[I32VALUE:%.*]] = arith.constant 128 : i32
// CHECK-NEXT: [[IDXVALUE:%.*]] = arith.constant 8 : index
// CHECK:      acc.loop gang worker vector {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(static=[[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop worker([[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop worker([[I32VALUE]] : i32) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop worker([[IDXVALUE]] : index) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop vector([[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop vector([[I32VALUE]] : i32) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop vector([[IDXVALUE]] : index) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I64VALUE]] : i64) worker vector {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I64VALUE]] : i64, static=[[I64VALUE]] : i64) worker([[I64VALUE]] : i64) vector([[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I32VALUE]] : i32, static=[[IDXVALUE]] : index) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop tile([[I64VALUE]], [[I64VALUE]] : i64, i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop tile([[I32VALUE]], [[I32VALUE]] : i32, i32) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I64VALUE]] : i64, static=[[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(dim=[[I64VALUE]] : i64, static=[[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      %{{.*}} = acc.cache varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT: acc.loop cache(%{{.*}} : memref<10xf32>) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }