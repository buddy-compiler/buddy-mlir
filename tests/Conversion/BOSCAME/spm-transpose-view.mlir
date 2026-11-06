// RUN: buddy-opt %s --bosc-ame-spm-dma-check -o /dev/null
// RUN: buddy-opt %s --lower-bosc-ame -o /dev/null

// Transpose changes a descriptor, with no implicit SPM copy or scalar access.
// Fill a physical row of a column-major B window through its row-major view.
module {
  func.func @column_padding() {
    %c0 = arith.constant 0 : index
    %minus = arith.constant -2 : i8
    %vector = vector.broadcast %minus : i8 to vector<64xi8>
    %storage = bosc_ame.spm.view {bank = 1 : i32, offset = 64 : i64}
      : memref<16384xi8, #bosc_ame.spm<1>>
    %columns = memref.reinterpret_cast %storage to
      offset: [0], sizes: [256, 64], strides: [1, 256]
      : memref<16384xi8, #bosc_ame.spm<1>>
      to memref<256x64xi8, strided<[1, 256]>, #bosc_ame.spm<1>>
    %rows = memref.transpose %columns (d0, d1) -> (d1, d0)
      : memref<256x64xi8, strided<[1, 256]>, #bosc_ame.spm<1>>
      to memref<64x256xi8, strided<[256, 1]>, #bosc_ame.spm<1>>
    vector.store %vector, %rows[%c0, %c0]
      : memref<64x256xi8, strided<[256, 1]>, #bosc_ame.spm<1>>, vector<64xi8>
    return
  }
}
