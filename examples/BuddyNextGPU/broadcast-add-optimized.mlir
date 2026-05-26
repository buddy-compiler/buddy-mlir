// Optimized: nest 4D parallel into 3D(block) + 1D(thread)
// blocks=(2, 6, 1024) threads=(128, 1, 1) — all 128 hidden elements parallel
func.func @broadcast_add_opt(%A: memref<2x1x1024x128xf32>,
                              %B: memref<2x6x1024x128xf32>,
                              %C: memref<2x6x1024x128xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index
  %c1024 = arith.constant 1024 : index
  %c128 = arith.constant 128 : index

  // Outer 3D → blocks
  scf.parallel (%head, %group, %seq) = (%c0, %c0, %c0) to (%c2, %c6, %c1024) step (%c1, %c1, %c1) {
    // Inner 1D → threads
    scf.parallel (%hid) = (%c0) to (%c128) step (%c1) {
      %a = memref.load %A[%head, %c0, %seq, %hid] : memref<2x1x1024x128xf32>
      %b = memref.load %B[%head, %group, %seq, %hid] : memref<2x6x1024x128xf32>
      %sum = arith.addf %a, %b : f32
      memref.store %sum, %C[%head, %group, %seq, %hid] : memref<2x6x1024x128xf32>
      scf.reduce
    }
    scf.reduce
  }
  return
}
