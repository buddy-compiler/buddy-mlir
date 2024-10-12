func.func @buddy_whisperPreprocess(%in : memref<?xf64>) -> memref<1x80x3000xf32> {
  %out = dap.whisper_preprocess %in : memref<?xf64> to memref<1x80x3000xf32>
  return %out : memref<1x80x3000xf32>
}
func.func @buddy_whisperPreprocessRFFT(%in : memref<?xf64>) -> () {
  dap.rfft %in : memref<?xf64>
  // return %in : memref<400xf64>
  return
}
