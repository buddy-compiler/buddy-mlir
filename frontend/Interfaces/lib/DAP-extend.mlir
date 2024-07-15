func.func @buddy_whisperPreprocess(%in : memref<?xf64>, %out : memref<1x80x3000xf32>) -> () {
  dap.whisper_preprocess %in, %out : memref<?xf64>, memref<1x80x3000xf32>
  return
}
