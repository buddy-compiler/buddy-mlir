func.func @buddy_whisperPreprocess(%in : memref<?xf64>) -> memref<1x80x3000xf32> {
  %out = dap.whisper_preprocess %in : memref<?xf64> to memref<1x80x3000xf32>
  return %out : memref<1x80x3000xf32>
}
