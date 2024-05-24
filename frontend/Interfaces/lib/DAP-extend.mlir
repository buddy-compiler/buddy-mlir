func.func @buddy_WhisperPreprocess(%in : memref<?xf64>, %out : memref<1x80x3000xf32>) -> () {
  dap.WhisperPreprocess %in, %out : memref<?xf64>, memref<1x80x3000xf32>
  return
}
