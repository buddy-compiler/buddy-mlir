func.func @main() {
  %1 = memref_exp.null : memref<4x4xf32>
  %2 = memref.extract_aligned_pointer_as_index %1 : memref<4x4xf32> -> index
  vector.print %2 : index
  return
}
