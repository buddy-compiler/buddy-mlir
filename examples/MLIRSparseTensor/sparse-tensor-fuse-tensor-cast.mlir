// RUN: buddy-opt %s \
// RUN: --pre-sparsification-rewrite

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ]
}>

#SortedCOO = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ]
}>

#Slice = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ],
  slice = [ (?, 1, 1), (?, 3, 1) ]
}>


/// FuseTensorCast: Nop Cast
///
/// If source type match the destination type, these type cast operations will be eliminated.
///
/// For example:
///
/// ```pseudo-mlir
/// // Before
/// %0 = tensor.cast %a : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
/// %1 = tensor.cast %0 : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
///
/// // After
/// (nop)
/// ```
func.func @sparse_nop_cast(%a : tensor<?xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %a : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  %1 = tensor.cast %0 : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  %2 = tensor.cast %1 : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %2 : tensor<?xf32, #SparseVector>
}

/// FuseTensorCast: Fuse tensor cast
///
/// If the source type and target type in tensor.cast are the same type when ignoring the attribute,
/// then the tensor cast is fused into the operation where the type is produced.
///
/// For example:
///
/// ```mlir
/// // Before
/// %extracted_slice = tensor.extract_slice %a[1, 0] [1, 3] [1, 1] : tensor<2x3xi64, #SortedCOO> to tensor<1x3xi64>
/// %cast = tensor.cast %extracted_slice : tensor<1x3xi64> to tensor<1x3xi64, #Slice>
/// 
/// // After
/// %extracted_slice = tensor.extract_slice %a[1, 0] [1, 3] [1, 1] : tensor<2x3xi64, #SortedCOO> to tensor<1x3xi64, #Slice>
/// ```
func.func @sparse_fuse_slice(%a : tensor<2x3xi64, #SortedCOO>) -> tensor<1x3xi64, #SortedCOO> {
  %extracted_slice = tensor.extract_slice %a[1, 0] [1, 3] [1, 1] : tensor<2x3xi64, #SortedCOO> to tensor<1x3xi64>
  %cast = tensor.cast %extracted_slice : tensor<1x3xi64> to tensor<1x3xi64, #Slice>
  %0 = sparse_tensor.convert %cast : tensor<1x3xi64, #Slice> to tensor<1x3xi64, #SortedCOO>
  return %0 : tensor<1x3xi64, #SortedCOO>
}


/// FuseTensorCast: Repair tensor cast
///
/// If any of the operand in the tensor.cast operation has sparse attribute, then this rewrite pattern
/// will replace it with sparse_tensor.convert operation.
///
/// The below example will be optimized into:
///
/// ```pseudo-mlir
/// func.func @sparse_repair_cast(%arg0: tensor<?xf32>) -> tensor<?xf32, #SparseVector> {
///   // NOTICE: tensor.cast is replaced with sparse_tensor.convert, because the return value has #SparseVector attr.
///   %0 = sparse_tensor.convert %arg0 : tensor<?xf32> to tensor<?xf32, #SparseVector>
///   return %0 : tensor<?xf32, #SparseVector>
/// }
/// ```
func.func @sparse_repair_cast(%a : tensor<?xf32>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %a : tensor<?xf32> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}
