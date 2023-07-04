#SV = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ]
}>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // A
    affine_map<(i) -> (i)>,  // B
    affine_map<(i) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = A(i) + B(i)"
}

// The first example usage of the `sparse_tensor.binary` operation. This function will return a intersection of the two given sparse vector.
func.func @intersection(%sva: tensor<?xf64, #SV>, %svb: tensor<?xf64, #SV>) -> tensor<?xf64, #SV> {
  %size = arith.constant 64 : index
  %shape = bufferization.alloc_tensor(%size) : tensor<?xf64, #SV>
  %c0 = arith.constant 0.0 : f64

  %0 = linalg.generic #trait
  ins (%sva, %svb: tensor<?xf64, #SV>,
        tensor<?xf64, #SV>)
  outs(%shape: tensor<?xf64, #SV>) {
    ^bb0(%a: f64, %b: f64, %c: f64):
      // The main logic stay here.
      // The `overlap` block is evaluate when non-zero entry present in both input sparse tensor.
      // The empty block in `left` and `right` means do nothing when element present only in one sparse tensor.
      %result = sparse_tensor.binary %a, %b : f64, f64 to f64
        overlap={
          ^bb0(%arg0: f64, %arg1: f64):
            %cmp = arith.cmpf "oeq", %arg0, %arg1 : f64
            %ret = scf.if %cmp -> f64 {
              scf.yield %arg0 : f64
            } else {
              // when the value is not equal, yield zero back
              scf.yield %c0 : f64
            }
            sparse_tensor.yield %ret : f64
        }
        left={}
        right={}
    linalg.yield %result : f64
  } -> tensor<?xf64, #SV>

  return %0 : tensor<?xf64, #SV>
}

// This function use the `@intersection` function to get intersection of two sparse vector.
func.func @intersec_example() {
  %c0 = arith.constant 0 : index
  %fpad = arith.constant -1.0 : f64
  %ind_pad = arith.constant -1 : index

  // Create the first sparse vector, with value on index 0, 4, 16, 64, 256, 1023
  %cv0 = arith.constant sparse<
    [ [0], [4], [16], [64], [256], [1023] ],
    [ 1.1, 2.2, 3.3,   4.4,  5.5,   6.6 ]
    > : tensor<1024xf64>
  %sv0 = sparse_tensor.convert %cv0 : tensor<1024xf64> to tensor<?xf64, #SV>

  // Create the second sparse vector, with value on index 1, 4, 8, 64, 512, 1023
  %cv1 = arith.constant sparse<
    [ [1], [4], [8], [64], [512], [1023] ],
    [ 1.1, 2.2, 3.3, 4.4,   5.5,   6.6 ]
    > : tensor<1024xf64>
  %sv1 = sparse_tensor.convert %cv1 : tensor<1024xf64> to tensor<?xf64, #SV>

  // Get the intersection sparse vector from the two sparse vectors
  %it0 = call @intersection(%sv0, %sv1)
    : (tensor<?xf64, #SV>, tensor<?xf64, #SV>) -> tensor<?xf64, #SV>

  // Print the value, this should be (2.2, 4.4, 6.6)
  %val0 = sparse_tensor.values %it0 : tensor<?xf64, #SV> to memref<?xf64>
  %v0 = vector.transfer_read %val0[%c0], %fpad : memref<?xf64>, vector<3xf64>
  vector.print %v0 : vector<3xf64>

  // Print the intersection index, this should be (4, 64, 1023)
  %ind0 = sparse_tensor.coordinates %it0 { level = 0 : index }
    : tensor<?xf64, #SV> to memref<?xindex>
  %v1 = vector.transfer_read %ind0[%c0], %ind_pad : memref<?xindex>, vector<3xindex>
  vector.print %v1 : vector<3xindex>

  return
}

func.func @main() {
  call @intersec_example() : () -> ()

  return
}
