// RUN: buddy-opt %s -lower-ime="target=k3" -verify-diagnostics

func.func @oversized(%c: memref<9x4xi32>, %a: memref<9x8xi8>, %b: memref<4x8xi8>) {
  // expected-error @+2 {{K3 IME expects C[M,N], A[M,K], packed B[N,K]}}
  // expected-error @+1 {{failed to legalize operation 'ime.vmadot'}}
  ime.vmadot %c, %a, %b : memref<9x4xi32>, memref<9x8xi8>, memref<4x8xi8>
  return
}
