// RUN: buddy-opt -conv-nhwc-fhwc-optimize --canonicalize --split-input-file %s | FileCheck %s

func.func @conv_2d_nhwc_fhwc_dynamic(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
  linalg.conv_2d_nhwc_fhwc ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) outs(%arg2 : memref<?x?x?x?xf32>)
  func.return
}
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:   func.func @conv_2d_nhwc_fhwc_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?x?x?x?xf32>, %[[ARG1:.*]]: memref<?x?x?x?xf32>, %[[ARG2:.*]]: memref<?x?x?x?xf32>) {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 3 : index
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 16 : index
// CHECK-DAG:           %[[VAL_6:.*]] = memref.dim %[[ARG2]], %[[VAL_3]] : memref<?x?x?x?xf32>
// CHECK-DAG:           %[[VAL_7:.*]] = memref.dim %[[ARG2]], %[[VAL_4]] : memref<?x?x?x?xf32>
// CHECK-DAG:           %[[VAL_8:.*]] = memref.dim %[[ARG2]], %[[VAL_1]] : memref<?x?x?x?xf32>
// CHECK-DAG:           %[[VAL_9:.*]] = memref.dim %[[ARG2]], %[[VAL_0]] : memref<?x?x?x?xf32>
// CHECK-DAG:           %[[VAL_10:.*]] = memref.dim %[[ARG1]], %[[VAL_4]] : memref<?x?x?x?xf32>
// CHECK-DAG:           %[[VAL_11:.*]] = memref.dim %[[ARG1]], %[[VAL_1]] : memref<?x?x?x?xf32>
// CHECK-DAG:           %[[VAL_12:.*]] = memref.dim %[[ARG1]], %[[VAL_0]] : memref<?x?x?x?xf32>
// CHECK:           scf.forall (%[[VAL_13:.*]], %[[VAL_14:.*]], %[[VAL_15:.*]], %[[VAL_16:.*]]) in (%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]]) {
// CHECK:             %[[VAL_17:.*]] = scf.for %[[VAL_18:.*]] = %[[VAL_3]] to %[[VAL_10]] step %[[VAL_4]] iter_args(%[[VAL_19:.*]] = %[[VAL_2]]) -> (vector<16xf32>) {
// CHECK:               %[[VAL_20:.*]] = scf.for %[[VAL_21:.*]] = %[[VAL_3]] to %[[VAL_11]] step %[[VAL_4]] iter_args(%[[VAL_22:.*]] = %[[VAL_19]]) -> (vector<16xf32>) {
// CHECK:                 %[[VAL_23:.*]] = scf.for %[[VAL_24:.*]] = %[[VAL_3]] to %[[VAL_12]] step %[[VAL_5]] iter_args(%[[VAL_25:.*]] = %[[VAL_22]]) -> (vector<16xf32>) {
// CHECK:                   %[[VAL_26:.*]] = affine.min #[[$ATTR_0]](%[[VAL_24]]){{\[}}%[[VAL_12]]]
// CHECK:                   %[[VAL_27:.*]] = vector.create_mask %[[VAL_26]] : vector<16xi1>
// CHECK-DAG:                   %[[VAL_28:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_14]], %[[VAL_18]])
// CHECK-DAG:                   %[[VAL_29:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_15]], %[[VAL_21]])
// CHECK:                   %[[VAL_30:.*]] = vector.maskedload %[[ARG0]]{{\[}}%[[VAL_13]], %[[VAL_28]], %[[VAL_29]], %[[VAL_24]]], %[[VAL_27]], %[[VAL_2]] : memref<?x?x?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK:                   %[[VAL_31:.*]] = vector.maskedload %[[ARG1]]{{\[}}%[[VAL_16]], %[[VAL_18]], %[[VAL_21]], %[[VAL_24]]], %[[VAL_27]], %[[VAL_2]] : memref<?x?x?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK:                   %[[VAL_32:.*]] = vector.fma %[[VAL_30]], %[[VAL_31]], %[[VAL_25]] : vector<16xf32>
// CHECK:                   scf.yield %[[VAL_32]] : vector<16xf32>
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_23]] : vector<16xf32>
// CHECK:               }
// CHECK:               scf.yield %[[VAL_20]] : vector<16xf32>
// CHECK:             }
// CHECK:             %[[VAL_33:.*]] = vector.reduction <add>, %[[VAL_17]] : vector<16xf32> into f32
// CHECK:             %[[VAL_34:.*]] = memref.load %[[ARG2]]{{\[}}%[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]]] : memref<?x?x?x?xf32>
// CHECK:             %[[VAL_35:.*]] = arith.addf %[[VAL_34]], %[[VAL_33]] : f32
// CHECK:             memref.store %[[VAL_35]], %[[ARG2]]{{\[}}%[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]]] : memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----
func.func @conv_2d_nhwc_fhwc_static(%arg0: memref<4x6x6x31xf32>, %arg1: memref<33x2x2x31xf32>, %arg2: memref<4x5x5x33xf32>) {
  linalg.conv_2d_nhwc_fhwc ins(%arg0, %arg1 : memref<4x6x6x31xf32>, memref<33x2x2x31xf32>) outs(%arg2 : memref<4x5x5x33xf32>)
  func.return
}
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (-d0 + 31, 16)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:   func.func @conv_2d_nhwc_fhwc_static(
// CHECK-SAME:      %[[ARG0:.*]]: memref<4x6x6x31xf32>, %[[ARG1:.*]]: memref<33x2x2x31xf32>, %[[ARG2:.*]]: memref<4x5x5x33xf32>) {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 31 : index
// CHECK:           scf.forall (%[[VAL_6:.*]], %[[VAL_7:.*]], %[[VAL_8:.*]], %[[VAL_9:.*]]) in (4, 5, 5, 33) {
// CHECK:             %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_1]] to %[[VAL_4]] step %[[VAL_2]] iter_args(%[[VAL_12:.*]] = %[[VAL_0]]) -> (vector<16xf32>) {
// CHECK:               %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_1]] to %[[VAL_4]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (vector<16xf32>) {
// CHECK:                 %[[VAL_16:.*]] = scf.for %[[VAL_17:.*]] = %[[VAL_1]] to %[[VAL_5]] step %[[VAL_3]] iter_args(%[[VAL_18:.*]] = %[[VAL_15]]) -> (vector<16xf32>) {
// CHECK:                   %[[VAL_19:.*]] = affine.min #[[$ATTR_0]](%[[VAL_17]])
// CHECK:                   %[[VAL_20:.*]] = vector.create_mask %[[VAL_19]] : vector<16xi1>
// CHECK-DAG:                   %[[VAL_21:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]], %[[VAL_11]])
// CHECK-DAG:                   %[[VAL_22:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_8]], %[[VAL_14]])
// CHECK:                   %[[VAL_23:.*]] = vector.maskedload %[[ARG0]]{{\[}}%[[VAL_6]], %[[VAL_21]], %[[VAL_22]], %[[VAL_17]]], %[[VAL_20]], %[[VAL_0]] : memref<4x6x6x31xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK:                   %[[VAL_24:.*]] = vector.maskedload %[[ARG1]]{{\[}}%[[VAL_9]], %[[VAL_11]], %[[VAL_14]], %[[VAL_17]]], %[[VAL_20]], %[[VAL_0]] : memref<33x2x2x31xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK:                   %[[VAL_25:.*]] = vector.fma %[[VAL_23]], %[[VAL_24]], %[[VAL_18]] : vector<16xf32>
// CHECK:                   scf.yield %[[VAL_25]] : vector<16xf32>
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_16]] : vector<16xf32>
// CHECK:               }
// CHECK:               scf.yield %[[VAL_13]] : vector<16xf32>
// CHECK:             }
// CHECK:             %[[VAL_26:.*]] = vector.reduction <add>, %[[VAL_10]] : vector<16xf32> into f32
// CHECK:             %[[VAL_27:.*]] = memref.load %[[ARG2]]{{\[}}%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]]] : memref<4x5x5x33xf32>
// CHECK:             %[[VAL_28:.*]] = arith.addf %[[VAL_27]], %[[VAL_26]] : f32
// CHECK:             memref.store %[[VAL_28]], %[[ARG2]]{{\[}}%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]]] : memref<4x5x5x33xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
