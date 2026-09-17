module {
func.func @kernel_dequantize_1x2048(%x:memref<1x2048xi32>, %row:memref<1xf32>, %col:memref<2048xf32>, %y:memref<1x2048xf32>) attributes {llvm.emit_c_interface} {
 linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i)>,affine_map<(i,j)->(j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%x,%row,%col:memref<1x2048xi32>,memref<1xf32>,memref<2048xf32>) outs(%y:memref<1x2048xf32>) {
 ^bb0(%xv:i32,%rs:f32,%cs:f32,%unused:f32):
  %f = arith.sitofp %xv : i32 to f32
  %a = arith.mulf %f,%rs : f32
  %b = arith.mulf %a,%cs : f32
  linalg.yield %b : f32
 }
 return
}
}
