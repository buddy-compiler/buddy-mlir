module {
func.func @kernel_quantize_1x2048(%x: memref<1x2048xf32>, %q: memref<1x2048xi8>, %scale: memref<1xf32>) attributes {llvm.emit_c_interface} {
 %z = arith.constant 0.0 : f32
 %one = arith.constant 1.0 : f32
 %lim = arith.constant 127.0 : f32
 %neg = arith.constant -127.0 : f32
 %half = arith.constant 0.5 : f32
 %nhalf = arith.constant -0.5 : f32
 linalg.fill ins(%z : f32) outs(%scale : memref<1xf32>)
 linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i)>], iterator_types=["parallel","reduction"]} ins(%x:memref<1x2048xf32>) outs(%scale:memref<1xf32>) {
 ^bb0(%v:f32,%a:f32):
  %abs = math.absf %v : f32
  %max = arith.maximumf %a,%abs : f32
  linalg.yield %max : f32
 }
 linalg.generic {indexing_maps=[affine_map<(i)->(i)>],iterator_types=["parallel"]} outs(%scale:memref<1xf32>) {
 ^bb0(%v:f32):
  %zero = arith.cmpf oeq,%v,%z : f32
  %s = arith.divf %v,%lim : f32
  %safe = arith.select %zero,%one,%s : f32
  linalg.yield %safe : f32
 }
 linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%x,%scale:memref<1x2048xf32>,memref<1xf32>) outs(%q:memref<1x2048xi8>) {
 ^bb0(%v:f32,%s:f32,%unused:i8):
  %d = arith.divf %v,%s : f32
  %positive = arith.cmpf oge,%d,%z : f32
  %offset = arith.select %positive,%half,%nhalf : f32
  %rounded = arith.addf %d,%offset : f32
  %lo = arith.maximumf %rounded,%neg : f32
  %hi = arith.minimumf %lo,%lim : f32
  %i = arith.fptosi %hi : f32 to i32
  %byte = arith.trunci %i : i32 to i8
  linalg.yield %byte : i8
 }
 return
}
}
