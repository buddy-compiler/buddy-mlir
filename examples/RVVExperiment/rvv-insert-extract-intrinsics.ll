define <vscale x 4 x float> @insert_scalable_fixed_over(<vscale x 4 x float> %vec, <16 x float> %subvec) {
  %1 = call <vscale x 4 x float> @llvm.experimental.vector.insert.nxv4f32.v16f32(<vscale x 4 x float> %vec, <16 x float> %subvec, i64 0)
  ret <vscale x 4 x float> %1
}

declare <vscale x 4 x float> @llvm.experimental.vector.insert.nxv4f32.v16f32(<vscale x 4 x float> %vec, <16 x float> %subvec, i64 %idx)
