// RUN: buddy-opt %s --verify-diagnostics --split-input-file

module {
  func.func @bad_output_width(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq: memref<1x1x64x1024xi8>, %ws: memref<1x63xf32>,
      %output: memref<1x63xf32>) {
    // expected-error@+1 {{D must be divisible by OUTBLK (64)}}
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<1x1x64x1024xi8>, memref<1x63xf32>,
         memref<1x63xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_layout(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq: memref<1x1x64x1024xi8>, %ws: memref<1x64xf32>,
      %output: memref<1x64xf32>) {
    // expected-error@+1 {{weight_layout must be 'ame_outblk64'}}
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 1024 : i64, weight_layout = "row_major"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<1x1x64x1024xi8>, memref<1x64xf32>,
         memref<1x64xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_group_divisibility(
      %xq: memref<1x1024xi8>, %xs: memref<1x4xf32>,
      %wq: memref<1x4x64x300xi8>, %ws: memref<4x64xf32>,
      %output: memref<1x64xf32>) {
    // expected-error@+1 {{K must be divisible by group_size}}
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 300 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x4xf32>,
         memref<1x4x64x300xi8>, memref<4x64xf32>,
         memref<1x64xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_k64(
      %xq: memref<1x1000xi8>, %xs: memref<1x4xf32>,
      %wq: memref<1x4x64x250xi8>, %ws: memref<4x64xf32>,
      %output: memref<1x64xf32>) {
    // expected-error@+1 {{K must be divisible by the AME K tile (64)}}
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 250 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1000xi8>, memref<1x4xf32>,
         memref<1x4x64x250xi8>, memref<4x64xf32>,
         memref<1x64xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @group_too_large(
      %xq: memref<1x2048xi8>, %xs: memref<1x1xf32>,
      %wq: memref<1x1x64x2048xi8>, %ws: memref<1x64xf32>,
      %output: memref<1x64xf32>) {
    // expected-error@+1 {{group_size must be in [1, 1024]}}
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 2048 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x2048xi8>, memref<1x1xf32>,
         memref<1x1x64x2048xi8>, memref<1x64xf32>,
         memref<1x64xf32>) -> ()
    return
  }
}
