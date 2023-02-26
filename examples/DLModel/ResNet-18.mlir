module {
  func.func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %1 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %2 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %3 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1000x512xf32>} : () -> tensor<1000x512xf32>
    %4 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x512x3x3xf32>} : () -> tensor<512x512x3x3xf32>
    %5 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x512x3x3xf32>} : () -> tensor<512x512x3x3xf32>
    %6 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x256x1x1xf32>} : () -> tensor<512x256x1x1xf32>
    %7 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x512x3x3xf32>} : () -> tensor<512x512x3x3xf32>
    %8 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x256x3x3xf32>} : () -> tensor<512x256x3x3xf32>
    %9 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x256x3x3xf32>} : () -> tensor<256x256x3x3xf32>
    %10 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x256x3x3xf32>} : () -> tensor<256x256x3x3xf32>
    %11 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x128x1x1xf32>} : () -> tensor<256x128x1x1xf32>
    %12 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x256x3x3xf32>} : () -> tensor<256x256x3x3xf32>
    %13 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x128x3x3xf32>} : () -> tensor<256x128x3x3xf32>
    %14 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x128x3x3xf32>} : () -> tensor<128x128x3x3xf32>
    %15 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x128x3x3xf32>} : () -> tensor<128x128x3x3xf32>
    %16 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x64x1x1xf32>} : () -> tensor<128x64x1x1xf32>
    %17 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x128x3x3xf32>} : () -> tensor<128x128x3x3xf32>
    %18 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x64x3x3xf32>} : () -> tensor<128x64x3x3xf32>
    %19 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x64x3x3xf32>} : () -> tensor<64x64x3x3xf32>
    %20 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x64x3x3xf32>} : () -> tensor<64x64x3x3xf32>
    %21 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x64x3x3xf32>} : () -> tensor<64x64x3x3xf32>
    %22 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x64x3x3xf32>} : () -> tensor<64x64x3x3xf32>
    %23 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x3x7x7xf32>} : () -> tensor<64x3x7x7xf32>
    %24 = "tosa.const"() {value = dense<0.000000e+00> : tensor<64xf32>} : () -> tensor<64xf32>
    %25 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
    %26 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %27 = "tosa.const"() {value = dense<0.000000e+00> : tensor<128xf32>} : () -> tensor<128xf32>
    %28 = "tosa.const"() {value = dense<0.000000e+00> : tensor<256xf32>} : () -> tensor<256xf32>
    %29 = "tosa.const"() {value = dense<0.000000e+00> : tensor<512xf32>} : () -> tensor<512xf32>
    %30 = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %31 = "tosa.const"() {value = dense<9.99999974E-6> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
    %32 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x1x1xf32>} : () -> tensor<64x1x1xf32>
    %33 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %34 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %35 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %36 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x1x1xf32>} : () -> tensor<64x1x1xf32>
    %37 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %38 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %39 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %40 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x1x1xf32>} : () -> tensor<64x1x1xf32>
    %41 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %42 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %43 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %44 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x1x1xf32>} : () -> tensor<64x1x1xf32>
    %45 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %46 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %47 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %48 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x1x1xf32>} : () -> tensor<64x1x1xf32>
    %49 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %50 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %51 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %52 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x1x1xf32>} : () -> tensor<128x1x1xf32>
    %53 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %54 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %55 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %56 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x1x1xf32>} : () -> tensor<128x1x1xf32>
    %57 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %58 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %59 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x1x1xf32>} : () -> tensor<128x1x1xf32>
    %60 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %61 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %62 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x1x1xf32>} : () -> tensor<128x1x1xf32>
    %63 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %64 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %65 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %66 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x1x1xf32>} : () -> tensor<128x1x1xf32>
    %67 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %68 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %69 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %70 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x1x1xf32>} : () -> tensor<256x1x1xf32>
    %71 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %72 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %73 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %74 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x1x1xf32>} : () -> tensor<256x1x1xf32>
    %75 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %76 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %77 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x1x1xf32>} : () -> tensor<256x1x1xf32>
    %78 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %79 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %80 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x1x1xf32>} : () -> tensor<256x1x1xf32>
    %81 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %82 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %83 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %84 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x1x1xf32>} : () -> tensor<256x1x1xf32>
    %85 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %86 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %87 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %88 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x1x1xf32>} : () -> tensor<512x1x1xf32>
    %89 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %90 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %91 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %92 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x1x1xf32>} : () -> tensor<512x1x1xf32>
    %93 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %94 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %95 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x1x1xf32>} : () -> tensor<512x1x1xf32>
    %96 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %97 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %98 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x1x1xf32>} : () -> tensor<512x1x1xf32>
    %99 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %100 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %101 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %102 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x1x1xf32>} : () -> tensor<512x1x1xf32>
    %103 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %104 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %105 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %106 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x1000xf32>} : () -> tensor<1x1000xf32>
    %107 = "tosa.transpose"(%arg0, %25) : (tensor<1x3x224x224xf32>, tensor<4xi32>) -> tensor<1x224x224x3xf32>
    %108 = "tosa.transpose"(%23, %25) : (tensor<64x3x7x7xf32>, tensor<4xi32>) -> tensor<64x7x7x3xf32>
    %109 = "tosa.conv2d"(%107, %108, %24) {dilation = [1, 1], pad = [3, 3, 3, 3], stride = [2, 2]} : (tensor<1x224x224x3xf32>, tensor<64x7x7x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
    %110 = "tosa.transpose"(%109, %26) : (tensor<1x112x112x64xf32>, tensor<4xi32>) -> tensor<1x64x112x112xf32>
    %111 = "tosa.sub"(%110, %33) : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %112 = "tosa.add"(%32, %31) : (tensor<64x1x1xf32>, tensor<1x1x1xf32>) -> tensor<64x1x1xf32>
    %113 = "tosa.rsqrt"(%112) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
    %114 = "tosa.reshape"(%113) {new_shape = [1, 64, 1, 1]} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %115 = "tosa.mul"(%111, %114) {shift = 0 : i32} : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %116 = "tosa.mul"(%115, %34) {shift = 0 : i32} : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %117 = "tosa.add"(%116, %35) : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %118 = "tosa.clamp"(%117) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %119 = "tosa.transpose"(%118, %25) : (tensor<1x64x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x64xf32>
    %120 = "tosa.max_pool2d"(%119) {kernel = [3, 3], pad = [1, 1, 1, 1], stride = [2, 2]} : (tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32>
    %121 = "tosa.transpose"(%120, %26) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %122 = "tosa.transpose"(%121, %25) : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %123 = "tosa.transpose"(%22, %25) : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %124 = "tosa.conv2d"(%122, %123, %24) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %125 = "tosa.transpose"(%124, %26) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %126 = "tosa.sub"(%125, %37) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %127 = "tosa.add"(%36, %31) : (tensor<64x1x1xf32>, tensor<1x1x1xf32>) -> tensor<64x1x1xf32>
    %128 = "tosa.rsqrt"(%127) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
    %129 = "tosa.reshape"(%128) {new_shape = [1, 64, 1, 1]} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %130 = "tosa.mul"(%126, %129) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %131 = "tosa.mul"(%130, %38) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %132 = "tosa.add"(%131, %39) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %133 = "tosa.clamp"(%132) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %134 = "tosa.transpose"(%133, %25) : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %135 = "tosa.transpose"(%21, %25) : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %136 = "tosa.conv2d"(%134, %135, %24) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %137 = "tosa.transpose"(%136, %26) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %138 = "tosa.sub"(%137, %41) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %139 = "tosa.add"(%40, %31) : (tensor<64x1x1xf32>, tensor<1x1x1xf32>) -> tensor<64x1x1xf32>
    %140 = "tosa.rsqrt"(%139) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
    %141 = "tosa.reshape"(%140) {new_shape = [1, 64, 1, 1]} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %142 = "tosa.mul"(%138, %141) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %143 = "tosa.mul"(%142, %42) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %144 = "tosa.add"(%143, %43) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %145 = "tosa.add"(%144, %121) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %146 = "tosa.clamp"(%145) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %147 = "tosa.transpose"(%146, %25) : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %148 = "tosa.transpose"(%20, %25) : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %149 = "tosa.conv2d"(%147, %148, %24) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %150 = "tosa.transpose"(%149, %26) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %151 = "tosa.sub"(%150, %45) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %152 = "tosa.add"(%44, %31) : (tensor<64x1x1xf32>, tensor<1x1x1xf32>) -> tensor<64x1x1xf32>
    %153 = "tosa.rsqrt"(%152) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
    %154 = "tosa.reshape"(%153) {new_shape = [1, 64, 1, 1]} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %155 = "tosa.mul"(%151, %154) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %156 = "tosa.mul"(%155, %46) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %157 = "tosa.add"(%156, %47) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %158 = "tosa.clamp"(%157) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %159 = "tosa.transpose"(%158, %25) : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %160 = "tosa.transpose"(%19, %25) : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %161 = "tosa.conv2d"(%159, %160, %24) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %162 = "tosa.transpose"(%161, %26) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %163 = "tosa.sub"(%162, %49) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %164 = "tosa.add"(%48, %31) : (tensor<64x1x1xf32>, tensor<1x1x1xf32>) -> tensor<64x1x1xf32>
    %165 = "tosa.rsqrt"(%164) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
    %166 = "tosa.reshape"(%165) {new_shape = [1, 64, 1, 1]} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %167 = "tosa.mul"(%163, %166) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %168 = "tosa.mul"(%167, %50) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %169 = "tosa.add"(%168, %51) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %170 = "tosa.add"(%169, %146) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %171 = "tosa.clamp"(%170) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %172 = "tosa.transpose"(%171, %25) : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %173 = "tosa.transpose"(%18, %25) : (tensor<128x64x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x64xf32>
    %174 = "tosa.conv2d"(%172, %173, %27) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [2, 2]} : (tensor<1x56x56x64xf32>, tensor<128x3x3x64xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %175 = "tosa.transpose"(%174, %26) : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %176 = "tosa.sub"(%175, %53) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %177 = "tosa.add"(%52, %31) : (tensor<128x1x1xf32>, tensor<1x1x1xf32>) -> tensor<128x1x1xf32>
    %178 = "tosa.rsqrt"(%177) : (tensor<128x1x1xf32>) -> tensor<128x1x1xf32>
    %179 = "tosa.reshape"(%178) {new_shape = [1, 128, 1, 1]} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %180 = "tosa.mul"(%176, %179) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %181 = "tosa.mul"(%180, %54) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %182 = "tosa.add"(%181, %55) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %183 = "tosa.clamp"(%182) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %184 = "tosa.transpose"(%183, %25) : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %185 = "tosa.transpose"(%17, %25) : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %186 = "tosa.conv2d"(%184, %185, %27) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %187 = "tosa.transpose"(%186, %26) : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %188 = "tosa.sub"(%187, %57) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %189 = "tosa.add"(%56, %31) : (tensor<128x1x1xf32>, tensor<1x1x1xf32>) -> tensor<128x1x1xf32>
    %190 = "tosa.rsqrt"(%189) : (tensor<128x1x1xf32>) -> tensor<128x1x1xf32>
    %191 = "tosa.reshape"(%190) {new_shape = [1, 128, 1, 1]} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %192 = "tosa.mul"(%188, %191) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %193 = "tosa.mul"(%192, %58) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %194 = "tosa.add"(%193, %2) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %195 = "tosa.transpose"(%16, %25) : (tensor<128x64x1x1xf32>, tensor<4xi32>) -> tensor<128x1x1x64xf32>
    %196 = "tosa.conv2d"(%172, %195, %27) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x56x56x64xf32>, tensor<128x1x1x64xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %197 = "tosa.transpose"(%196, %26) : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %198 = "tosa.sub"(%197, %60) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %199 = "tosa.add"(%59, %31) : (tensor<128x1x1xf32>, tensor<1x1x1xf32>) -> tensor<128x1x1xf32>
    %200 = "tosa.rsqrt"(%199) : (tensor<128x1x1xf32>) -> tensor<128x1x1xf32>
    %201 = "tosa.reshape"(%200) {new_shape = [1, 128, 1, 1]} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %202 = "tosa.mul"(%198, %201) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %203 = "tosa.mul"(%202, %61) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %204 = "tosa.add"(%203, %2) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %205 = "tosa.add"(%194, %204) : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %206 = "tosa.clamp"(%205) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %207 = "tosa.transpose"(%206, %25) : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %208 = "tosa.transpose"(%15, %25) : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %209 = "tosa.conv2d"(%207, %208, %27) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %210 = "tosa.transpose"(%209, %26) : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %211 = "tosa.sub"(%210, %63) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %212 = "tosa.add"(%62, %31) : (tensor<128x1x1xf32>, tensor<1x1x1xf32>) -> tensor<128x1x1xf32>
    %213 = "tosa.rsqrt"(%212) : (tensor<128x1x1xf32>) -> tensor<128x1x1xf32>
    %214 = "tosa.reshape"(%213) {new_shape = [1, 128, 1, 1]} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %215 = "tosa.mul"(%211, %214) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %216 = "tosa.mul"(%215, %64) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %217 = "tosa.add"(%216, %65) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %218 = "tosa.clamp"(%217) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %219 = "tosa.transpose"(%218, %25) : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %220 = "tosa.transpose"(%14, %25) : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %221 = "tosa.conv2d"(%219, %220, %27) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %222 = "tosa.transpose"(%221, %26) : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %223 = "tosa.sub"(%222, %67) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %224 = "tosa.add"(%66, %31) : (tensor<128x1x1xf32>, tensor<1x1x1xf32>) -> tensor<128x1x1xf32>
    %225 = "tosa.rsqrt"(%224) : (tensor<128x1x1xf32>) -> tensor<128x1x1xf32>
    %226 = "tosa.reshape"(%225) {new_shape = [1, 128, 1, 1]} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %227 = "tosa.mul"(%223, %226) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %228 = "tosa.mul"(%227, %68) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %229 = "tosa.add"(%228, %69) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %230 = "tosa.add"(%229, %206) : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %231 = "tosa.clamp"(%230) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %232 = "tosa.transpose"(%231, %25) : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %233 = "tosa.transpose"(%13, %25) : (tensor<256x128x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x128xf32>
    %234 = "tosa.conv2d"(%232, %233, %28) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [2, 2]} : (tensor<1x28x28x128xf32>, tensor<256x3x3x128xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %235 = "tosa.transpose"(%234, %26) : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %236 = "tosa.sub"(%235, %71) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %237 = "tosa.add"(%70, %31) : (tensor<256x1x1xf32>, tensor<1x1x1xf32>) -> tensor<256x1x1xf32>
    %238 = "tosa.rsqrt"(%237) : (tensor<256x1x1xf32>) -> tensor<256x1x1xf32>
    %239 = "tosa.reshape"(%238) {new_shape = [1, 256, 1, 1]} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %240 = "tosa.mul"(%236, %239) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %241 = "tosa.mul"(%240, %72) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %242 = "tosa.add"(%241, %73) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %243 = "tosa.clamp"(%242) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %244 = "tosa.transpose"(%243, %25) : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %245 = "tosa.transpose"(%12, %25) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %246 = "tosa.conv2d"(%244, %245, %28) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %247 = "tosa.transpose"(%246, %26) : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %248 = "tosa.sub"(%247, %75) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %249 = "tosa.add"(%74, %31) : (tensor<256x1x1xf32>, tensor<1x1x1xf32>) -> tensor<256x1x1xf32>
    %250 = "tosa.rsqrt"(%249) : (tensor<256x1x1xf32>) -> tensor<256x1x1xf32>
    %251 = "tosa.reshape"(%250) {new_shape = [1, 256, 1, 1]} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %252 = "tosa.mul"(%248, %251) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %253 = "tosa.mul"(%252, %76) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %254 = "tosa.add"(%253, %1) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %255 = "tosa.transpose"(%11, %25) : (tensor<256x128x1x1xf32>, tensor<4xi32>) -> tensor<256x1x1x128xf32>
    %256 = "tosa.conv2d"(%232, %255, %28) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x28x28x128xf32>, tensor<256x1x1x128xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %257 = "tosa.transpose"(%256, %26) : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %258 = "tosa.sub"(%257, %78) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %259 = "tosa.add"(%77, %31) : (tensor<256x1x1xf32>, tensor<1x1x1xf32>) -> tensor<256x1x1xf32>
    %260 = "tosa.rsqrt"(%259) : (tensor<256x1x1xf32>) -> tensor<256x1x1xf32>
    %261 = "tosa.reshape"(%260) {new_shape = [1, 256, 1, 1]} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %262 = "tosa.mul"(%258, %261) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %263 = "tosa.mul"(%262, %79) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %264 = "tosa.add"(%263, %1) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %265 = "tosa.add"(%254, %264) : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %266 = "tosa.clamp"(%265) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %267 = "tosa.transpose"(%266, %25) : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %268 = "tosa.transpose"(%10, %25) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %269 = "tosa.conv2d"(%267, %268, %28) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %270 = "tosa.transpose"(%269, %26) : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %271 = "tosa.sub"(%270, %81) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %272 = "tosa.add"(%80, %31) : (tensor<256x1x1xf32>, tensor<1x1x1xf32>) -> tensor<256x1x1xf32>
    %273 = "tosa.rsqrt"(%272) : (tensor<256x1x1xf32>) -> tensor<256x1x1xf32>
    %274 = "tosa.reshape"(%273) {new_shape = [1, 256, 1, 1]} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %275 = "tosa.mul"(%271, %274) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %276 = "tosa.mul"(%275, %82) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %277 = "tosa.add"(%276, %83) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %278 = "tosa.clamp"(%277) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %279 = "tosa.transpose"(%278, %25) : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %280 = "tosa.transpose"(%9, %25) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %281 = "tosa.conv2d"(%279, %280, %28) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %282 = "tosa.transpose"(%281, %26) : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %283 = "tosa.sub"(%282, %85) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %284 = "tosa.add"(%84, %31) : (tensor<256x1x1xf32>, tensor<1x1x1xf32>) -> tensor<256x1x1xf32>
    %285 = "tosa.rsqrt"(%284) : (tensor<256x1x1xf32>) -> tensor<256x1x1xf32>
    %286 = "tosa.reshape"(%285) {new_shape = [1, 256, 1, 1]} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %287 = "tosa.mul"(%283, %286) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %288 = "tosa.mul"(%287, %86) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %289 = "tosa.add"(%288, %87) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %290 = "tosa.add"(%289, %266) : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %291 = "tosa.clamp"(%290) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %292 = "tosa.transpose"(%291, %25) : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %293 = "tosa.transpose"(%8, %25) : (tensor<512x256x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x256xf32>
    %294 = "tosa.conv2d"(%292, %293, %29) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [2, 2]} : (tensor<1x14x14x256xf32>, tensor<512x3x3x256xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %295 = "tosa.transpose"(%294, %26) : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %296 = "tosa.sub"(%295, %89) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %297 = "tosa.add"(%88, %31) : (tensor<512x1x1xf32>, tensor<1x1x1xf32>) -> tensor<512x1x1xf32>
    %298 = "tosa.rsqrt"(%297) : (tensor<512x1x1xf32>) -> tensor<512x1x1xf32>
    %299 = "tosa.reshape"(%298) {new_shape = [1, 512, 1, 1]} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %300 = "tosa.mul"(%296, %299) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %301 = "tosa.mul"(%300, %90) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %302 = "tosa.add"(%301, %91) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %303 = "tosa.clamp"(%302) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %304 = "tosa.transpose"(%303, %25) : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %305 = "tosa.transpose"(%7, %25) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %306 = "tosa.conv2d"(%304, %305, %29) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %307 = "tosa.transpose"(%306, %26) : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %308 = "tosa.sub"(%307, %93) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %309 = "tosa.add"(%92, %31) : (tensor<512x1x1xf32>, tensor<1x1x1xf32>) -> tensor<512x1x1xf32>
    %310 = "tosa.rsqrt"(%309) : (tensor<512x1x1xf32>) -> tensor<512x1x1xf32>
    %311 = "tosa.reshape"(%310) {new_shape = [1, 512, 1, 1]} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %312 = "tosa.mul"(%308, %311) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %313 = "tosa.mul"(%312, %94) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %314 = "tosa.add"(%313, %0) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %315 = "tosa.transpose"(%6, %25) : (tensor<512x256x1x1xf32>, tensor<4xi32>) -> tensor<512x1x1x256xf32>
    %316 = "tosa.conv2d"(%292, %315, %29) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x14x14x256xf32>, tensor<512x1x1x256xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %317 = "tosa.transpose"(%316, %26) : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %318 = "tosa.sub"(%317, %96) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %319 = "tosa.add"(%95, %31) : (tensor<512x1x1xf32>, tensor<1x1x1xf32>) -> tensor<512x1x1xf32>
    %320 = "tosa.rsqrt"(%319) : (tensor<512x1x1xf32>) -> tensor<512x1x1xf32>
    %321 = "tosa.reshape"(%320) {new_shape = [1, 512, 1, 1]} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %322 = "tosa.mul"(%318, %321) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %323 = "tosa.mul"(%322, %97) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %324 = "tosa.add"(%323, %0) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %325 = "tosa.add"(%314, %324) : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %326 = "tosa.clamp"(%325) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %327 = "tosa.transpose"(%326, %25) : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %328 = "tosa.transpose"(%5, %25) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %329 = "tosa.conv2d"(%327, %328, %29) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %330 = "tosa.transpose"(%329, %26) : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %331 = "tosa.sub"(%330, %99) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %332 = "tosa.add"(%98, %31) : (tensor<512x1x1xf32>, tensor<1x1x1xf32>) -> tensor<512x1x1xf32>
    %333 = "tosa.rsqrt"(%332) : (tensor<512x1x1xf32>) -> tensor<512x1x1xf32>
    %334 = "tosa.reshape"(%333) {new_shape = [1, 512, 1, 1]} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %335 = "tosa.mul"(%331, %334) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %336 = "tosa.mul"(%335, %100) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %337 = "tosa.add"(%336, %101) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %338 = "tosa.clamp"(%337) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %339 = "tosa.transpose"(%338, %25) : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %340 = "tosa.transpose"(%4, %25) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %341 = "tosa.conv2d"(%339, %340, %29) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]} : (tensor<1x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %342 = "tosa.transpose"(%341, %26) : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %343 = "tosa.sub"(%342, %103) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %344 = "tosa.add"(%102, %31) : (tensor<512x1x1xf32>, tensor<1x1x1xf32>) -> tensor<512x1x1xf32>
    %345 = "tosa.rsqrt"(%344) : (tensor<512x1x1xf32>) -> tensor<512x1x1xf32>
    %346 = "tosa.reshape"(%345) {new_shape = [1, 512, 1, 1]} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %347 = "tosa.mul"(%343, %346) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %348 = "tosa.mul"(%347, %104) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %349 = "tosa.add"(%348, %105) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %350 = "tosa.add"(%349, %326) : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %351 = "tosa.clamp"(%350) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %352 = "tosa.transpose"(%351, %25) : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %353 = "tosa.avg_pool2d"(%352) {kernel = [7, 7], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x7x7x512xf32>) -> tensor<1x1x1x512xf32>
    %354 = "tosa.transpose"(%353, %26) : (tensor<1x1x1x512xf32>, tensor<4xi32>) -> tensor<1x512x1x1xf32>
    %355 = "tosa.transpose"(%3, %30) : (tensor<1000x512xf32>, tensor<2xi32>) -> tensor<512x1000xf32>
    %356 = "tosa.reshape"(%354) {new_shape = [1, 1, 512]} : (tensor<1x512x1x1xf32>) -> tensor<1x1x512xf32>
    %357 = "tosa.reshape"(%355) {new_shape = [1, 512, 1000]} : (tensor<512x1000xf32>) -> tensor<1x512x1000xf32>
    %358 = "tosa.matmul"(%356, %357) : (tensor<1x1x512xf32>, tensor<1x512x1000xf32>) -> tensor<1x1x1000xf32>
    %359 = "tosa.reshape"(%358) {new_shape = [1, 1000]} : (tensor<1x1x1000xf32>) -> tensor<1x1000xf32>
    %360 = "tosa.add"(%359, %106) : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %360 : tensor<1x1000xf32>
  }
}

