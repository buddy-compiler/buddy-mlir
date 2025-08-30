 module attributes {torch.debug_module_name = "ResNet"} {
  func.func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %1 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %2 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %3 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1000x512xf32>} : () -> tensor<1000x512xf32>
    %4 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x512x3x3xf32>} : () -> tensor<512x512x3x3xf32>
    %5 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x512x3x3xf32>} : () -> tensor<512x512x3x3xf32>
    %6 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x512x3x3xf32>} : () -> tensor<512x512x3x3xf32>
    %7 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x256x3x3xf32>} : () -> tensor<512x256x3x3xf32>
    %8 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x256x3x3xf32>} : () -> tensor<256x256x3x3xf32>
    %9 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x256x3x3xf32>} : () -> tensor<256x256x3x3xf32>
    %10 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x256x3x3xf32>} : () -> tensor<256x256x3x3xf32>
    %11 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x128x3x3xf32>} : () -> tensor<256x128x3x3xf32>
    %12 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x128x3x3xf32>} : () -> tensor<128x128x3x3xf32>
    %13 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x128x3x3xf32>} : () -> tensor<128x128x3x3xf32>
    %14 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x128x3x3xf32>} : () -> tensor<128x128x3x3xf32>
    %15 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x64x3x3xf32>} : () -> tensor<128x64x3x3xf32>
    %16 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x64x3x3xf32>} : () -> tensor<64x64x3x3xf32>
    %17 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x64x3x3xf32>} : () -> tensor<64x64x3x3xf32>
    %18 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x64x3x3xf32>} : () -> tensor<64x64x3x3xf32>
    %19 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x64x3x3xf32>} : () -> tensor<64x64x3x3xf32>
    %20 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x3x7x7xf32>} : () -> tensor<64x3x7x7xf32>
    %21 = "tosa.const"() {value = dense<0.000000e+00> : tensor<64xf32>} : () -> tensor<64xf32>
    %22 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
    %23 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %24 = "tosa.const"() {value = dense<0.000000e+00> : tensor<128xf32>} : () -> tensor<128xf32>
    %25 = "tosa.const"() {value = dense<0.000000e+00> : tensor<256xf32>} : () -> tensor<256xf32>
    %26 = "tosa.const"() {value = dense<0.000000e+00> : tensor<512xf32>} : () -> tensor<512xf32>
    %27 = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %28 = "tosa.const"() {value = dense<9.99999974E-6> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
    %29 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x1x1xf32>} : () -> tensor<64x1x1xf32>
    %30 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %31 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %32 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %33 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x1x1xf32>} : () -> tensor<64x1x1xf32>
    %34 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %35 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %36 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %37 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x1x1xf32>} : () -> tensor<64x1x1xf32>
    %38 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %39 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %40 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %41 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x1x1xf32>} : () -> tensor<64x1x1xf32>
    %42 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %43 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %44 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %45 = "tosa.const"() {value = dense_resource<__elided__> : tensor<64x1x1xf32>} : () -> tensor<64x1x1xf32>
    %46 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %47 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %48 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x64x1x1xf32>} : () -> tensor<1x64x1x1xf32>
    %49 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x1x1xf32>} : () -> tensor<128x1x1xf32>
    %50 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %51 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %52 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %53 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x1x1xf32>} : () -> tensor<128x1x1xf32>
    %54 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %55 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %56 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x1x1x64xf32>} : () -> tensor<128x1x1x64xf32>
    %57 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x1x1xf32>} : () -> tensor<128x1x1xf32>
    %58 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %59 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %60 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x1x1xf32>} : () -> tensor<128x1x1xf32>
    %61 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %62 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %63 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %64 = "tosa.const"() {value = dense_resource<__elided__> : tensor<128x1x1xf32>} : () -> tensor<128x1x1xf32>
    %65 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %66 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %67 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x128x1x1xf32>} : () -> tensor<1x128x1x1xf32>
    %68 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x1x1xf32>} : () -> tensor<256x1x1xf32>
    %69 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %70 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %71 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %72 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x1x1xf32>} : () -> tensor<256x1x1xf32>
    %73 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %74 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %75 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x1x1x128xf32>} : () -> tensor<256x1x1x128xf32>
    %76 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x1x1xf32>} : () -> tensor<256x1x1xf32>
    %77 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %78 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %79 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x1x1xf32>} : () -> tensor<256x1x1xf32>
    %80 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %81 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %82 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %83 = "tosa.const"() {value = dense_resource<__elided__> : tensor<256x1x1xf32>} : () -> tensor<256x1x1xf32>
    %84 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %85 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %86 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x256x1x1xf32>} : () -> tensor<1x256x1x1xf32>
    %87 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x1x1xf32>} : () -> tensor<512x1x1xf32>
    %88 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %89 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %90 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %91 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x1x1xf32>} : () -> tensor<512x1x1xf32>
    %92 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %93 = "tosa.const"() {value = dense_resource<__elided__> : tensor<1x512x1x1xf32>} : () -> tensor<1x512x1x1xf32>
    %94 = "tosa.const"() {value = dense_resource<__elided__> : tensor<512x1x1x256xf32>} : () -> tensor<512x1x1x256xf32>
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
    %107 = "tosa.transpose"(%arg0, %22) : (tensor<1x3x224x224xf32>, tensor<4xi32>) -> tensor<1x224x224x3xf32>
    %108 = "tosa.transpose"(%20, %22) : (tensor<64x3x7x7xf32>, tensor<4xi32>) -> tensor<64x7x7x3xf32>
    %109 = "tosa.conv2d"(%107, %108, %21) {dilation = array<i64: 1, 1>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>} : (tensor<1x224x224x3xf32>, tensor<64x7x7x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
    %110 = "tosa.transpose"(%109, %23) : (tensor<1x112x112x64xf32>, tensor<4xi32>) -> tensor<1x64x112x112xf32>
    %111 = "tosa.sub"(%110, %30) : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %112 = "tosa.add"(%29, %28) : (tensor<64x1x1xf32>, tensor<1x1x1xf32>) -> tensor<64x1x1xf32>
    %113 = "tosa.rsqrt"(%112) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
    %114 = "tosa.reshape"(%113) {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %115 = "tosa.mul"(%111, %114) {shift = 0 : i32} : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %116 = "tosa.mul"(%115, %31) {shift = 0 : i32} : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %117 = "tosa.add"(%116, %32) : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %118 = "tosa.clamp"(%117) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %119 = "tosa.transpose"(%118, %22) : (tensor<1x64x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x64xf32>
    %120 = "tosa.max_pool2d"(%119) {kernel = array<i64: 3, 3>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>} : (tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32>
    %121 = "tosa.transpose"(%120, %23) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %122 = "tosa.transpose"(%19, %22) : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %123 = "tosa.conv2d"(%120, %122, %21) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %124 = "tosa.transpose"(%123, %23) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %125 = "tosa.sub"(%124, %34) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %126 = "tosa.add"(%33, %28) : (tensor<64x1x1xf32>, tensor<1x1x1xf32>) -> tensor<64x1x1xf32>
    %127 = "tosa.rsqrt"(%126) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
    %128 = "tosa.reshape"(%127) {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %129 = "tosa.mul"(%125, %128) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %130 = "tosa.mul"(%129, %35) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %131 = "tosa.add"(%130, %36) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %132 = "tosa.clamp"(%131) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %133 = "tosa.transpose"(%132, %22) : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %134 = "tosa.transpose"(%18, %22) : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %135 = "tosa.conv2d"(%133, %134, %21) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %136 = "tosa.transpose"(%135, %23) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %137 = "tosa.sub"(%136, %38) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %138 = "tosa.add"(%37, %28) : (tensor<64x1x1xf32>, tensor<1x1x1xf32>) -> tensor<64x1x1xf32>
    %139 = "tosa.rsqrt"(%138) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
    %140 = "tosa.reshape"(%139) {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %141 = "tosa.mul"(%137, %140) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %142 = "tosa.mul"(%141, %39) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %143 = "tosa.add"(%142, %40) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %144 = "tosa.add"(%143, %121) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %145 = "tosa.clamp"(%144) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %146 = "tosa.transpose"(%145, %22) : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %147 = "tosa.transpose"(%17, %22) : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %148 = "tosa.conv2d"(%146, %147, %21) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %149 = "tosa.transpose"(%148, %23) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %150 = "tosa.sub"(%149, %42) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %151 = "tosa.add"(%41, %28) : (tensor<64x1x1xf32>, tensor<1x1x1xf32>) -> tensor<64x1x1xf32>
    %152 = "tosa.rsqrt"(%151) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
    %153 = "tosa.reshape"(%152) {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %154 = "tosa.mul"(%150, %153) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %155 = "tosa.mul"(%154, %43) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %156 = "tosa.add"(%155, %44) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %157 = "tosa.clamp"(%156) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %158 = "tosa.transpose"(%157, %22) : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %159 = "tosa.transpose"(%16, %22) : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %160 = "tosa.conv2d"(%158, %159, %21) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %161 = "tosa.transpose"(%160, %23) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %162 = "tosa.sub"(%161, %46) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %163 = "tosa.add"(%45, %28) : (tensor<64x1x1xf32>, tensor<1x1x1xf32>) -> tensor<64x1x1xf32>
    %164 = "tosa.rsqrt"(%163) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
    %165 = "tosa.reshape"(%164) {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %166 = "tosa.mul"(%162, %165) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %167 = "tosa.mul"(%166, %47) {shift = 0 : i32} : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %168 = "tosa.add"(%167, %48) : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %169 = "tosa.add"(%168, %145) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %170 = "tosa.clamp"(%169) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %171 = "tosa.transpose"(%170, %22) : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %172 = "tosa.transpose"(%15, %22) : (tensor<128x64x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x64xf32>
    %173 = "tosa.conv2d"(%171, %172, %24) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x56x56x64xf32>, tensor<128x3x3x64xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %174 = "tosa.transpose"(%173, %23) : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %175 = "tosa.sub"(%174, %50) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %176 = "tosa.add"(%49, %28) : (tensor<128x1x1xf32>, tensor<1x1x1xf32>) -> tensor<128x1x1xf32>
    %177 = "tosa.rsqrt"(%176) : (tensor<128x1x1xf32>) -> tensor<128x1x1xf32>
    %178 = "tosa.reshape"(%177) {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %179 = "tosa.mul"(%175, %178) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %180 = "tosa.mul"(%179, %51) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %181 = "tosa.add"(%180, %52) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %182 = "tosa.clamp"(%181) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %183 = "tosa.transpose"(%182, %22) : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %184 = "tosa.transpose"(%14, %22) : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %185 = "tosa.conv2d"(%183, %184, %24) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %186 = "tosa.transpose"(%185, %23) : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %187 = "tosa.sub"(%186, %54) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %188 = "tosa.add"(%53, %28) : (tensor<128x1x1xf32>, tensor<1x1x1xf32>) -> tensor<128x1x1xf32>
    %189 = "tosa.rsqrt"(%188) : (tensor<128x1x1xf32>) -> tensor<128x1x1xf32>
    %190 = "tosa.reshape"(%189) {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %191 = "tosa.mul"(%187, %190) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %192 = "tosa.mul"(%191, %55) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %193 = "tosa.add"(%192, %2) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %194 = "tosa.conv2d"(%171, %56, %24) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x56x56x64xf32>, tensor<128x1x1x64xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %195 = "tosa.transpose"(%194, %23) : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %196 = "tosa.sub"(%195, %58) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %197 = "tosa.add"(%57, %28) : (tensor<128x1x1xf32>, tensor<1x1x1xf32>) -> tensor<128x1x1xf32>
    %198 = "tosa.rsqrt"(%197) : (tensor<128x1x1xf32>) -> tensor<128x1x1xf32>
    %199 = "tosa.reshape"(%198) {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %200 = "tosa.mul"(%196, %199) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %201 = "tosa.mul"(%200, %59) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %202 = "tosa.add"(%201, %2) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %203 = "tosa.add"(%193, %202) : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %204 = "tosa.clamp"(%203) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %205 = "tosa.transpose"(%204, %22) : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %206 = "tosa.transpose"(%13, %22) : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %207 = "tosa.conv2d"(%205, %206, %24) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %208 = "tosa.transpose"(%207, %23) : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %209 = "tosa.sub"(%208, %61) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %210 = "tosa.add"(%60, %28) : (tensor<128x1x1xf32>, tensor<1x1x1xf32>) -> tensor<128x1x1xf32>
    %211 = "tosa.rsqrt"(%210) : (tensor<128x1x1xf32>) -> tensor<128x1x1xf32>
    %212 = "tosa.reshape"(%211) {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %213 = "tosa.mul"(%209, %212) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %214 = "tosa.mul"(%213, %62) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %215 = "tosa.add"(%214, %63) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %216 = "tosa.clamp"(%215) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %217 = "tosa.transpose"(%216, %22) : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %218 = "tosa.transpose"(%12, %22) : (tensor<128x128x3x3xf32>, tensor<4xi32>) -> tensor<128x3x3x128xf32>
    %219 = "tosa.conv2d"(%217, %218, %24) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x28x28x128xf32>, tensor<128x3x3x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %220 = "tosa.transpose"(%219, %23) : (tensor<1x28x28x128xf32>, tensor<4xi32>) -> tensor<1x128x28x28xf32>
    %221 = "tosa.sub"(%220, %65) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %222 = "tosa.add"(%64, %28) : (tensor<128x1x1xf32>, tensor<1x1x1xf32>) -> tensor<128x1x1xf32>
    %223 = "tosa.rsqrt"(%222) : (tensor<128x1x1xf32>) -> tensor<128x1x1xf32>
    %224 = "tosa.reshape"(%223) {new_shape = array<i64: 1, 128, 1, 1>} : (tensor<128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %225 = "tosa.mul"(%221, %224) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %226 = "tosa.mul"(%225, %66) {shift = 0 : i32} : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %227 = "tosa.add"(%226, %67) : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %228 = "tosa.add"(%227, %204) : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %229 = "tosa.clamp"(%228) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %230 = "tosa.transpose"(%229, %22) : (tensor<1x128x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x128xf32>
    %231 = "tosa.transpose"(%11, %22) : (tensor<256x128x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x128xf32>
    %232 = "tosa.conv2d"(%230, %231, %25) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x28x28x128xf32>, tensor<256x3x3x128xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %233 = "tosa.transpose"(%232, %23) : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %234 = "tosa.sub"(%233, %69) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %235 = "tosa.add"(%68, %28) : (tensor<256x1x1xf32>, tensor<1x1x1xf32>) -> tensor<256x1x1xf32>
    %236 = "tosa.rsqrt"(%235) : (tensor<256x1x1xf32>) -> tensor<256x1x1xf32>
    %237 = "tosa.reshape"(%236) {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %238 = "tosa.mul"(%234, %237) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %239 = "tosa.mul"(%238, %70) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %240 = "tosa.add"(%239, %71) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %241 = "tosa.clamp"(%240) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %242 = "tosa.transpose"(%241, %22) : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %243 = "tosa.transpose"(%10, %22) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %244 = "tosa.conv2d"(%242, %243, %25) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %245 = "tosa.transpose"(%244, %23) : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %246 = "tosa.sub"(%245, %73) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %247 = "tosa.add"(%72, %28) : (tensor<256x1x1xf32>, tensor<1x1x1xf32>) -> tensor<256x1x1xf32>
    %248 = "tosa.rsqrt"(%247) : (tensor<256x1x1xf32>) -> tensor<256x1x1xf32>
    %249 = "tosa.reshape"(%248) {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %250 = "tosa.mul"(%246, %249) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %251 = "tosa.mul"(%250, %74) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %252 = "tosa.add"(%251, %1) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %253 = "tosa.conv2d"(%230, %75, %25) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x28x28x128xf32>, tensor<256x1x1x128xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %254 = "tosa.transpose"(%253, %23) : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %255 = "tosa.sub"(%254, %77) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %256 = "tosa.add"(%76, %28) : (tensor<256x1x1xf32>, tensor<1x1x1xf32>) -> tensor<256x1x1xf32>
    %257 = "tosa.rsqrt"(%256) : (tensor<256x1x1xf32>) -> tensor<256x1x1xf32>
    %258 = "tosa.reshape"(%257) {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %259 = "tosa.mul"(%255, %258) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %260 = "tosa.mul"(%259, %78) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %261 = "tosa.add"(%260, %1) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %262 = "tosa.add"(%252, %261) : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %263 = "tosa.clamp"(%262) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %264 = "tosa.transpose"(%263, %22) : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %265 = "tosa.transpose"(%9, %22) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %266 = "tosa.conv2d"(%264, %265, %25) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %267 = "tosa.transpose"(%266, %23) : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %268 = "tosa.sub"(%267, %80) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %269 = "tosa.add"(%79, %28) : (tensor<256x1x1xf32>, tensor<1x1x1xf32>) -> tensor<256x1x1xf32>
    %270 = "tosa.rsqrt"(%269) : (tensor<256x1x1xf32>) -> tensor<256x1x1xf32>
    %271 = "tosa.reshape"(%270) {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %272 = "tosa.mul"(%268, %271) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %273 = "tosa.mul"(%272, %81) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %274 = "tosa.add"(%273, %82) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %275 = "tosa.clamp"(%274) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %276 = "tosa.transpose"(%275, %22) : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %277 = "tosa.transpose"(%8, %22) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %278 = "tosa.conv2d"(%276, %277, %25) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %279 = "tosa.transpose"(%278, %23) : (tensor<1x14x14x256xf32>, tensor<4xi32>) -> tensor<1x256x14x14xf32>
    %280 = "tosa.sub"(%279, %84) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %281 = "tosa.add"(%83, %28) : (tensor<256x1x1xf32>, tensor<1x1x1xf32>) -> tensor<256x1x1xf32>
    %282 = "tosa.rsqrt"(%281) : (tensor<256x1x1xf32>) -> tensor<256x1x1xf32>
    %283 = "tosa.reshape"(%282) {new_shape = array<i64: 1, 256, 1, 1>} : (tensor<256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %284 = "tosa.mul"(%280, %283) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %285 = "tosa.mul"(%284, %85) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %286 = "tosa.add"(%285, %86) : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %287 = "tosa.add"(%286, %263) : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %288 = "tosa.clamp"(%287) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %289 = "tosa.transpose"(%288, %22) : (tensor<1x256x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x256xf32>
    %290 = "tosa.transpose"(%7, %22) : (tensor<512x256x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x256xf32>
    %291 = "tosa.conv2d"(%289, %290, %26) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x14x14x256xf32>, tensor<512x3x3x256xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %292 = "tosa.transpose"(%291, %23) : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %293 = "tosa.sub"(%292, %88) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %294 = "tosa.add"(%87, %28) : (tensor<512x1x1xf32>, tensor<1x1x1xf32>) -> tensor<512x1x1xf32>
    %295 = "tosa.rsqrt"(%294) : (tensor<512x1x1xf32>) -> tensor<512x1x1xf32>
    %296 = "tosa.reshape"(%295) {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %297 = "tosa.mul"(%293, %296) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %298 = "tosa.mul"(%297, %89) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %299 = "tosa.add"(%298, %90) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %300 = "tosa.clamp"(%299) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %301 = "tosa.transpose"(%300, %22) : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %302 = "tosa.transpose"(%6, %22) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %303 = "tosa.conv2d"(%301, %302, %26) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %304 = "tosa.transpose"(%303, %23) : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %305 = "tosa.sub"(%304, %92) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %306 = "tosa.add"(%91, %28) : (tensor<512x1x1xf32>, tensor<1x1x1xf32>) -> tensor<512x1x1xf32>
    %307 = "tosa.rsqrt"(%306) : (tensor<512x1x1xf32>) -> tensor<512x1x1xf32>
    %308 = "tosa.reshape"(%307) {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %309 = "tosa.mul"(%305, %308) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %310 = "tosa.mul"(%309, %93) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %311 = "tosa.add"(%310, %0) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %312 = "tosa.conv2d"(%289, %94, %26) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x14x14x256xf32>, tensor<512x1x1x256xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %313 = "tosa.transpose"(%312, %23) : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %314 = "tosa.sub"(%313, %96) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %315 = "tosa.add"(%95, %28) : (tensor<512x1x1xf32>, tensor<1x1x1xf32>) -> tensor<512x1x1xf32>
    %316 = "tosa.rsqrt"(%315) : (tensor<512x1x1xf32>) -> tensor<512x1x1xf32>
    %317 = "tosa.reshape"(%316) {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %318 = "tosa.mul"(%314, %317) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %319 = "tosa.mul"(%318, %97) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %320 = "tosa.add"(%319, %0) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %321 = "tosa.add"(%311, %320) : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %322 = "tosa.clamp"(%321) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %323 = "tosa.transpose"(%322, %22) : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %324 = "tosa.transpose"(%5, %22) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %325 = "tosa.conv2d"(%323, %324, %26) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %326 = "tosa.transpose"(%325, %23) : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %327 = "tosa.sub"(%326, %99) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %328 = "tosa.add"(%98, %28) : (tensor<512x1x1xf32>, tensor<1x1x1xf32>) -> tensor<512x1x1xf32>
    %329 = "tosa.rsqrt"(%328) : (tensor<512x1x1xf32>) -> tensor<512x1x1xf32>
    %330 = "tosa.reshape"(%329) {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %331 = "tosa.mul"(%327, %330) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %332 = "tosa.mul"(%331, %100) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %333 = "tosa.add"(%332, %101) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %334 = "tosa.clamp"(%333) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %335 = "tosa.transpose"(%334, %22) : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %336 = "tosa.transpose"(%4, %22) : (tensor<512x512x3x3xf32>, tensor<4xi32>) -> tensor<512x3x3x512xf32>
    %337 = "tosa.conv2d"(%335, %336, %26) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %338 = "tosa.transpose"(%337, %23) : (tensor<1x7x7x512xf32>, tensor<4xi32>) -> tensor<1x512x7x7xf32>
    %339 = "tosa.sub"(%338, %103) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %340 = "tosa.add"(%102, %28) : (tensor<512x1x1xf32>, tensor<1x1x1xf32>) -> tensor<512x1x1xf32>
    %341 = "tosa.rsqrt"(%340) : (tensor<512x1x1xf32>) -> tensor<512x1x1xf32>
    %342 = "tosa.reshape"(%341) {new_shape = array<i64: 1, 512, 1, 1>} : (tensor<512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %343 = "tosa.mul"(%339, %342) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %344 = "tosa.mul"(%343, %104) {shift = 0 : i32} : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %345 = "tosa.add"(%344, %105) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %346 = "tosa.add"(%345, %322) : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %347 = "tosa.clamp"(%346) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %348 = "tosa.transpose"(%347, %22) : (tensor<1x512x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x512xf32>
    %349 = "tosa.avg_pool2d"(%348) {kernel = array<i64: 7, 7>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x512xf32>) -> tensor<1x1x1x512xf32>
    %350 = "tosa.transpose"(%3, %27) : (tensor<1000x512xf32>, tensor<2xi32>) -> tensor<512x1000xf32>
    %351 = "tosa.reshape"(%349) {new_shape = array<i64: 1, 1, 512>} : (tensor<1x1x1x512xf32>) -> tensor<1x1x512xf32>
    %352 = "tosa.reshape"(%350) {new_shape = array<i64: 1, 512, 1000>} : (tensor<512x1000xf32>) -> tensor<1x512x1000xf32>
    %353 = "tosa.matmul"(%351, %352) : (tensor<1x1x512xf32>, tensor<1x512x1000xf32>) -> tensor<1x1x1000xf32>
    %354 = "tosa.reshape"(%353) {new_shape = array<i64: 1, 1000>} : (tensor<1x1x1000xf32>) -> tensor<1x1000xf32>
    %355 = "tosa.add"(%354, %106) : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %355 : tensor<1x1000xf32>
  }
}
