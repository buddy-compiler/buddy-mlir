#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
#map7 = affine_map<(d0, d1) -> (0, d0, d1)>
module {
  func.func @subgraph0(%arg0: tensor<32000x4096xf32>, %arg1: tensor<1x40xi64>, %arg2: tensor<4096xf32>, %arg3: tensor<4096x4096xf32>, %arg4: tensor<4096x4096xf32>, %arg5: tensor<4096x4096xf32>, %arg6: tensor<1x1x2048x128xf32>, %arg7: tensor<1x1x2048x128xf32>, %arg8: tensor<4096x4096xf32>, %arg9: tensor<4096xf32>, %arg10: tensor<11008x4096xf32>, %arg11: tensor<11008x4096xf32>, %arg12: tensor<4096x11008xf32>, %arg13: tensor<4096xf32>, %arg14: tensor<4096x4096xf32>, %arg15: tensor<4096x4096xf32>, %arg16: tensor<4096x4096xf32>, %arg17: tensor<1x1x2048x128xf32>, %arg18: tensor<1x1x2048x128xf32>, %arg19: tensor<4096x4096xf32>, %arg20: tensor<4096xf32>, %arg21: tensor<11008x4096xf32>, %arg22: tensor<11008x4096xf32>, %arg23: tensor<4096x11008xf32>, %arg24: tensor<4096xf32>, %arg25: tensor<4096x4096xf32>, %arg26: tensor<4096x4096xf32>, %arg27: tensor<4096x4096xf32>, %arg28: tensor<1x1x2048x128xf32>, %arg29: tensor<1x1x2048x128xf32>, %arg30: tensor<4096x4096xf32>, %arg31: tensor<4096xf32>, %arg32: tensor<11008x4096xf32>, %arg33: tensor<11008x4096xf32>, %arg34: tensor<4096x11008xf32>, %arg35: tensor<4096xf32>, %arg36: tensor<4096x4096xf32>, %arg37: tensor<4096x4096xf32>, %arg38: tensor<4096x4096xf32>, %arg39: tensor<1x1x2048x128xf32>, %arg40: tensor<1x1x2048x128xf32>, %arg41: tensor<4096x4096xf32>, %arg42: tensor<4096xf32>, %arg43: tensor<11008x4096xf32>, %arg44: tensor<11008x4096xf32>, %arg45: tensor<4096x11008xf32>, %arg46: tensor<4096xf32>, %arg47: tensor<4096x4096xf32>, %arg48: tensor<4096x4096xf32>, %arg49: tensor<4096x4096xf32>, %arg50: tensor<1x1x2048x128xf32>, %arg51: tensor<1x1x2048x128xf32>, %arg52: tensor<4096x4096xf32>, %arg53: tensor<4096xf32>, %arg54: tensor<11008x4096xf32>, %arg55: tensor<11008x4096xf32>, %arg56: tensor<4096x11008xf32>, %arg57: tensor<4096xf32>, %arg58: tensor<4096x4096xf32>, %arg59: tensor<4096x4096xf32>, %arg60: tensor<4096x4096xf32>, %arg61: tensor<1x1x2048x128xf32>, %arg62: tensor<1x1x2048x128xf32>, %arg63: tensor<4096x4096xf32>, %arg64: tensor<4096xf32>, %arg65: tensor<11008x4096xf32>, %arg66: tensor<11008x4096xf32>, %arg67: tensor<4096x11008xf32>, %arg68: tensor<4096xf32>, %arg69: tensor<4096x4096xf32>, %arg70: tensor<4096x4096xf32>, %arg71: tensor<4096x4096xf32>, %arg72: tensor<1x1x2048x128xf32>, %arg73: tensor<1x1x2048x128xf32>, %arg74: tensor<4096x4096xf32>, %arg75: tensor<4096xf32>, %arg76: tensor<11008x4096xf32>, %arg77: tensor<11008x4096xf32>, %arg78: tensor<4096x11008xf32>, %arg79: tensor<4096xf32>, %arg80: tensor<4096x4096xf32>, %arg81: tensor<4096x4096xf32>, %arg82: tensor<4096x4096xf32>, %arg83: tensor<1x1x2048x128xf32>, %arg84: tensor<1x1x2048x128xf32>, %arg85: tensor<4096x4096xf32>, %arg86: tensor<4096xf32>, %arg87: tensor<11008x4096xf32>, %arg88: tensor<11008x4096xf32>, %arg89: tensor<4096x11008xf32>, %arg90: tensor<4096xf32>, %arg91: tensor<4096x4096xf32>, %arg92: tensor<4096x4096xf32>, %arg93: tensor<4096x4096xf32>, %arg94: tensor<1x1x2048x128xf32>, %arg95: tensor<1x1x2048x128xf32>, %arg96: tensor<4096x4096xf32>, %arg97: tensor<4096xf32>, %arg98: tensor<11008x4096xf32>, %arg99: tensor<11008x4096xf32>, %arg100: tensor<4096x11008xf32>, %arg101: tensor<4096xf32>, %arg102: tensor<4096x4096xf32>, %arg103: tensor<4096x4096xf32>, %arg104: tensor<4096x4096xf32>, %arg105: tensor<1x1x2048x128xf32>, %arg106: tensor<1x1x2048x128xf32>, %arg107: tensor<4096x4096xf32>, %arg108: tensor<4096xf32>, %arg109: tensor<11008x4096xf32>, %arg110: tensor<11008x4096xf32>, %arg111: tensor<4096x11008xf32>, %arg112: tensor<4096xf32>, %arg113: tensor<4096x4096xf32>, %arg114: tensor<4096x4096xf32>, %arg115: tensor<4096x4096xf32>, %arg116: tensor<1x1x2048x128xf32>, %arg117: tensor<1x1x2048x128xf32>, %arg118: tensor<4096x4096xf32>, %arg119: tensor<4096xf32>, %arg120: tensor<11008x4096xf32>, %arg121: tensor<11008x4096xf32>, %arg122: tensor<4096x11008xf32>, %arg123: tensor<4096xf32>, %arg124: tensor<4096x4096xf32>, %arg125: tensor<4096x4096xf32>, %arg126: tensor<4096x4096xf32>, %arg127: tensor<1x1x2048x128xf32>, %arg128: tensor<1x1x2048x128xf32>, %arg129: tensor<4096x4096xf32>, %arg130: tensor<4096xf32>, %arg131: tensor<11008x4096xf32>, %arg132: tensor<11008x4096xf32>, %arg133: tensor<4096x11008xf32>, %arg134: tensor<4096xf32>, %arg135: tensor<4096x4096xf32>, %arg136: tensor<4096x4096xf32>, %arg137: tensor<4096x4096xf32>, %arg138: tensor<1x1x2048x128xf32>, %arg139: tensor<1x1x2048x128xf32>, %arg140: tensor<4096x4096xf32>, %arg141: tensor<4096xf32>, %arg142: tensor<11008x4096xf32>, %arg143: tensor<11008x4096xf32>, %arg144: tensor<4096x11008xf32>, %arg145: tensor<4096xf32>, %arg146: tensor<4096x4096xf32>, %arg147: tensor<4096x4096xf32>, %arg148: tensor<4096x4096xf32>, %arg149: tensor<1x1x2048x128xf32>, %arg150: tensor<1x1x2048x128xf32>, %arg151: tensor<4096x4096xf32>, %arg152: tensor<4096xf32>, %arg153: tensor<11008x4096xf32>, %arg154: tensor<11008x4096xf32>, %arg155: tensor<4096x11008xf32>, %arg156: tensor<4096xf32>, %arg157: tensor<4096x4096xf32>, %arg158: tensor<4096x4096xf32>, %arg159: tensor<4096x4096xf32>, %arg160: tensor<1x1x2048x128xf32>, %arg161: tensor<1x1x2048x128xf32>, %arg162: tensor<4096x4096xf32>, %arg163: tensor<4096xf32>, %arg164: tensor<11008x4096xf32>, %arg165: tensor<11008x4096xf32>, %arg166: tensor<4096x11008xf32>, %arg167: tensor<4096xf32>, %arg168: tensor<4096x4096xf32>, %arg169: tensor<4096x4096xf32>, %arg170: tensor<4096x4096xf32>, %arg171: tensor<1x1x2048x128xf32>, %arg172: tensor<1x1x2048x128xf32>, %arg173: tensor<4096x4096xf32>, %arg174: tensor<4096xf32>, %arg175: tensor<11008x4096xf32>, %arg176: tensor<11008x4096xf32>, %arg177: tensor<4096x11008xf32>, %arg178: tensor<4096xf32>, %arg179: tensor<4096x4096xf32>, %arg180: tensor<4096x4096xf32>, %arg181: tensor<4096x4096xf32>, %arg182: tensor<1x1x2048x128xf32>, %arg183: tensor<1x1x2048x128xf32>, %arg184: tensor<4096x4096xf32>, %arg185: tensor<4096xf32>, %arg186: tensor<11008x4096xf32>, %arg187: tensor<11008x4096xf32>, %arg188: tensor<4096x11008xf32>, %arg189: tensor<4096xf32>, %arg190: tensor<4096x4096xf32>, %arg191: tensor<4096x4096xf32>, %arg192: tensor<4096x4096xf32>, %arg193: tensor<1x1x2048x128xf32>, %arg194: tensor<1x1x2048x128xf32>, %arg195: tensor<4096x4096xf32>, %arg196: tensor<4096xf32>, %arg197: tensor<11008x4096xf32>, %arg198: tensor<11008x4096xf32>, %arg199: tensor<4096x11008xf32>, %arg200: tensor<4096xf32>, %arg201: tensor<4096x4096xf32>, %arg202: tensor<4096x4096xf32>, %arg203: tensor<4096x4096xf32>, %arg204: tensor<1x1x2048x128xf32>, %arg205: tensor<1x1x2048x128xf32>, %arg206: tensor<4096x4096xf32>, %arg207: tensor<4096xf32>, %arg208: tensor<11008x4096xf32>, %arg209: tensor<11008x4096xf32>, %arg210: tensor<4096x11008xf32>, %arg211: tensor<4096xf32>, %arg212: tensor<4096x4096xf32>, %arg213: tensor<4096x4096xf32>, %arg214: tensor<4096x4096xf32>, %arg215: tensor<1x1x2048x128xf32>, %arg216: tensor<1x1x2048x128xf32>, %arg217: tensor<4096x4096xf32>, %arg218: tensor<4096xf32>, %arg219: tensor<11008x4096xf32>, %arg220: tensor<11008x4096xf32>, %arg221: tensor<4096x11008xf32>, %arg222: tensor<4096xf32>, %arg223: tensor<4096x4096xf32>, %arg224: tensor<4096x4096xf32>, %arg225: tensor<4096x4096xf32>, %arg226: tensor<1x1x2048x128xf32>, %arg227: tensor<1x1x2048x128xf32>, %arg228: tensor<4096x4096xf32>, %arg229: tensor<4096xf32>, %arg230: tensor<11008x4096xf32>, %arg231: tensor<11008x4096xf32>, %arg232: tensor<4096x11008xf32>, %arg233: tensor<4096xf32>, %arg234: tensor<4096x4096xf32>, %arg235: tensor<4096x4096xf32>, %arg236: tensor<4096x4096xf32>, %arg237: tensor<1x1x2048x128xf32>, %arg238: tensor<1x1x2048x128xf32>, %arg239: tensor<4096x4096xf32>, %arg240: tensor<4096xf32>, %arg241: tensor<11008x4096xf32>, %arg242: tensor<11008x4096xf32>, %arg243: tensor<4096x11008xf32>, %arg244: tensor<4096xf32>, %arg245: tensor<4096x4096xf32>, %arg246: tensor<4096x4096xf32>, %arg247: tensor<4096x4096xf32>, %arg248: tensor<1x1x2048x128xf32>, %arg249: tensor<1x1x2048x128xf32>, %arg250: tensor<4096x4096xf32>, %arg251: tensor<4096xf32>, %arg252: tensor<11008x4096xf32>, %arg253: tensor<11008x4096xf32>, %arg254: tensor<4096x11008xf32>, %arg255: tensor<4096xf32>, %arg256: tensor<4096x4096xf32>, %arg257: tensor<4096x4096xf32>, %arg258: tensor<4096x4096xf32>, %arg259: tensor<1x1x2048x128xf32>, %arg260: tensor<1x1x2048x128xf32>, %arg261: tensor<4096x4096xf32>, %arg262: tensor<4096xf32>, %arg263: tensor<11008x4096xf32>, %arg264: tensor<11008x4096xf32>, %arg265: tensor<4096x11008xf32>, %arg266: tensor<4096xf32>, %arg267: tensor<4096x4096xf32>, %arg268: tensor<4096x4096xf32>, %arg269: tensor<4096x4096xf32>, %arg270: tensor<1x1x2048x128xf32>, %arg271: tensor<1x1x2048x128xf32>, %arg272: tensor<4096x4096xf32>, %arg273: tensor<4096xf32>, %arg274: tensor<11008x4096xf32>, %arg275: tensor<11008x4096xf32>, %arg276: tensor<4096x11008xf32>, %arg277: tensor<4096xf32>, %arg278: tensor<4096x4096xf32>, %arg279: tensor<4096x4096xf32>, %arg280: tensor<4096x4096xf32>, %arg281: tensor<1x1x2048x128xf32>, %arg282: tensor<1x1x2048x128xf32>, %arg283: tensor<4096x4096xf32>, %arg284: tensor<4096xf32>, %arg285: tensor<11008x4096xf32>, %arg286: tensor<11008x4096xf32>, %arg287: tensor<4096x11008xf32>, %arg288: tensor<4096xf32>, %arg289: tensor<4096x4096xf32>, %arg290: tensor<4096x4096xf32>, %arg291: tensor<4096x4096xf32>, %arg292: tensor<1x1x2048x128xf32>, %arg293: tensor<1x1x2048x128xf32>, %arg294: tensor<4096x4096xf32>, %arg295: tensor<4096xf32>, %arg296: tensor<11008x4096xf32>, %arg297: tensor<11008x4096xf32>, %arg298: tensor<4096x11008xf32>, %arg299: tensor<4096xf32>, %arg300: tensor<4096x4096xf32>, %arg301: tensor<4096x4096xf32>, %arg302: tensor<4096x4096xf32>, %arg303: tensor<1x1x2048x128xf32>, %arg304: tensor<1x1x2048x128xf32>, %arg305: tensor<4096x4096xf32>, %arg306: tensor<4096xf32>, %arg307: tensor<11008x4096xf32>, %arg308: tensor<11008x4096xf32>, %arg309: tensor<4096x11008xf32>, %arg310: tensor<4096xf32>, %arg311: tensor<4096x4096xf32>, %arg312: tensor<4096x4096xf32>, %arg313: tensor<4096x4096xf32>, %arg314: tensor<1x1x2048x128xf32>, %arg315: tensor<1x1x2048x128xf32>, %arg316: tensor<4096x4096xf32>, %arg317: tensor<4096xf32>, %arg318: tensor<11008x4096xf32>, %arg319: tensor<11008x4096xf32>, %arg320: tensor<4096x11008xf32>, %arg321: tensor<4096xf32>, %arg322: tensor<4096x4096xf32>, %arg323: tensor<4096x4096xf32>, %arg324: tensor<4096x4096xf32>, %arg325: tensor<1x1x2048x128xf32>, %arg326: tensor<1x1x2048x128xf32>, %arg327: tensor<4096x4096xf32>, %arg328: tensor<4096xf32>, %arg329: tensor<11008x4096xf32>, %arg330: tensor<11008x4096xf32>, %arg331: tensor<4096x11008xf32>, %arg332: tensor<4096xf32>, %arg333: tensor<4096x4096xf32>, %arg334: tensor<4096x4096xf32>, %arg335: tensor<4096x4096xf32>, %arg336: tensor<1x1x2048x128xf32>, %arg337: tensor<1x1x2048x128xf32>, %arg338: tensor<4096x4096xf32>, %arg339: tensor<4096xf32>, %arg340: tensor<11008x4096xf32>, %arg341: tensor<11008x4096xf32>, %arg342: tensor<4096x11008xf32>, %arg343: tensor<4096xf32>, %arg344: tensor<4096x4096xf32>, %arg345: tensor<4096x4096xf32>, %arg346: tensor<4096x4096xf32>, %arg347: tensor<1x1x2048x128xf32>, %arg348: tensor<1x1x2048x128xf32>, %arg349: tensor<4096x4096xf32>, %arg350: tensor<4096xf32>, %arg351: tensor<11008x4096xf32>, %arg352: tensor<11008x4096xf32>, %arg353: tensor<4096x11008xf32>, %arg354: tensor<4096xf32>, %arg355: tensor<32000x4096xf32>) -> (tensor<1x40x4096xf32>, tensor<1x40x32000xf32>) {
    %0 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
    %1 = tosa.reshape %0 {new_shape = array<i64: 1, 40>} : (tensor<40xi64>) -> tensor<1x40xi64>
    %2 = tosa.reshape %1 {new_shape = array<i64: 1, 40>} : (tensor<1x40xi64>) -> tensor<1x40xi64>
    %3 = tosa.cast %arg1 : (tensor<1x40xi64>) -> tensor<1x40xi32>
    %4 = tosa.reshape %arg0 {new_shape = array<i64: 1, 32000, 4096>} : (tensor<32000x4096xf32>) -> tensor<1x32000x4096xf32>
    %5 = tosa.gather %4, %3 : (tensor<1x32000x4096xf32>, tensor<1x40xi32>) -> tensor<1x40x4096xf32>
    %6 = tosa.reshape %5 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %cst = arith.constant dense<true> : tensor<1x40xi1>
    %cst_0 = arith.constant dense<-3.40282347E+38> : tensor<40x40xf32>
    %7 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
    %8 = "tosa.const"() <{value = dense<1> : tensor<40xi64>}> : () -> tensor<40xi64>
    %9 = tosa.add %7, %8 : (tensor<40xi64>, tensor<40xi64>) -> tensor<40xi64>
    %10 = tosa.reshape %9 {new_shape = array<i64: 40, 1>} : (tensor<40xi64>) -> tensor<40x1xi64>
    %11 = tensor.empty() : tensor<40x40xi1>
    %12 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7, %10 : tensor<40xi64>, tensor<40x1xi64>) outs(%11 : tensor<40x40xi1>) {
    ^bb0(%in: i64, %in_742: i64, %out: i1):
      %4175 = arith.cmpi slt, %in, %in_742 : i64
      linalg.yield %4175 : i1
    } -> tensor<40x40xi1>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %13 = tensor.empty() : tensor<40x40xf32>
    %14 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%12, %cst_0 : tensor<40x40xi1>, tensor<40x40xf32>) outs(%13 : tensor<40x40xf32>) {
    ^bb0(%in: i1, %in_742: f32, %out: f32):
      %4175 = arith.select %in, %cst_1, %in_742 : f32
      linalg.yield %4175 : f32
    } -> tensor<40x40xf32>
    %extracted_slice = tensor.extract_slice %cst[0, 0] [1, 40] [1, 1] : tensor<1x40xi1> to tensor<1x40xi1>
    %15 = tosa.reshape %extracted_slice {new_shape = array<i64: 1, 1, 40>} : (tensor<1x40xi1>) -> tensor<1x1x40xi1>
    %16 = tosa.reshape %15 {new_shape = array<i64: 1, 1, 1, 40>} : (tensor<1x1x40xi1>) -> tensor<1x1x1x40xi1>
    %extracted_slice_2 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 1, 40] [1, 1, 1, 1] : tensor<1x1x1x40xi1> to tensor<1x1x1x40xi1>
    %17 = "tosa.const"() <{value = dense<false> : tensor<1x1x40x40xi1>}> : () -> tensor<1x1x40x40xi1>
    %18 = tosa.add %extracted_slice_2, %17 : (tensor<1x1x1x40xi1>, tensor<1x1x40x40xi1>) -> tensor<1x1x40x40xi1>
    %19 = tosa.cast %18 : (tensor<1x1x40x40xi1>) -> tensor<1x1x40x40xf32>
    %20 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x40x40xf32>}> : () -> tensor<1x1x40x40xf32>
    %21 = tosa.sub %20, %19 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
    %22 = tosa.cast %21 : (tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xi1>
    %cst_3 = arith.constant -3.40282347E+38 : f32
    %23 = tensor.empty() : tensor<1x1x40x40xf32>
    %24 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %21 : tensor<1x1x40x40xi1>, tensor<1x1x40x40xf32>) outs(%23 : tensor<1x1x40x40xf32>) {
    ^bb0(%in: i1, %in_742: f32, %out: f32):
      %4175 = arith.select %in, %cst_3, %in_742 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x1x40x40xf32>
    %25 = tosa.reshape %14 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %26 = tosa.reshape %25 {new_shape = array<i64: 1, 1, 40, 40>} : (tensor<1x40x40xf32>) -> tensor<1x1x40x40xf32>
    %extracted_slice_4 = tensor.extract_slice %26[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_5 = tensor.extract_slice %extracted_slice_4[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %27 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x40xf32>}> : () -> tensor<1x1x40x40xf32>
    %28 = tosa.add %extracted_slice_5, %27 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
    %29 = tosa.add %24, %28 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
    // RMSNorm begins
    %30 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32 = arith.constant 2 : i32
    %31 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<1x40x4096xf32>) outs(%30 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %32 = tosa.reduce_sum %31 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %33 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %34 = tosa.reciprocal %33 : (tensor<1xf32>) -> tensor<1xf32>
    %35 = tosa.mul %34, %32 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %36 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %37 = tosa.add %35, %36 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %38 = tosa.rsqrt %37 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %39 = tosa.mul %6, %38 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %40 = tosa.reshape %arg2 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    // %41 is the input matrix X after embedding,
    // then there are three consecutive similar codes representing the calculation of Q, K, V (%46, %51, %56):
    %41 = tosa.mul %40, %39 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    
    %42 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %43 = tosa.transpose %arg3, %42 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %44 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %45 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%44, %43 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_6 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %46 = tosa.reshape %45 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    
    %47 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %48 = tosa.transpose %arg4, %47 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %49 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %50 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%49, %48 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_7 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %51 = tosa.reshape %50 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    
    %52 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %53 = tosa.transpose %arg5, %52 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %54 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %55 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%54, %53 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_8 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %56 = tosa.reshape %55 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    // completed the calculation of Q, K, V; dimensions is (batch, seq_len, num_heads, head_dims)
    // transpose Q, K, V dimensions for RoPE and dot product

    // // begin of RoPE
    %57 = tosa.reshape %46 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %58 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %59 = tosa.transpose %57, %58 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    
    %60 = tosa.reshape %51 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %61 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %62 = tosa.transpose %60, %61 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    
    %63 = tosa.reshape %56 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %64 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %65 = tosa.transpose %63, %64 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    
    %extracted_slice_9 = tensor.extract_slice %arg6[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_10 = tensor.extract_slice %extracted_slice_9[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_11 = tensor.extract_slice %extracted_slice_10[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_12 = tensor.extract_slice %arg7[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_13 = tensor.extract_slice %extracted_slice_12[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_14 = tensor.extract_slice %extracted_slice_13[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %66 = tensor.empty() : tensor<1x40x128xf32>
    %67 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_11 : tensor<1x1x40x128xf32>) outs(%66 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %68 = tensor.empty() : tensor<40x128xf32>
    %69 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%67 : tensor<1x40x128xf32>) outs(%68 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %70 = tensor.empty() : tensor<1x40x128xf32>
    %71 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<1x1x40x128xf32>) outs(%70 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %72 = tensor.empty() : tensor<40x128xf32>
    %73 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%71 : tensor<1x40x128xf32>) outs(%72 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    // precompute_theta_pos_frequencies function, which is used to calculating special values ​​of RoPE according to: https://hyper.ai/wiki/29220
    %74 = tensor.empty() : tensor<1x40x128xf32>
    %75 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%74 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %69[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %76 = tosa.reshape %75 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %77 = tensor.empty() : tensor<1x40x128xf32>
    %78 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%77 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %73[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %79 = tosa.reshape %78 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %80 = tosa.mul %59, %76 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_15 = tensor.extract_slice %59[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_16 = tensor.extract_slice %59[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %81 = tosa.negate %extracted_slice_16 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %82 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice = tensor.insert_slice %81 into %82[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_17 = tensor.insert_slice %extracted_slice_15 into %inserted_slice[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %83 = tosa.mul %inserted_slice_17, %79 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %84 = tosa.add %80, %83 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %85 = tosa.mul %62, %76 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_18 = tensor.extract_slice %62[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_19 = tensor.extract_slice %62[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %86 = tosa.negate %extracted_slice_19 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %87 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_20 = tensor.insert_slice %86 into %87[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_21 = tensor.insert_slice %extracted_slice_18 into %inserted_slice_20[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    // end of RoPE, begin of Softmax(QK/sqrt(d_k)):
    %88 = tosa.mul %inserted_slice_21, %79 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %89 = tosa.add %85, %88 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %90 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %91 = tosa.transpose %89, %90 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %92 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %93 = tosa.add %84, %92 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %94 = tosa.reshape %93 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %95 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %96 = tosa.add %91, %95 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %97 = tosa.reshape %96 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %98 = tosa.matmul %94, %97 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %99 = tosa.reshape %98 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %100 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %101 = tosa.reciprocal %100 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %102 = tosa.mul %99, %101 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %103 = tosa.add %102, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %104 = tosa.reduce_max %103 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %105 = tosa.sub %103, %104 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %106 = tosa.exp %105 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %107 = tosa.reduce_sum %106 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %108 = tosa.reciprocal %107 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %109 = tosa.mul %106, %108 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    // end of Softmax(QK/sqrt(d_k)), begin of matmul with V
    %110 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %111 = tosa.add %109, %110 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %112 = tosa.reshape %111 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %113 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %114 = tosa.add %65, %113 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %115 = tosa.reshape %114 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %116 = tosa.matmul %112, %115 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %117 = tosa.reshape %116 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    // complete one head Softmax(QK/sqrt(d_k)), collect all heads.
    %118 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %119 = tosa.transpose %117, %118 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %120 = tosa.identity %119 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %121 = tosa.reshape %120 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %122 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %123 = tosa.transpose %arg8, %122 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %124 = tosa.reshape %121 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_22 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %125 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%124, %123 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_22 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %126 = tosa.reshape %125 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %127 = tosa.add %6, %126 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    // end of GQA(Group Query Attention) block, begin of FFN block(RMSNorm --> SwiGLU).
    %128 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_23 = arith.constant 2 : i32
    %129 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127 : tensor<1x40x4096xf32>) outs(%128 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_23 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %130 = tosa.reduce_sum %129 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %131 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %132 = tosa.reciprocal %131 : (tensor<1xf32>) -> tensor<1xf32>
    %133 = tosa.mul %132, %130 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %134 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %135 = tosa.add %133, %134 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %136 = tosa.rsqrt %135 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %137 = tosa.mul %127, %136 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %138 = tosa.reshape %arg9 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %139 = tosa.mul %138, %137 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %140 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %141 = tosa.transpose %arg10, %140 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %142 = tosa.reshape %139 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_24 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %143 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%142, %141 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_24 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %144 = tosa.reshape %143 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %145 = tosa.sigmoid %144 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %146 = tosa.mul %144, %145 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %147 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %148 = tosa.transpose %arg11, %147 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %149 = tosa.reshape %139 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_25 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %150 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%149, %148 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_25 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %151 = tosa.reshape %150 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %152 = tosa.mul %146, %151 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %153 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %154 = tosa.transpose %arg12, %153 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %155 = tosa.reshape %152 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_26 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %156 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%155, %154 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_26 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %157 = tosa.reshape %156 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %158 = tosa.add %127, %157 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    // end of last decoder block, begin of new decoder block.
    %159 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_27 = arith.constant 2 : i32
    %160 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%158 : tensor<1x40x4096xf32>) outs(%159 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_27 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %161 = tosa.reduce_sum %160 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %162 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %163 = tosa.reciprocal %162 : (tensor<1xf32>) -> tensor<1xf32>
    %164 = tosa.mul %163, %161 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %165 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %166 = tosa.add %164, %165 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %167 = tosa.rsqrt %166 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %168 = tosa.mul %158, %167 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %169 = tosa.reshape %arg13 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    // %170 is the input matrix X after embedding, 
    // then there are three consecutive similar code block representing the calculation of Q, K, V (%175, %180, %185):
    %170 = tosa.mul %169, %168 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    
    %171 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %172 = tosa.transpose %arg14, %171 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %173 = tosa.reshape %170 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_28 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %174 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%173, %172 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_28 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %175 = tosa.reshape %174 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

    %176 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %177 = tosa.transpose %arg15, %176 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %178 = tosa.reshape %170 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_29 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %179 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%178, %177 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_29 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %180 = tosa.reshape %179 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

    %181 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %182 = tosa.transpose %arg16, %181 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %183 = tosa.reshape %170 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_30 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %184 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%183, %182 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_30 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %185 = tosa.reshape %184 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    // completed the calculation of Q, K, V above.
    %186 = tosa.reshape %175 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %187 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %188 = tosa.transpose %186, %187 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>

    %189 = tosa.reshape %180 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %190 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %191 = tosa.transpose %189, %190 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    
    %192 = tosa.reshape %185 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %193 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %194 = tosa.transpose %192, %193 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    
    %extracted_slice_31 = tensor.extract_slice %arg17[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_32 = tensor.extract_slice %extracted_slice_31[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_33 = tensor.extract_slice %extracted_slice_32[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_34 = tensor.extract_slice %arg18[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_35 = tensor.extract_slice %extracted_slice_34[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_36 = tensor.extract_slice %extracted_slice_35[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %195 = tensor.empty() : tensor<1x40x128xf32>
    %196 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_33 : tensor<1x1x40x128xf32>) outs(%195 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %197 = tensor.empty() : tensor<40x128xf32>
    // #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
    // #map3 = affine_map<(d0, d1) -> (d0, d1)>
    // #map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    // #map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    // #map6 = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
    // #map7 = affine_map<(d0, d1) -> (0, d0, d1)>
    %198 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%196 : tensor<1x40x128xf32>) outs(%197 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %199 = tensor.empty() : tensor<1x40x128xf32>
    %200 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_36 : tensor<1x1x40x128xf32>) outs(%199 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %201 = tensor.empty() : tensor<40x128xf32>
    %202 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%200 : tensor<1x40x128xf32>) outs(%201 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %203 = tensor.empty() : tensor<1x40x128xf32>
    %204 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%203 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %198[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %205 = tosa.reshape %204 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %206 = tensor.empty() : tensor<1x40x128xf32>
    %207 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%206 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %202[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %208 = tosa.reshape %207 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %209 = tosa.mul %188, %205 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_37 = tensor.extract_slice %188[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_38 = tensor.extract_slice %188[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %210 = tosa.negate %extracted_slice_38 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %211 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_39 = tensor.insert_slice %210 into %211[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_40 = tensor.insert_slice %extracted_slice_37 into %inserted_slice_39[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %212 = tosa.mul %inserted_slice_40, %208 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %213 = tosa.add %209, %212 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    
    %214 = tosa.mul %191, %205 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_41 = tensor.extract_slice %191[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_42 = tensor.extract_slice %191[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %215 = tosa.negate %extracted_slice_42 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %216 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_43 = tensor.insert_slice %215 into %216[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_44 = tensor.insert_slice %extracted_slice_41 into %inserted_slice_43[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %217 = tosa.mul %inserted_slice_44, %208 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %218 = tosa.add %214, %217 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    
    %219 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %220 = tosa.transpose %218, %219 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %221 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %222 = tosa.add %213, %221 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %223 = tosa.reshape %222 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %224 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %225 = tosa.add %220, %224 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %226 = tosa.reshape %225 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %227 = tosa.matmul %223, %226 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %228 = tosa.reshape %227 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %229 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %230 = tosa.reciprocal %229 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %231 = tosa.mul %228, %230 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %232 = tosa.add %231, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %233 = tosa.reduce_max %232 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %234 = tosa.sub %232, %233 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %235 = tosa.exp %234 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %236 = tosa.reduce_sum %235 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %237 = tosa.reciprocal %236 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %238 = tosa.mul %235, %237 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %239 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %240 = tosa.add %238, %239 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %241 = tosa.reshape %240 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %242 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %243 = tosa.add %194, %242 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %244 = tosa.reshape %243 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %245 = tosa.matmul %241, %244 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %246 = tosa.reshape %245 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %247 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %248 = tosa.transpose %246, %247 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %249 = tosa.identity %248 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %250 = tosa.reshape %249 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %251 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %252 = tosa.transpose %arg19, %251 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %253 = tosa.reshape %250 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_45 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %254 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%253, %252 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_45 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %255 = tosa.reshape %254 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %256 = tosa.add %158, %255 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %257 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_46 = arith.constant 2 : i32
    %258 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%256 : tensor<1x40x4096xf32>) outs(%257 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_46 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %259 = tosa.reduce_sum %258 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %260 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %261 = tosa.reciprocal %260 : (tensor<1xf32>) -> tensor<1xf32>
    %262 = tosa.mul %261, %259 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %263 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %264 = tosa.add %262, %263 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %265 = tosa.rsqrt %264 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %266 = tosa.mul %256, %265 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %267 = tosa.reshape %arg20 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %268 = tosa.mul %267, %266 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %269 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %270 = tosa.transpose %arg21, %269 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %271 = tosa.reshape %268 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_47 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %272 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%271, %270 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_47 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %273 = tosa.reshape %272 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %274 = tosa.sigmoid %273 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %275 = tosa.mul %273, %274 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %276 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %277 = tosa.transpose %arg22, %276 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %278 = tosa.reshape %268 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_48 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %279 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%278, %277 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_48 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %280 = tosa.reshape %279 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %281 = tosa.mul %275, %280 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %282 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %283 = tosa.transpose %arg23, %282 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %284 = tosa.reshape %281 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_49 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %285 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%284, %283 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_49 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %286 = tosa.reshape %285 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %287 = tosa.add %256, %286 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %288 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_50 = arith.constant 2 : i32
    %289 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%287 : tensor<1x40x4096xf32>) outs(%288 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_50 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %290 = tosa.reduce_sum %289 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %291 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %292 = tosa.reciprocal %291 : (tensor<1xf32>) -> tensor<1xf32>
    %293 = tosa.mul %292, %290 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %294 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %295 = tosa.add %293, %294 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %296 = tosa.rsqrt %295 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %297 = tosa.mul %287, %296 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %298 = tosa.reshape %arg24 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %299 = tosa.mul %298, %297 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %300 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %301 = tosa.transpose %arg25, %300 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %302 = tosa.reshape %299 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_51 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %303 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%302, %301 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_51 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %304 = tosa.reshape %303 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %305 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %306 = tosa.transpose %arg26, %305 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %307 = tosa.reshape %299 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_52 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %308 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%307, %306 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_52 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %309 = tosa.reshape %308 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %310 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %311 = tosa.transpose %arg27, %310 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %312 = tosa.reshape %299 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_53 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %313 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%312, %311 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_53 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %314 = tosa.reshape %313 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %315 = tosa.reshape %304 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %316 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %317 = tosa.transpose %315, %316 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %318 = tosa.reshape %309 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %319 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %320 = tosa.transpose %318, %319 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %321 = tosa.reshape %314 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %322 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %323 = tosa.transpose %321, %322 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_54 = tensor.extract_slice %arg28[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_55 = tensor.extract_slice %extracted_slice_54[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_56 = tensor.extract_slice %extracted_slice_55[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_57 = tensor.extract_slice %arg29[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_58 = tensor.extract_slice %extracted_slice_57[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_59 = tensor.extract_slice %extracted_slice_58[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %324 = tensor.empty() : tensor<1x40x128xf32>
    %325 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_56 : tensor<1x1x40x128xf32>) outs(%324 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %326 = tensor.empty() : tensor<40x128xf32>
    %327 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%325 : tensor<1x40x128xf32>) outs(%326 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %328 = tensor.empty() : tensor<1x40x128xf32>
    %329 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_59 : tensor<1x1x40x128xf32>) outs(%328 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %330 = tensor.empty() : tensor<40x128xf32>
    %331 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%329 : tensor<1x40x128xf32>) outs(%330 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %332 = tensor.empty() : tensor<1x40x128xf32>
    %333 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%332 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %327[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %334 = tosa.reshape %333 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %335 = tensor.empty() : tensor<1x40x128xf32>
    %336 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%335 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %331[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %337 = tosa.reshape %336 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %338 = tosa.mul %317, %334 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_60 = tensor.extract_slice %317[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_61 = tensor.extract_slice %317[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %339 = tosa.negate %extracted_slice_61 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %340 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_62 = tensor.insert_slice %339 into %340[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_63 = tensor.insert_slice %extracted_slice_60 into %inserted_slice_62[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %341 = tosa.mul %inserted_slice_63, %337 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %342 = tosa.add %338, %341 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %343 = tosa.mul %320, %334 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_64 = tensor.extract_slice %320[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_65 = tensor.extract_slice %320[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %344 = tosa.negate %extracted_slice_65 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %345 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_66 = tensor.insert_slice %344 into %345[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_67 = tensor.insert_slice %extracted_slice_64 into %inserted_slice_66[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %346 = tosa.mul %inserted_slice_67, %337 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %347 = tosa.add %343, %346 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %348 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %349 = tosa.transpose %347, %348 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %350 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %351 = tosa.add %342, %350 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %352 = tosa.reshape %351 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %353 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %354 = tosa.add %349, %353 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %355 = tosa.reshape %354 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %356 = tosa.matmul %352, %355 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %357 = tosa.reshape %356 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %358 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %359 = tosa.reciprocal %358 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %360 = tosa.mul %357, %359 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %361 = tosa.add %360, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %362 = tosa.reduce_max %361 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %363 = tosa.sub %361, %362 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %364 = tosa.exp %363 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %365 = tosa.reduce_sum %364 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %366 = tosa.reciprocal %365 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %367 = tosa.mul %364, %366 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %368 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %369 = tosa.add %367, %368 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %370 = tosa.reshape %369 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %371 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %372 = tosa.add %323, %371 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %373 = tosa.reshape %372 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %374 = tosa.matmul %370, %373 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %375 = tosa.reshape %374 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %376 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %377 = tosa.transpose %375, %376 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %378 = tosa.identity %377 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %379 = tosa.reshape %378 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %380 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %381 = tosa.transpose %arg30, %380 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %382 = tosa.reshape %379 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_68 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %383 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%382, %381 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_68 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %384 = tosa.reshape %383 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %385 = tosa.add %287, %384 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %386 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_69 = arith.constant 2 : i32
    %387 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%385 : tensor<1x40x4096xf32>) outs(%386 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_69 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %388 = tosa.reduce_sum %387 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %389 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %390 = tosa.reciprocal %389 : (tensor<1xf32>) -> tensor<1xf32>
    %391 = tosa.mul %390, %388 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %392 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %393 = tosa.add %391, %392 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %394 = tosa.rsqrt %393 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %395 = tosa.mul %385, %394 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %396 = tosa.reshape %arg31 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %397 = tosa.mul %396, %395 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %398 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %399 = tosa.transpose %arg32, %398 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %400 = tosa.reshape %397 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_70 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %401 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%400, %399 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_70 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %402 = tosa.reshape %401 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %403 = tosa.sigmoid %402 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %404 = tosa.mul %402, %403 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %405 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %406 = tosa.transpose %arg33, %405 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %407 = tosa.reshape %397 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_71 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %408 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%407, %406 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_71 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %409 = tosa.reshape %408 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %410 = tosa.mul %404, %409 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %411 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %412 = tosa.transpose %arg34, %411 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %413 = tosa.reshape %410 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_72 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %414 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%413, %412 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_72 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %415 = tosa.reshape %414 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %416 = tosa.add %385, %415 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %417 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_73 = arith.constant 2 : i32
    %418 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%416 : tensor<1x40x4096xf32>) outs(%417 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_73 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %419 = tosa.reduce_sum %418 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %420 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %421 = tosa.reciprocal %420 : (tensor<1xf32>) -> tensor<1xf32>
    %422 = tosa.mul %421, %419 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %423 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %424 = tosa.add %422, %423 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %425 = tosa.rsqrt %424 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %426 = tosa.mul %416, %425 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %427 = tosa.reshape %arg35 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %428 = tosa.mul %427, %426 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %429 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %430 = tosa.transpose %arg36, %429 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %431 = tosa.reshape %428 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_74 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %432 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%431, %430 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_74 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %433 = tosa.reshape %432 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %434 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %435 = tosa.transpose %arg37, %434 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %436 = tosa.reshape %428 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_75 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %437 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%436, %435 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_75 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %438 = tosa.reshape %437 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %439 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %440 = tosa.transpose %arg38, %439 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %441 = tosa.reshape %428 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_76 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %442 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%441, %440 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_76 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %443 = tosa.reshape %442 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %444 = tosa.reshape %433 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %445 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %446 = tosa.transpose %444, %445 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %447 = tosa.reshape %438 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %448 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %449 = tosa.transpose %447, %448 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %450 = tosa.reshape %443 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %451 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %452 = tosa.transpose %450, %451 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_77 = tensor.extract_slice %arg39[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_78 = tensor.extract_slice %extracted_slice_77[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_79 = tensor.extract_slice %extracted_slice_78[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_80 = tensor.extract_slice %arg40[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_81 = tensor.extract_slice %extracted_slice_80[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_82 = tensor.extract_slice %extracted_slice_81[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %453 = tensor.empty() : tensor<1x40x128xf32>
    %454 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_79 : tensor<1x1x40x128xf32>) outs(%453 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %455 = tensor.empty() : tensor<40x128xf32>
    %456 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%454 : tensor<1x40x128xf32>) outs(%455 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %457 = tensor.empty() : tensor<1x40x128xf32>
    %458 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_82 : tensor<1x1x40x128xf32>) outs(%457 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %459 = tensor.empty() : tensor<40x128xf32>
    %460 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%458 : tensor<1x40x128xf32>) outs(%459 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %461 = tensor.empty() : tensor<1x40x128xf32>
    %462 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%461 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %456[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %463 = tosa.reshape %462 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %464 = tensor.empty() : tensor<1x40x128xf32>
    %465 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%464 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %460[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %466 = tosa.reshape %465 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %467 = tosa.mul %446, %463 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_83 = tensor.extract_slice %446[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_84 = tensor.extract_slice %446[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %468 = tosa.negate %extracted_slice_84 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %469 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_85 = tensor.insert_slice %468 into %469[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_86 = tensor.insert_slice %extracted_slice_83 into %inserted_slice_85[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %470 = tosa.mul %inserted_slice_86, %466 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %471 = tosa.add %467, %470 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %472 = tosa.mul %449, %463 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_87 = tensor.extract_slice %449[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_88 = tensor.extract_slice %449[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %473 = tosa.negate %extracted_slice_88 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %474 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_89 = tensor.insert_slice %473 into %474[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_90 = tensor.insert_slice %extracted_slice_87 into %inserted_slice_89[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %475 = tosa.mul %inserted_slice_90, %466 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %476 = tosa.add %472, %475 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %477 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %478 = tosa.transpose %476, %477 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %479 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %480 = tosa.add %471, %479 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %481 = tosa.reshape %480 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %482 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %483 = tosa.add %478, %482 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %484 = tosa.reshape %483 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %485 = tosa.matmul %481, %484 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %486 = tosa.reshape %485 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %487 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %488 = tosa.reciprocal %487 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %489 = tosa.mul %486, %488 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %490 = tosa.add %489, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %491 = tosa.reduce_max %490 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %492 = tosa.sub %490, %491 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %493 = tosa.exp %492 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %494 = tosa.reduce_sum %493 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %495 = tosa.reciprocal %494 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %496 = tosa.mul %493, %495 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %497 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %498 = tosa.add %496, %497 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %499 = tosa.reshape %498 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %500 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %501 = tosa.add %452, %500 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %502 = tosa.reshape %501 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %503 = tosa.matmul %499, %502 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %504 = tosa.reshape %503 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %505 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %506 = tosa.transpose %504, %505 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %507 = tosa.identity %506 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %508 = tosa.reshape %507 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %509 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %510 = tosa.transpose %arg41, %509 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %511 = tosa.reshape %508 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_91 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %512 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%511, %510 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_91 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %513 = tosa.reshape %512 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %514 = tosa.add %416, %513 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %515 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_92 = arith.constant 2 : i32
    %516 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%514 : tensor<1x40x4096xf32>) outs(%515 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_92 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %517 = tosa.reduce_sum %516 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %518 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %519 = tosa.reciprocal %518 : (tensor<1xf32>) -> tensor<1xf32>
    %520 = tosa.mul %519, %517 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %521 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %522 = tosa.add %520, %521 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %523 = tosa.rsqrt %522 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %524 = tosa.mul %514, %523 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %525 = tosa.reshape %arg42 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %526 = tosa.mul %525, %524 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %527 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %528 = tosa.transpose %arg43, %527 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %529 = tosa.reshape %526 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_93 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %530 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%529, %528 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_93 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %531 = tosa.reshape %530 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %532 = tosa.sigmoid %531 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %533 = tosa.mul %531, %532 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %534 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %535 = tosa.transpose %arg44, %534 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %536 = tosa.reshape %526 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_94 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %537 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%536, %535 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_94 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %538 = tosa.reshape %537 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %539 = tosa.mul %533, %538 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %540 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %541 = tosa.transpose %arg45, %540 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %542 = tosa.reshape %539 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_95 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %543 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%542, %541 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_95 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %544 = tosa.reshape %543 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %545 = tosa.add %514, %544 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %546 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_96 = arith.constant 2 : i32
    %547 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%545 : tensor<1x40x4096xf32>) outs(%546 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_96 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %548 = tosa.reduce_sum %547 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %549 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %550 = tosa.reciprocal %549 : (tensor<1xf32>) -> tensor<1xf32>
    %551 = tosa.mul %550, %548 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %552 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %553 = tosa.add %551, %552 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %554 = tosa.rsqrt %553 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %555 = tosa.mul %545, %554 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %556 = tosa.reshape %arg46 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %557 = tosa.mul %556, %555 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %558 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %559 = tosa.transpose %arg47, %558 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %560 = tosa.reshape %557 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_97 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %561 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%560, %559 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_97 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %562 = tosa.reshape %561 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %563 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %564 = tosa.transpose %arg48, %563 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %565 = tosa.reshape %557 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_98 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %566 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%565, %564 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_98 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %567 = tosa.reshape %566 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %568 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %569 = tosa.transpose %arg49, %568 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %570 = tosa.reshape %557 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_99 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %571 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%570, %569 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_99 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %572 = tosa.reshape %571 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %573 = tosa.reshape %562 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %574 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %575 = tosa.transpose %573, %574 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %576 = tosa.reshape %567 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %577 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %578 = tosa.transpose %576, %577 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %579 = tosa.reshape %572 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %580 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %581 = tosa.transpose %579, %580 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_100 = tensor.extract_slice %arg50[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_101 = tensor.extract_slice %extracted_slice_100[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_102 = tensor.extract_slice %extracted_slice_101[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_103 = tensor.extract_slice %arg51[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_104 = tensor.extract_slice %extracted_slice_103[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_105 = tensor.extract_slice %extracted_slice_104[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %582 = tensor.empty() : tensor<1x40x128xf32>
    %583 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_102 : tensor<1x1x40x128xf32>) outs(%582 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %584 = tensor.empty() : tensor<40x128xf32>
    %585 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%583 : tensor<1x40x128xf32>) outs(%584 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %586 = tensor.empty() : tensor<1x40x128xf32>
    %587 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_105 : tensor<1x1x40x128xf32>) outs(%586 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %588 = tensor.empty() : tensor<40x128xf32>
    %589 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%587 : tensor<1x40x128xf32>) outs(%588 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %590 = tensor.empty() : tensor<1x40x128xf32>
    %591 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%590 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %585[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %592 = tosa.reshape %591 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %593 = tensor.empty() : tensor<1x40x128xf32>
    %594 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%593 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %589[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %595 = tosa.reshape %594 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %596 = tosa.mul %575, %592 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_106 = tensor.extract_slice %575[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_107 = tensor.extract_slice %575[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %597 = tosa.negate %extracted_slice_107 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %598 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_108 = tensor.insert_slice %597 into %598[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_109 = tensor.insert_slice %extracted_slice_106 into %inserted_slice_108[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %599 = tosa.mul %inserted_slice_109, %595 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %600 = tosa.add %596, %599 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %601 = tosa.mul %578, %592 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_110 = tensor.extract_slice %578[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_111 = tensor.extract_slice %578[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %602 = tosa.negate %extracted_slice_111 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %603 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_112 = tensor.insert_slice %602 into %603[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_113 = tensor.insert_slice %extracted_slice_110 into %inserted_slice_112[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %604 = tosa.mul %inserted_slice_113, %595 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %605 = tosa.add %601, %604 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %606 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %607 = tosa.transpose %605, %606 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %608 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %609 = tosa.add %600, %608 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %610 = tosa.reshape %609 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %611 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %612 = tosa.add %607, %611 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %613 = tosa.reshape %612 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %614 = tosa.matmul %610, %613 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %615 = tosa.reshape %614 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %616 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %617 = tosa.reciprocal %616 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %618 = tosa.mul %615, %617 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %619 = tosa.add %618, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %620 = tosa.reduce_max %619 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %621 = tosa.sub %619, %620 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %622 = tosa.exp %621 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %623 = tosa.reduce_sum %622 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %624 = tosa.reciprocal %623 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %625 = tosa.mul %622, %624 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %626 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %627 = tosa.add %625, %626 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %628 = tosa.reshape %627 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %629 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %630 = tosa.add %581, %629 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %631 = tosa.reshape %630 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %632 = tosa.matmul %628, %631 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %633 = tosa.reshape %632 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %634 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %635 = tosa.transpose %633, %634 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %636 = tosa.identity %635 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %637 = tosa.reshape %636 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %638 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %639 = tosa.transpose %arg52, %638 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %640 = tosa.reshape %637 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_114 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %641 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%640, %639 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_114 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %642 = tosa.reshape %641 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %643 = tosa.add %545, %642 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %644 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_115 = arith.constant 2 : i32
    %645 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%643 : tensor<1x40x4096xf32>) outs(%644 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_115 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %646 = tosa.reduce_sum %645 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %647 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %648 = tosa.reciprocal %647 : (tensor<1xf32>) -> tensor<1xf32>
    %649 = tosa.mul %648, %646 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %650 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %651 = tosa.add %649, %650 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %652 = tosa.rsqrt %651 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %653 = tosa.mul %643, %652 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %654 = tosa.reshape %arg53 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %655 = tosa.mul %654, %653 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %656 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %657 = tosa.transpose %arg54, %656 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %658 = tosa.reshape %655 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_116 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %659 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%658, %657 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_116 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %660 = tosa.reshape %659 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %661 = tosa.sigmoid %660 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %662 = tosa.mul %660, %661 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %663 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %664 = tosa.transpose %arg55, %663 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %665 = tosa.reshape %655 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_117 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %666 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%665, %664 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_117 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %667 = tosa.reshape %666 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %668 = tosa.mul %662, %667 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %669 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %670 = tosa.transpose %arg56, %669 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %671 = tosa.reshape %668 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_118 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %672 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%671, %670 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_118 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %673 = tosa.reshape %672 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %674 = tosa.add %643, %673 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %675 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_119 = arith.constant 2 : i32
    %676 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%674 : tensor<1x40x4096xf32>) outs(%675 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_119 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %677 = tosa.reduce_sum %676 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %678 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %679 = tosa.reciprocal %678 : (tensor<1xf32>) -> tensor<1xf32>
    %680 = tosa.mul %679, %677 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %681 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %682 = tosa.add %680, %681 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %683 = tosa.rsqrt %682 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %684 = tosa.mul %674, %683 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %685 = tosa.reshape %arg57 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %686 = tosa.mul %685, %684 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %687 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %688 = tosa.transpose %arg58, %687 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %689 = tosa.reshape %686 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_120 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %690 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%689, %688 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_120 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %691 = tosa.reshape %690 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %692 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %693 = tosa.transpose %arg59, %692 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %694 = tosa.reshape %686 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_121 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %695 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%694, %693 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_121 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %696 = tosa.reshape %695 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %697 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %698 = tosa.transpose %arg60, %697 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %699 = tosa.reshape %686 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_122 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %700 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%699, %698 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_122 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %701 = tosa.reshape %700 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %702 = tosa.reshape %691 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %703 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %704 = tosa.transpose %702, %703 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %705 = tosa.reshape %696 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %706 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %707 = tosa.transpose %705, %706 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %708 = tosa.reshape %701 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %709 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %710 = tosa.transpose %708, %709 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_123 = tensor.extract_slice %arg61[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_124 = tensor.extract_slice %extracted_slice_123[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_125 = tensor.extract_slice %extracted_slice_124[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_126 = tensor.extract_slice %arg62[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_127 = tensor.extract_slice %extracted_slice_126[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_128 = tensor.extract_slice %extracted_slice_127[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %711 = tensor.empty() : tensor<1x40x128xf32>
    %712 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_125 : tensor<1x1x40x128xf32>) outs(%711 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %713 = tensor.empty() : tensor<40x128xf32>
    %714 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%712 : tensor<1x40x128xf32>) outs(%713 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %715 = tensor.empty() : tensor<1x40x128xf32>
    %716 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_128 : tensor<1x1x40x128xf32>) outs(%715 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %717 = tensor.empty() : tensor<40x128xf32>
    %718 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%716 : tensor<1x40x128xf32>) outs(%717 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %719 = tensor.empty() : tensor<1x40x128xf32>
    %720 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%719 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %714[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %721 = tosa.reshape %720 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %722 = tensor.empty() : tensor<1x40x128xf32>
    %723 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%722 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %718[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %724 = tosa.reshape %723 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %725 = tosa.mul %704, %721 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_129 = tensor.extract_slice %704[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_130 = tensor.extract_slice %704[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %726 = tosa.negate %extracted_slice_130 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %727 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_131 = tensor.insert_slice %726 into %727[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_132 = tensor.insert_slice %extracted_slice_129 into %inserted_slice_131[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %728 = tosa.mul %inserted_slice_132, %724 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %729 = tosa.add %725, %728 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %730 = tosa.mul %707, %721 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_133 = tensor.extract_slice %707[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_134 = tensor.extract_slice %707[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %731 = tosa.negate %extracted_slice_134 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %732 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_135 = tensor.insert_slice %731 into %732[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_136 = tensor.insert_slice %extracted_slice_133 into %inserted_slice_135[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %733 = tosa.mul %inserted_slice_136, %724 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %734 = tosa.add %730, %733 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %735 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %736 = tosa.transpose %734, %735 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %737 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %738 = tosa.add %729, %737 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %739 = tosa.reshape %738 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %740 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %741 = tosa.add %736, %740 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %742 = tosa.reshape %741 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %743 = tosa.matmul %739, %742 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %744 = tosa.reshape %743 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %745 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %746 = tosa.reciprocal %745 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %747 = tosa.mul %744, %746 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %748 = tosa.add %747, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %749 = tosa.reduce_max %748 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %750 = tosa.sub %748, %749 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %751 = tosa.exp %750 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %752 = tosa.reduce_sum %751 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %753 = tosa.reciprocal %752 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %754 = tosa.mul %751, %753 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %755 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %756 = tosa.add %754, %755 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %757 = tosa.reshape %756 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %758 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %759 = tosa.add %710, %758 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %760 = tosa.reshape %759 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %761 = tosa.matmul %757, %760 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %762 = tosa.reshape %761 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %763 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %764 = tosa.transpose %762, %763 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %765 = tosa.identity %764 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %766 = tosa.reshape %765 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %767 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %768 = tosa.transpose %arg63, %767 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %769 = tosa.reshape %766 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_137 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %770 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%769, %768 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_137 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %771 = tosa.reshape %770 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %772 = tosa.add %674, %771 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %773 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_138 = arith.constant 2 : i32
    %774 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%772 : tensor<1x40x4096xf32>) outs(%773 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_138 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %775 = tosa.reduce_sum %774 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %776 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %777 = tosa.reciprocal %776 : (tensor<1xf32>) -> tensor<1xf32>
    %778 = tosa.mul %777, %775 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %779 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %780 = tosa.add %778, %779 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %781 = tosa.rsqrt %780 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %782 = tosa.mul %772, %781 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %783 = tosa.reshape %arg64 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %784 = tosa.mul %783, %782 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %785 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %786 = tosa.transpose %arg65, %785 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %787 = tosa.reshape %784 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_139 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %788 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%787, %786 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_139 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %789 = tosa.reshape %788 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %790 = tosa.sigmoid %789 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %791 = tosa.mul %789, %790 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %792 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %793 = tosa.transpose %arg66, %792 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %794 = tosa.reshape %784 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_140 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %795 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%794, %793 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_140 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %796 = tosa.reshape %795 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %797 = tosa.mul %791, %796 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %798 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %799 = tosa.transpose %arg67, %798 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %800 = tosa.reshape %797 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_141 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %801 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%800, %799 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_141 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %802 = tosa.reshape %801 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %803 = tosa.add %772, %802 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %804 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_142 = arith.constant 2 : i32
    %805 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%803 : tensor<1x40x4096xf32>) outs(%804 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_142 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %806 = tosa.reduce_sum %805 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %807 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %808 = tosa.reciprocal %807 : (tensor<1xf32>) -> tensor<1xf32>
    %809 = tosa.mul %808, %806 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %810 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %811 = tosa.add %809, %810 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %812 = tosa.rsqrt %811 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %813 = tosa.mul %803, %812 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %814 = tosa.reshape %arg68 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %815 = tosa.mul %814, %813 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %816 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %817 = tosa.transpose %arg69, %816 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %818 = tosa.reshape %815 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_143 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %819 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%818, %817 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_143 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %820 = tosa.reshape %819 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %821 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %822 = tosa.transpose %arg70, %821 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %823 = tosa.reshape %815 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_144 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %824 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%823, %822 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_144 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %825 = tosa.reshape %824 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %826 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %827 = tosa.transpose %arg71, %826 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %828 = tosa.reshape %815 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_145 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %829 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%828, %827 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_145 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %830 = tosa.reshape %829 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %831 = tosa.reshape %820 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %832 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %833 = tosa.transpose %831, %832 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %834 = tosa.reshape %825 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %835 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %836 = tosa.transpose %834, %835 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %837 = tosa.reshape %830 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %838 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %839 = tosa.transpose %837, %838 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_146 = tensor.extract_slice %arg72[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_147 = tensor.extract_slice %extracted_slice_146[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_148 = tensor.extract_slice %extracted_slice_147[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_149 = tensor.extract_slice %arg73[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_150 = tensor.extract_slice %extracted_slice_149[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_151 = tensor.extract_slice %extracted_slice_150[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %840 = tensor.empty() : tensor<1x40x128xf32>
    %841 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_148 : tensor<1x1x40x128xf32>) outs(%840 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %842 = tensor.empty() : tensor<40x128xf32>
    %843 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%841 : tensor<1x40x128xf32>) outs(%842 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %844 = tensor.empty() : tensor<1x40x128xf32>
    %845 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_151 : tensor<1x1x40x128xf32>) outs(%844 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %846 = tensor.empty() : tensor<40x128xf32>
    %847 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%845 : tensor<1x40x128xf32>) outs(%846 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %848 = tensor.empty() : tensor<1x40x128xf32>
    %849 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%848 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %843[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %850 = tosa.reshape %849 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %851 = tensor.empty() : tensor<1x40x128xf32>
    %852 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%851 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %847[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %853 = tosa.reshape %852 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %854 = tosa.mul %833, %850 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_152 = tensor.extract_slice %833[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_153 = tensor.extract_slice %833[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %855 = tosa.negate %extracted_slice_153 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %856 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_154 = tensor.insert_slice %855 into %856[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_155 = tensor.insert_slice %extracted_slice_152 into %inserted_slice_154[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %857 = tosa.mul %inserted_slice_155, %853 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %858 = tosa.add %854, %857 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %859 = tosa.mul %836, %850 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_156 = tensor.extract_slice %836[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_157 = tensor.extract_slice %836[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %860 = tosa.negate %extracted_slice_157 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %861 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_158 = tensor.insert_slice %860 into %861[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_159 = tensor.insert_slice %extracted_slice_156 into %inserted_slice_158[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %862 = tosa.mul %inserted_slice_159, %853 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %863 = tosa.add %859, %862 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %864 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %865 = tosa.transpose %863, %864 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %866 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %867 = tosa.add %858, %866 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %868 = tosa.reshape %867 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %869 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %870 = tosa.add %865, %869 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %871 = tosa.reshape %870 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %872 = tosa.matmul %868, %871 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %873 = tosa.reshape %872 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %874 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %875 = tosa.reciprocal %874 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %876 = tosa.mul %873, %875 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %877 = tosa.add %876, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %878 = tosa.reduce_max %877 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %879 = tosa.sub %877, %878 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %880 = tosa.exp %879 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %881 = tosa.reduce_sum %880 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %882 = tosa.reciprocal %881 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %883 = tosa.mul %880, %882 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %884 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %885 = tosa.add %883, %884 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %886 = tosa.reshape %885 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %887 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %888 = tosa.add %839, %887 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %889 = tosa.reshape %888 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %890 = tosa.matmul %886, %889 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %891 = tosa.reshape %890 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %892 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %893 = tosa.transpose %891, %892 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %894 = tosa.identity %893 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %895 = tosa.reshape %894 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %896 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %897 = tosa.transpose %arg74, %896 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %898 = tosa.reshape %895 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_160 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %899 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%898, %897 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_160 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %900 = tosa.reshape %899 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %901 = tosa.add %803, %900 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %902 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_161 = arith.constant 2 : i32
    %903 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%901 : tensor<1x40x4096xf32>) outs(%902 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_161 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %904 = tosa.reduce_sum %903 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %905 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %906 = tosa.reciprocal %905 : (tensor<1xf32>) -> tensor<1xf32>
    %907 = tosa.mul %906, %904 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %908 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %909 = tosa.add %907, %908 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %910 = tosa.rsqrt %909 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %911 = tosa.mul %901, %910 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %912 = tosa.reshape %arg75 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %913 = tosa.mul %912, %911 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %914 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %915 = tosa.transpose %arg76, %914 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %916 = tosa.reshape %913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_162 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %917 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%916, %915 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_162 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %918 = tosa.reshape %917 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %919 = tosa.sigmoid %918 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %920 = tosa.mul %918, %919 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %921 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %922 = tosa.transpose %arg77, %921 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %923 = tosa.reshape %913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_163 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %924 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%923, %922 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_163 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %925 = tosa.reshape %924 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %926 = tosa.mul %920, %925 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %927 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %928 = tosa.transpose %arg78, %927 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %929 = tosa.reshape %926 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_164 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %930 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%929, %928 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_164 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %931 = tosa.reshape %930 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %932 = tosa.add %901, %931 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %933 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_165 = arith.constant 2 : i32
    %934 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%932 : tensor<1x40x4096xf32>) outs(%933 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_165 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %935 = tosa.reduce_sum %934 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %936 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %937 = tosa.reciprocal %936 : (tensor<1xf32>) -> tensor<1xf32>
    %938 = tosa.mul %937, %935 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %939 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %940 = tosa.add %938, %939 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %941 = tosa.rsqrt %940 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %942 = tosa.mul %932, %941 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %943 = tosa.reshape %arg79 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %944 = tosa.mul %943, %942 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %945 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %946 = tosa.transpose %arg80, %945 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %947 = tosa.reshape %944 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_166 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %948 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%947, %946 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_166 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %949 = tosa.reshape %948 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %950 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %951 = tosa.transpose %arg81, %950 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %952 = tosa.reshape %944 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_167 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %953 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%952, %951 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_167 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %954 = tosa.reshape %953 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %955 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %956 = tosa.transpose %arg82, %955 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %957 = tosa.reshape %944 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_168 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %958 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%957, %956 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_168 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %959 = tosa.reshape %958 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %960 = tosa.reshape %949 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %961 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %962 = tosa.transpose %960, %961 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %963 = tosa.reshape %954 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %964 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %965 = tosa.transpose %963, %964 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %966 = tosa.reshape %959 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %967 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %968 = tosa.transpose %966, %967 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_169 = tensor.extract_slice %arg83[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_170 = tensor.extract_slice %extracted_slice_169[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_171 = tensor.extract_slice %extracted_slice_170[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_172 = tensor.extract_slice %arg84[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_173 = tensor.extract_slice %extracted_slice_172[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_174 = tensor.extract_slice %extracted_slice_173[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %969 = tensor.empty() : tensor<1x40x128xf32>
    %970 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_171 : tensor<1x1x40x128xf32>) outs(%969 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %971 = tensor.empty() : tensor<40x128xf32>
    %972 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%970 : tensor<1x40x128xf32>) outs(%971 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %973 = tensor.empty() : tensor<1x40x128xf32>
    %974 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_174 : tensor<1x1x40x128xf32>) outs(%973 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %975 = tensor.empty() : tensor<40x128xf32>
    %976 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%974 : tensor<1x40x128xf32>) outs(%975 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %977 = tensor.empty() : tensor<1x40x128xf32>
    %978 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%977 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %972[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %979 = tosa.reshape %978 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %980 = tensor.empty() : tensor<1x40x128xf32>
    %981 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%980 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %976[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %982 = tosa.reshape %981 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %983 = tosa.mul %962, %979 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_175 = tensor.extract_slice %962[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_176 = tensor.extract_slice %962[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %984 = tosa.negate %extracted_slice_176 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %985 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_177 = tensor.insert_slice %984 into %985[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_178 = tensor.insert_slice %extracted_slice_175 into %inserted_slice_177[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %986 = tosa.mul %inserted_slice_178, %982 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %987 = tosa.add %983, %986 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %988 = tosa.mul %965, %979 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_179 = tensor.extract_slice %965[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_180 = tensor.extract_slice %965[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %989 = tosa.negate %extracted_slice_180 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %990 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_181 = tensor.insert_slice %989 into %990[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_182 = tensor.insert_slice %extracted_slice_179 into %inserted_slice_181[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %991 = tosa.mul %inserted_slice_182, %982 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %992 = tosa.add %988, %991 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %993 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %994 = tosa.transpose %992, %993 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %995 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %996 = tosa.add %987, %995 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %997 = tosa.reshape %996 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %998 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %999 = tosa.add %994, %998 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1000 = tosa.reshape %999 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1001 = tosa.matmul %997, %1000 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1002 = tosa.reshape %1001 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1003 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1004 = tosa.reciprocal %1003 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1005 = tosa.mul %1002, %1004 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1006 = tosa.add %1005, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1007 = tosa.reduce_max %1006 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1008 = tosa.sub %1006, %1007 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1009 = tosa.exp %1008 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1010 = tosa.reduce_sum %1009 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1011 = tosa.reciprocal %1010 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1012 = tosa.mul %1009, %1011 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1013 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1014 = tosa.add %1012, %1013 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1015 = tosa.reshape %1014 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1016 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1017 = tosa.add %968, %1016 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1018 = tosa.reshape %1017 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1019 = tosa.matmul %1015, %1018 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1020 = tosa.reshape %1019 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1021 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1022 = tosa.transpose %1020, %1021 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1023 = tosa.identity %1022 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1024 = tosa.reshape %1023 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1025 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1026 = tosa.transpose %arg85, %1025 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1027 = tosa.reshape %1024 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_183 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1028 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1027, %1026 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_183 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1029 = tosa.reshape %1028 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1030 = tosa.add %932, %1029 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1031 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_184 = arith.constant 2 : i32
    %1032 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1030 : tensor<1x40x4096xf32>) outs(%1031 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_184 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1033 = tosa.reduce_sum %1032 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1034 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1035 = tosa.reciprocal %1034 : (tensor<1xf32>) -> tensor<1xf32>
    %1036 = tosa.mul %1035, %1033 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1037 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1038 = tosa.add %1036, %1037 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1039 = tosa.rsqrt %1038 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1040 = tosa.mul %1030, %1039 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1041 = tosa.reshape %arg86 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1042 = tosa.mul %1041, %1040 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1043 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1044 = tosa.transpose %arg87, %1043 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1045 = tosa.reshape %1042 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_185 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1046 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1045, %1044 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_185 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1047 = tosa.reshape %1046 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1048 = tosa.sigmoid %1047 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1049 = tosa.mul %1047, %1048 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1050 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1051 = tosa.transpose %arg88, %1050 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1052 = tosa.reshape %1042 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_186 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1053 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1052, %1051 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_186 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1054 = tosa.reshape %1053 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1055 = tosa.mul %1049, %1054 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1056 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1057 = tosa.transpose %arg89, %1056 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1058 = tosa.reshape %1055 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_187 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1059 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1058, %1057 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_187 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1060 = tosa.reshape %1059 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1061 = tosa.add %1030, %1060 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1062 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_188 = arith.constant 2 : i32
    %1063 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1061 : tensor<1x40x4096xf32>) outs(%1062 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_188 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1064 = tosa.reduce_sum %1063 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1065 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1066 = tosa.reciprocal %1065 : (tensor<1xf32>) -> tensor<1xf32>
    %1067 = tosa.mul %1066, %1064 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1068 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1069 = tosa.add %1067, %1068 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1070 = tosa.rsqrt %1069 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1071 = tosa.mul %1061, %1070 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1072 = tosa.reshape %arg90 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1073 = tosa.mul %1072, %1071 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1074 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1075 = tosa.transpose %arg91, %1074 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1076 = tosa.reshape %1073 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_189 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1077 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1076, %1075 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_189 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1078 = tosa.reshape %1077 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1079 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1080 = tosa.transpose %arg92, %1079 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1081 = tosa.reshape %1073 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_190 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1082 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1081, %1080 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_190 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1083 = tosa.reshape %1082 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1084 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1085 = tosa.transpose %arg93, %1084 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1086 = tosa.reshape %1073 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_191 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1087 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1086, %1085 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_191 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1088 = tosa.reshape %1087 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1089 = tosa.reshape %1078 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1090 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1091 = tosa.transpose %1089, %1090 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1092 = tosa.reshape %1083 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1093 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1094 = tosa.transpose %1092, %1093 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1095 = tosa.reshape %1088 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1096 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1097 = tosa.transpose %1095, %1096 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_192 = tensor.extract_slice %arg94[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_193 = tensor.extract_slice %extracted_slice_192[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_194 = tensor.extract_slice %extracted_slice_193[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_195 = tensor.extract_slice %arg95[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_196 = tensor.extract_slice %extracted_slice_195[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_197 = tensor.extract_slice %extracted_slice_196[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1098 = tensor.empty() : tensor<1x40x128xf32>
    %1099 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_194 : tensor<1x1x40x128xf32>) outs(%1098 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1100 = tensor.empty() : tensor<40x128xf32>
    %1101 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1099 : tensor<1x40x128xf32>) outs(%1100 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1102 = tensor.empty() : tensor<1x40x128xf32>
    %1103 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_197 : tensor<1x1x40x128xf32>) outs(%1102 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1104 = tensor.empty() : tensor<40x128xf32>
    %1105 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1103 : tensor<1x40x128xf32>) outs(%1104 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1106 = tensor.empty() : tensor<1x40x128xf32>
    %1107 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1106 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1101[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1108 = tosa.reshape %1107 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1109 = tensor.empty() : tensor<1x40x128xf32>
    %1110 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1109 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1105[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1111 = tosa.reshape %1110 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1112 = tosa.mul %1091, %1108 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_198 = tensor.extract_slice %1091[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_199 = tensor.extract_slice %1091[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1113 = tosa.negate %extracted_slice_199 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1114 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_200 = tensor.insert_slice %1113 into %1114[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_201 = tensor.insert_slice %extracted_slice_198 into %inserted_slice_200[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1115 = tosa.mul %inserted_slice_201, %1111 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1116 = tosa.add %1112, %1115 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1117 = tosa.mul %1094, %1108 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_202 = tensor.extract_slice %1094[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_203 = tensor.extract_slice %1094[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1118 = tosa.negate %extracted_slice_203 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1119 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_204 = tensor.insert_slice %1118 into %1119[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_205 = tensor.insert_slice %extracted_slice_202 into %inserted_slice_204[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1120 = tosa.mul %inserted_slice_205, %1111 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1121 = tosa.add %1117, %1120 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1122 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1123 = tosa.transpose %1121, %1122 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1124 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1125 = tosa.add %1116, %1124 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1126 = tosa.reshape %1125 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1127 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1128 = tosa.add %1123, %1127 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1129 = tosa.reshape %1128 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1130 = tosa.matmul %1126, %1129 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1131 = tosa.reshape %1130 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1132 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1133 = tosa.reciprocal %1132 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1134 = tosa.mul %1131, %1133 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1135 = tosa.add %1134, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1136 = tosa.reduce_max %1135 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1137 = tosa.sub %1135, %1136 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1138 = tosa.exp %1137 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1139 = tosa.reduce_sum %1138 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1140 = tosa.reciprocal %1139 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1141 = tosa.mul %1138, %1140 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1142 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1143 = tosa.add %1141, %1142 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1144 = tosa.reshape %1143 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1145 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1146 = tosa.add %1097, %1145 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1147 = tosa.reshape %1146 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1148 = tosa.matmul %1144, %1147 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1149 = tosa.reshape %1148 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1150 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1151 = tosa.transpose %1149, %1150 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1152 = tosa.identity %1151 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1153 = tosa.reshape %1152 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1154 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1155 = tosa.transpose %arg96, %1154 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1156 = tosa.reshape %1153 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_206 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1157 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1156, %1155 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_206 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1158 = tosa.reshape %1157 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1159 = tosa.add %1061, %1158 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1160 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_207 = arith.constant 2 : i32
    %1161 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1159 : tensor<1x40x4096xf32>) outs(%1160 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_207 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1162 = tosa.reduce_sum %1161 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1163 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1164 = tosa.reciprocal %1163 : (tensor<1xf32>) -> tensor<1xf32>
    %1165 = tosa.mul %1164, %1162 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1166 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1167 = tosa.add %1165, %1166 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1168 = tosa.rsqrt %1167 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1169 = tosa.mul %1159, %1168 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1170 = tosa.reshape %arg97 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1171 = tosa.mul %1170, %1169 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1172 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1173 = tosa.transpose %arg98, %1172 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1174 = tosa.reshape %1171 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_208 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1175 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1174, %1173 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_208 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1176 = tosa.reshape %1175 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1177 = tosa.sigmoid %1176 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1178 = tosa.mul %1176, %1177 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1179 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1180 = tosa.transpose %arg99, %1179 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1181 = tosa.reshape %1171 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_209 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1182 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1181, %1180 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_209 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1183 = tosa.reshape %1182 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1184 = tosa.mul %1178, %1183 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1185 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1186 = tosa.transpose %arg100, %1185 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1187 = tosa.reshape %1184 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_210 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1188 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1187, %1186 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_210 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1189 = tosa.reshape %1188 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1190 = tosa.add %1159, %1189 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1191 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_211 = arith.constant 2 : i32
    %1192 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1190 : tensor<1x40x4096xf32>) outs(%1191 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_211 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1193 = tosa.reduce_sum %1192 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1194 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1195 = tosa.reciprocal %1194 : (tensor<1xf32>) -> tensor<1xf32>
    %1196 = tosa.mul %1195, %1193 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1197 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1198 = tosa.add %1196, %1197 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1199 = tosa.rsqrt %1198 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1200 = tosa.mul %1190, %1199 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1201 = tosa.reshape %arg101 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1202 = tosa.mul %1201, %1200 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1203 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1204 = tosa.transpose %arg102, %1203 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1205 = tosa.reshape %1202 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_212 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1206 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1205, %1204 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_212 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1207 = tosa.reshape %1206 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1208 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1209 = tosa.transpose %arg103, %1208 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1210 = tosa.reshape %1202 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_213 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1211 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1210, %1209 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_213 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1212 = tosa.reshape %1211 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1213 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1214 = tosa.transpose %arg104, %1213 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1215 = tosa.reshape %1202 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_214 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1216 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1215, %1214 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_214 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1217 = tosa.reshape %1216 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1218 = tosa.reshape %1207 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1219 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1220 = tosa.transpose %1218, %1219 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1221 = tosa.reshape %1212 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1222 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1223 = tosa.transpose %1221, %1222 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1224 = tosa.reshape %1217 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1225 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1226 = tosa.transpose %1224, %1225 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_215 = tensor.extract_slice %arg105[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_216 = tensor.extract_slice %extracted_slice_215[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_217 = tensor.extract_slice %extracted_slice_216[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_218 = tensor.extract_slice %arg106[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_219 = tensor.extract_slice %extracted_slice_218[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_220 = tensor.extract_slice %extracted_slice_219[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1227 = tensor.empty() : tensor<1x40x128xf32>
    %1228 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_217 : tensor<1x1x40x128xf32>) outs(%1227 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1229 = tensor.empty() : tensor<40x128xf32>
    %1230 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1228 : tensor<1x40x128xf32>) outs(%1229 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1231 = tensor.empty() : tensor<1x40x128xf32>
    %1232 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_220 : tensor<1x1x40x128xf32>) outs(%1231 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1233 = tensor.empty() : tensor<40x128xf32>
    %1234 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1232 : tensor<1x40x128xf32>) outs(%1233 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1235 = tensor.empty() : tensor<1x40x128xf32>
    %1236 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1235 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1230[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1237 = tosa.reshape %1236 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1238 = tensor.empty() : tensor<1x40x128xf32>
    %1239 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1238 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1234[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1240 = tosa.reshape %1239 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1241 = tosa.mul %1220, %1237 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_221 = tensor.extract_slice %1220[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_222 = tensor.extract_slice %1220[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1242 = tosa.negate %extracted_slice_222 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1243 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_223 = tensor.insert_slice %1242 into %1243[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_224 = tensor.insert_slice %extracted_slice_221 into %inserted_slice_223[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1244 = tosa.mul %inserted_slice_224, %1240 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1245 = tosa.add %1241, %1244 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1246 = tosa.mul %1223, %1237 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_225 = tensor.extract_slice %1223[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_226 = tensor.extract_slice %1223[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1247 = tosa.negate %extracted_slice_226 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1248 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_227 = tensor.insert_slice %1247 into %1248[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_228 = tensor.insert_slice %extracted_slice_225 into %inserted_slice_227[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1249 = tosa.mul %inserted_slice_228, %1240 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1250 = tosa.add %1246, %1249 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1251 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1252 = tosa.transpose %1250, %1251 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1253 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1254 = tosa.add %1245, %1253 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1255 = tosa.reshape %1254 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1256 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1257 = tosa.add %1252, %1256 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1258 = tosa.reshape %1257 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1259 = tosa.matmul %1255, %1258 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1260 = tosa.reshape %1259 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1261 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1262 = tosa.reciprocal %1261 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1263 = tosa.mul %1260, %1262 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1264 = tosa.add %1263, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1265 = tosa.reduce_max %1264 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1266 = tosa.sub %1264, %1265 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1267 = tosa.exp %1266 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1268 = tosa.reduce_sum %1267 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1269 = tosa.reciprocal %1268 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1270 = tosa.mul %1267, %1269 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1271 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1272 = tosa.add %1270, %1271 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1273 = tosa.reshape %1272 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1274 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1275 = tosa.add %1226, %1274 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1276 = tosa.reshape %1275 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1277 = tosa.matmul %1273, %1276 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1278 = tosa.reshape %1277 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1279 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1280 = tosa.transpose %1278, %1279 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1281 = tosa.identity %1280 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1282 = tosa.reshape %1281 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1283 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1284 = tosa.transpose %arg107, %1283 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1285 = tosa.reshape %1282 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_229 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1286 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1285, %1284 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_229 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1287 = tosa.reshape %1286 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1288 = tosa.add %1190, %1287 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1289 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_230 = arith.constant 2 : i32
    %1290 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1288 : tensor<1x40x4096xf32>) outs(%1289 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_230 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1291 = tosa.reduce_sum %1290 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1292 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1293 = tosa.reciprocal %1292 : (tensor<1xf32>) -> tensor<1xf32>
    %1294 = tosa.mul %1293, %1291 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1295 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1296 = tosa.add %1294, %1295 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1297 = tosa.rsqrt %1296 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1298 = tosa.mul %1288, %1297 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1299 = tosa.reshape %arg108 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1300 = tosa.mul %1299, %1298 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1301 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1302 = tosa.transpose %arg109, %1301 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1303 = tosa.reshape %1300 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_231 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1304 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1303, %1302 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_231 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1305 = tosa.reshape %1304 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1306 = tosa.sigmoid %1305 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1307 = tosa.mul %1305, %1306 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1308 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1309 = tosa.transpose %arg110, %1308 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1310 = tosa.reshape %1300 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_232 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1311 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1310, %1309 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_232 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1312 = tosa.reshape %1311 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1313 = tosa.mul %1307, %1312 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1314 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1315 = tosa.transpose %arg111, %1314 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1316 = tosa.reshape %1313 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_233 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1317 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1316, %1315 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_233 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1318 = tosa.reshape %1317 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1319 = tosa.add %1288, %1318 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1320 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_234 = arith.constant 2 : i32
    %1321 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1319 : tensor<1x40x4096xf32>) outs(%1320 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_234 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1322 = tosa.reduce_sum %1321 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1323 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1324 = tosa.reciprocal %1323 : (tensor<1xf32>) -> tensor<1xf32>
    %1325 = tosa.mul %1324, %1322 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1326 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1327 = tosa.add %1325, %1326 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1328 = tosa.rsqrt %1327 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1329 = tosa.mul %1319, %1328 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1330 = tosa.reshape %arg112 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1331 = tosa.mul %1330, %1329 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1332 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1333 = tosa.transpose %arg113, %1332 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1334 = tosa.reshape %1331 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_235 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1335 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1334, %1333 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_235 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1336 = tosa.reshape %1335 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1337 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1338 = tosa.transpose %arg114, %1337 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1339 = tosa.reshape %1331 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_236 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1340 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1339, %1338 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_236 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1341 = tosa.reshape %1340 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1342 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1343 = tosa.transpose %arg115, %1342 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1344 = tosa.reshape %1331 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_237 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1345 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1344, %1343 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_237 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1346 = tosa.reshape %1345 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1347 = tosa.reshape %1336 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1348 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1349 = tosa.transpose %1347, %1348 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1350 = tosa.reshape %1341 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1351 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1352 = tosa.transpose %1350, %1351 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1353 = tosa.reshape %1346 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1354 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1355 = tosa.transpose %1353, %1354 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_238 = tensor.extract_slice %arg116[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_239 = tensor.extract_slice %extracted_slice_238[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_240 = tensor.extract_slice %extracted_slice_239[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_241 = tensor.extract_slice %arg117[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_242 = tensor.extract_slice %extracted_slice_241[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_243 = tensor.extract_slice %extracted_slice_242[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1356 = tensor.empty() : tensor<1x40x128xf32>
    %1357 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_240 : tensor<1x1x40x128xf32>) outs(%1356 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1358 = tensor.empty() : tensor<40x128xf32>
    %1359 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1357 : tensor<1x40x128xf32>) outs(%1358 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1360 = tensor.empty() : tensor<1x40x128xf32>
    %1361 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_243 : tensor<1x1x40x128xf32>) outs(%1360 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1362 = tensor.empty() : tensor<40x128xf32>
    %1363 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1361 : tensor<1x40x128xf32>) outs(%1362 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1364 = tensor.empty() : tensor<1x40x128xf32>
    %1365 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1364 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1359[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1366 = tosa.reshape %1365 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1367 = tensor.empty() : tensor<1x40x128xf32>
    %1368 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1367 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1363[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1369 = tosa.reshape %1368 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1370 = tosa.mul %1349, %1366 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_244 = tensor.extract_slice %1349[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_245 = tensor.extract_slice %1349[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1371 = tosa.negate %extracted_slice_245 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1372 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_246 = tensor.insert_slice %1371 into %1372[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_247 = tensor.insert_slice %extracted_slice_244 into %inserted_slice_246[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1373 = tosa.mul %inserted_slice_247, %1369 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1374 = tosa.add %1370, %1373 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1375 = tosa.mul %1352, %1366 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_248 = tensor.extract_slice %1352[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_249 = tensor.extract_slice %1352[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1376 = tosa.negate %extracted_slice_249 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1377 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_250 = tensor.insert_slice %1376 into %1377[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_251 = tensor.insert_slice %extracted_slice_248 into %inserted_slice_250[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1378 = tosa.mul %inserted_slice_251, %1369 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1379 = tosa.add %1375, %1378 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1380 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1381 = tosa.transpose %1379, %1380 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1382 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1383 = tosa.add %1374, %1382 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1384 = tosa.reshape %1383 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1385 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1386 = tosa.add %1381, %1385 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1387 = tosa.reshape %1386 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1388 = tosa.matmul %1384, %1387 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1389 = tosa.reshape %1388 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1390 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1391 = tosa.reciprocal %1390 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1392 = tosa.mul %1389, %1391 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1393 = tosa.add %1392, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1394 = tosa.reduce_max %1393 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1395 = tosa.sub %1393, %1394 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1396 = tosa.exp %1395 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1397 = tosa.reduce_sum %1396 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1398 = tosa.reciprocal %1397 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1399 = tosa.mul %1396, %1398 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1400 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1401 = tosa.add %1399, %1400 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1402 = tosa.reshape %1401 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1403 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1404 = tosa.add %1355, %1403 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1405 = tosa.reshape %1404 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1406 = tosa.matmul %1402, %1405 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1407 = tosa.reshape %1406 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1408 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1409 = tosa.transpose %1407, %1408 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1410 = tosa.identity %1409 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1411 = tosa.reshape %1410 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1412 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1413 = tosa.transpose %arg118, %1412 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1414 = tosa.reshape %1411 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_252 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1415 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1414, %1413 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_252 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1416 = tosa.reshape %1415 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1417 = tosa.add %1319, %1416 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1418 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_253 = arith.constant 2 : i32
    %1419 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1417 : tensor<1x40x4096xf32>) outs(%1418 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_253 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1420 = tosa.reduce_sum %1419 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1421 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1422 = tosa.reciprocal %1421 : (tensor<1xf32>) -> tensor<1xf32>
    %1423 = tosa.mul %1422, %1420 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1424 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1425 = tosa.add %1423, %1424 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1426 = tosa.rsqrt %1425 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1427 = tosa.mul %1417, %1426 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1428 = tosa.reshape %arg119 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1429 = tosa.mul %1428, %1427 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1430 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1431 = tosa.transpose %arg120, %1430 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1432 = tosa.reshape %1429 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_254 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1433 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1432, %1431 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_254 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1434 = tosa.reshape %1433 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1435 = tosa.sigmoid %1434 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1436 = tosa.mul %1434, %1435 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1437 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1438 = tosa.transpose %arg121, %1437 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1439 = tosa.reshape %1429 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_255 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1440 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1439, %1438 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_255 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1441 = tosa.reshape %1440 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1442 = tosa.mul %1436, %1441 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1443 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1444 = tosa.transpose %arg122, %1443 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1445 = tosa.reshape %1442 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_256 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1446 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1445, %1444 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_256 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1447 = tosa.reshape %1446 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1448 = tosa.add %1417, %1447 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1449 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_257 = arith.constant 2 : i32
    %1450 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1448 : tensor<1x40x4096xf32>) outs(%1449 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_257 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1451 = tosa.reduce_sum %1450 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1452 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1453 = tosa.reciprocal %1452 : (tensor<1xf32>) -> tensor<1xf32>
    %1454 = tosa.mul %1453, %1451 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1455 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1456 = tosa.add %1454, %1455 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1457 = tosa.rsqrt %1456 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1458 = tosa.mul %1448, %1457 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1459 = tosa.reshape %arg123 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1460 = tosa.mul %1459, %1458 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1461 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1462 = tosa.transpose %arg124, %1461 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1463 = tosa.reshape %1460 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_258 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1464 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1463, %1462 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_258 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1465 = tosa.reshape %1464 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1466 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1467 = tosa.transpose %arg125, %1466 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1468 = tosa.reshape %1460 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_259 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1469 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1468, %1467 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_259 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1470 = tosa.reshape %1469 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1471 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1472 = tosa.transpose %arg126, %1471 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1473 = tosa.reshape %1460 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_260 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1474 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1473, %1472 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_260 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1475 = tosa.reshape %1474 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1476 = tosa.reshape %1465 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1477 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1478 = tosa.transpose %1476, %1477 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1479 = tosa.reshape %1470 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1480 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1481 = tosa.transpose %1479, %1480 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1482 = tosa.reshape %1475 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1483 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1484 = tosa.transpose %1482, %1483 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_261 = tensor.extract_slice %arg127[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_262 = tensor.extract_slice %extracted_slice_261[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_263 = tensor.extract_slice %extracted_slice_262[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_264 = tensor.extract_slice %arg128[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_265 = tensor.extract_slice %extracted_slice_264[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_266 = tensor.extract_slice %extracted_slice_265[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1485 = tensor.empty() : tensor<1x40x128xf32>
    %1486 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_263 : tensor<1x1x40x128xf32>) outs(%1485 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1487 = tensor.empty() : tensor<40x128xf32>
    %1488 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1486 : tensor<1x40x128xf32>) outs(%1487 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1489 = tensor.empty() : tensor<1x40x128xf32>
    %1490 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_266 : tensor<1x1x40x128xf32>) outs(%1489 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1491 = tensor.empty() : tensor<40x128xf32>
    %1492 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1490 : tensor<1x40x128xf32>) outs(%1491 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1493 = tensor.empty() : tensor<1x40x128xf32>
    %1494 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1493 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1488[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1495 = tosa.reshape %1494 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1496 = tensor.empty() : tensor<1x40x128xf32>
    %1497 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1496 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1492[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1498 = tosa.reshape %1497 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1499 = tosa.mul %1478, %1495 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_267 = tensor.extract_slice %1478[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_268 = tensor.extract_slice %1478[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1500 = tosa.negate %extracted_slice_268 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1501 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_269 = tensor.insert_slice %1500 into %1501[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_270 = tensor.insert_slice %extracted_slice_267 into %inserted_slice_269[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1502 = tosa.mul %inserted_slice_270, %1498 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1503 = tosa.add %1499, %1502 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1504 = tosa.mul %1481, %1495 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_271 = tensor.extract_slice %1481[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_272 = tensor.extract_slice %1481[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1505 = tosa.negate %extracted_slice_272 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1506 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_273 = tensor.insert_slice %1505 into %1506[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_274 = tensor.insert_slice %extracted_slice_271 into %inserted_slice_273[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1507 = tosa.mul %inserted_slice_274, %1498 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1508 = tosa.add %1504, %1507 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1509 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1510 = tosa.transpose %1508, %1509 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1511 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1512 = tosa.add %1503, %1511 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1513 = tosa.reshape %1512 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1514 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1515 = tosa.add %1510, %1514 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1516 = tosa.reshape %1515 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1517 = tosa.matmul %1513, %1516 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1518 = tosa.reshape %1517 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1519 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1520 = tosa.reciprocal %1519 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1521 = tosa.mul %1518, %1520 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1522 = tosa.add %1521, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1523 = tosa.reduce_max %1522 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1524 = tosa.sub %1522, %1523 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1525 = tosa.exp %1524 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1526 = tosa.reduce_sum %1525 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1527 = tosa.reciprocal %1526 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1528 = tosa.mul %1525, %1527 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1529 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1530 = tosa.add %1528, %1529 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1531 = tosa.reshape %1530 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1532 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1533 = tosa.add %1484, %1532 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1534 = tosa.reshape %1533 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1535 = tosa.matmul %1531, %1534 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1536 = tosa.reshape %1535 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1537 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1538 = tosa.transpose %1536, %1537 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1539 = tosa.identity %1538 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1540 = tosa.reshape %1539 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1541 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1542 = tosa.transpose %arg129, %1541 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1543 = tosa.reshape %1540 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_275 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1544 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1543, %1542 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_275 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1545 = tosa.reshape %1544 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1546 = tosa.add %1448, %1545 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1547 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_276 = arith.constant 2 : i32
    %1548 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1546 : tensor<1x40x4096xf32>) outs(%1547 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_276 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1549 = tosa.reduce_sum %1548 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1550 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1551 = tosa.reciprocal %1550 : (tensor<1xf32>) -> tensor<1xf32>
    %1552 = tosa.mul %1551, %1549 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1553 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1554 = tosa.add %1552, %1553 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1555 = tosa.rsqrt %1554 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1556 = tosa.mul %1546, %1555 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1557 = tosa.reshape %arg130 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1558 = tosa.mul %1557, %1556 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1559 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1560 = tosa.transpose %arg131, %1559 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1561 = tosa.reshape %1558 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_277 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1562 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1561, %1560 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_277 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1563 = tosa.reshape %1562 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1564 = tosa.sigmoid %1563 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1565 = tosa.mul %1563, %1564 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1566 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1567 = tosa.transpose %arg132, %1566 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1568 = tosa.reshape %1558 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_278 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1569 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1568, %1567 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_278 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1570 = tosa.reshape %1569 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1571 = tosa.mul %1565, %1570 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1572 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1573 = tosa.transpose %arg133, %1572 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1574 = tosa.reshape %1571 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_279 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1575 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1574, %1573 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_279 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1576 = tosa.reshape %1575 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1577 = tosa.add %1546, %1576 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1578 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_280 = arith.constant 2 : i32
    %1579 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1577 : tensor<1x40x4096xf32>) outs(%1578 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_280 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1580 = tosa.reduce_sum %1579 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1581 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1582 = tosa.reciprocal %1581 : (tensor<1xf32>) -> tensor<1xf32>
    %1583 = tosa.mul %1582, %1580 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1584 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1585 = tosa.add %1583, %1584 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1586 = tosa.rsqrt %1585 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1587 = tosa.mul %1577, %1586 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1588 = tosa.reshape %arg134 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1589 = tosa.mul %1588, %1587 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1590 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1591 = tosa.transpose %arg135, %1590 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1592 = tosa.reshape %1589 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_281 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1593 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1592, %1591 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_281 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1594 = tosa.reshape %1593 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1595 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1596 = tosa.transpose %arg136, %1595 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1597 = tosa.reshape %1589 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_282 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1598 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1597, %1596 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_282 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1599 = tosa.reshape %1598 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1600 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1601 = tosa.transpose %arg137, %1600 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1602 = tosa.reshape %1589 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_283 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1603 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1602, %1601 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_283 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1604 = tosa.reshape %1603 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1605 = tosa.reshape %1594 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1606 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1607 = tosa.transpose %1605, %1606 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1608 = tosa.reshape %1599 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1609 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1610 = tosa.transpose %1608, %1609 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1611 = tosa.reshape %1604 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1612 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1613 = tosa.transpose %1611, %1612 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_284 = tensor.extract_slice %arg138[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_285 = tensor.extract_slice %extracted_slice_284[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_286 = tensor.extract_slice %extracted_slice_285[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_287 = tensor.extract_slice %arg139[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_288 = tensor.extract_slice %extracted_slice_287[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_289 = tensor.extract_slice %extracted_slice_288[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1614 = tensor.empty() : tensor<1x40x128xf32>
    %1615 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_286 : tensor<1x1x40x128xf32>) outs(%1614 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1616 = tensor.empty() : tensor<40x128xf32>
    %1617 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1615 : tensor<1x40x128xf32>) outs(%1616 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1618 = tensor.empty() : tensor<1x40x128xf32>
    %1619 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_289 : tensor<1x1x40x128xf32>) outs(%1618 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1620 = tensor.empty() : tensor<40x128xf32>
    %1621 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1619 : tensor<1x40x128xf32>) outs(%1620 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1622 = tensor.empty() : tensor<1x40x128xf32>
    %1623 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1622 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1617[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1624 = tosa.reshape %1623 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1625 = tensor.empty() : tensor<1x40x128xf32>
    %1626 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1625 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1621[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1627 = tosa.reshape %1626 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1628 = tosa.mul %1607, %1624 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_290 = tensor.extract_slice %1607[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_291 = tensor.extract_slice %1607[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1629 = tosa.negate %extracted_slice_291 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1630 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_292 = tensor.insert_slice %1629 into %1630[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_293 = tensor.insert_slice %extracted_slice_290 into %inserted_slice_292[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1631 = tosa.mul %inserted_slice_293, %1627 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1632 = tosa.add %1628, %1631 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1633 = tosa.mul %1610, %1624 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_294 = tensor.extract_slice %1610[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_295 = tensor.extract_slice %1610[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1634 = tosa.negate %extracted_slice_295 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1635 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_296 = tensor.insert_slice %1634 into %1635[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_297 = tensor.insert_slice %extracted_slice_294 into %inserted_slice_296[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1636 = tosa.mul %inserted_slice_297, %1627 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1637 = tosa.add %1633, %1636 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1638 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1639 = tosa.transpose %1637, %1638 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1640 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1641 = tosa.add %1632, %1640 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1642 = tosa.reshape %1641 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1643 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1644 = tosa.add %1639, %1643 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1645 = tosa.reshape %1644 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1646 = tosa.matmul %1642, %1645 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1647 = tosa.reshape %1646 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1648 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1649 = tosa.reciprocal %1648 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1650 = tosa.mul %1647, %1649 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1651 = tosa.add %1650, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1652 = tosa.reduce_max %1651 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1653 = tosa.sub %1651, %1652 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1654 = tosa.exp %1653 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1655 = tosa.reduce_sum %1654 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1656 = tosa.reciprocal %1655 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1657 = tosa.mul %1654, %1656 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1658 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1659 = tosa.add %1657, %1658 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1660 = tosa.reshape %1659 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1661 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1662 = tosa.add %1613, %1661 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1663 = tosa.reshape %1662 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1664 = tosa.matmul %1660, %1663 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1665 = tosa.reshape %1664 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1666 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1667 = tosa.transpose %1665, %1666 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1668 = tosa.identity %1667 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1669 = tosa.reshape %1668 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1670 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1671 = tosa.transpose %arg140, %1670 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1672 = tosa.reshape %1669 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_298 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1673 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1672, %1671 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_298 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1674 = tosa.reshape %1673 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1675 = tosa.add %1577, %1674 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1676 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_299 = arith.constant 2 : i32
    %1677 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1675 : tensor<1x40x4096xf32>) outs(%1676 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_299 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1678 = tosa.reduce_sum %1677 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1679 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1680 = tosa.reciprocal %1679 : (tensor<1xf32>) -> tensor<1xf32>
    %1681 = tosa.mul %1680, %1678 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1682 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1683 = tosa.add %1681, %1682 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1684 = tosa.rsqrt %1683 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1685 = tosa.mul %1675, %1684 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1686 = tosa.reshape %arg141 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1687 = tosa.mul %1686, %1685 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1688 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1689 = tosa.transpose %arg142, %1688 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1690 = tosa.reshape %1687 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_300 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1691 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1690, %1689 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_300 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1692 = tosa.reshape %1691 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1693 = tosa.sigmoid %1692 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1694 = tosa.mul %1692, %1693 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1695 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1696 = tosa.transpose %arg143, %1695 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1697 = tosa.reshape %1687 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_301 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1698 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1697, %1696 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_301 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1699 = tosa.reshape %1698 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1700 = tosa.mul %1694, %1699 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1701 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1702 = tosa.transpose %arg144, %1701 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1703 = tosa.reshape %1700 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_302 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1704 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1703, %1702 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_302 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1705 = tosa.reshape %1704 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1706 = tosa.add %1675, %1705 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1707 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_303 = arith.constant 2 : i32
    %1708 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1706 : tensor<1x40x4096xf32>) outs(%1707 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_303 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1709 = tosa.reduce_sum %1708 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1710 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1711 = tosa.reciprocal %1710 : (tensor<1xf32>) -> tensor<1xf32>
    %1712 = tosa.mul %1711, %1709 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1713 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1714 = tosa.add %1712, %1713 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1715 = tosa.rsqrt %1714 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1716 = tosa.mul %1706, %1715 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1717 = tosa.reshape %arg145 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1718 = tosa.mul %1717, %1716 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1719 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1720 = tosa.transpose %arg146, %1719 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1721 = tosa.reshape %1718 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_304 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1722 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1721, %1720 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_304 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1723 = tosa.reshape %1722 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1724 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1725 = tosa.transpose %arg147, %1724 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1726 = tosa.reshape %1718 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_305 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1727 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1726, %1725 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_305 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1728 = tosa.reshape %1727 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1729 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1730 = tosa.transpose %arg148, %1729 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1731 = tosa.reshape %1718 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_306 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1732 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1731, %1730 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_306 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1733 = tosa.reshape %1732 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1734 = tosa.reshape %1723 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1735 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1736 = tosa.transpose %1734, %1735 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1737 = tosa.reshape %1728 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1738 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1739 = tosa.transpose %1737, %1738 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1740 = tosa.reshape %1733 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1741 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1742 = tosa.transpose %1740, %1741 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_307 = tensor.extract_slice %arg149[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_308 = tensor.extract_slice %extracted_slice_307[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_309 = tensor.extract_slice %extracted_slice_308[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_310 = tensor.extract_slice %arg150[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_311 = tensor.extract_slice %extracted_slice_310[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_312 = tensor.extract_slice %extracted_slice_311[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1743 = tensor.empty() : tensor<1x40x128xf32>
    %1744 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_309 : tensor<1x1x40x128xf32>) outs(%1743 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1745 = tensor.empty() : tensor<40x128xf32>
    %1746 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1744 : tensor<1x40x128xf32>) outs(%1745 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1747 = tensor.empty() : tensor<1x40x128xf32>
    %1748 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_312 : tensor<1x1x40x128xf32>) outs(%1747 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1749 = tensor.empty() : tensor<40x128xf32>
    %1750 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1748 : tensor<1x40x128xf32>) outs(%1749 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1751 = tensor.empty() : tensor<1x40x128xf32>
    %1752 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1751 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1746[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1753 = tosa.reshape %1752 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1754 = tensor.empty() : tensor<1x40x128xf32>
    %1755 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1754 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1750[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1756 = tosa.reshape %1755 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1757 = tosa.mul %1736, %1753 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_313 = tensor.extract_slice %1736[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_314 = tensor.extract_slice %1736[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1758 = tosa.negate %extracted_slice_314 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1759 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_315 = tensor.insert_slice %1758 into %1759[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_316 = tensor.insert_slice %extracted_slice_313 into %inserted_slice_315[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1760 = tosa.mul %inserted_slice_316, %1756 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1761 = tosa.add %1757, %1760 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1762 = tosa.mul %1739, %1753 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_317 = tensor.extract_slice %1739[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_318 = tensor.extract_slice %1739[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1763 = tosa.negate %extracted_slice_318 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1764 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_319 = tensor.insert_slice %1763 into %1764[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_320 = tensor.insert_slice %extracted_slice_317 into %inserted_slice_319[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1765 = tosa.mul %inserted_slice_320, %1756 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1766 = tosa.add %1762, %1765 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1767 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1768 = tosa.transpose %1766, %1767 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1769 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1770 = tosa.add %1761, %1769 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1771 = tosa.reshape %1770 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1772 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1773 = tosa.add %1768, %1772 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1774 = tosa.reshape %1773 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1775 = tosa.matmul %1771, %1774 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1776 = tosa.reshape %1775 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1777 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1778 = tosa.reciprocal %1777 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1779 = tosa.mul %1776, %1778 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1780 = tosa.add %1779, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1781 = tosa.reduce_max %1780 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1782 = tosa.sub %1780, %1781 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1783 = tosa.exp %1782 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1784 = tosa.reduce_sum %1783 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1785 = tosa.reciprocal %1784 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1786 = tosa.mul %1783, %1785 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1787 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1788 = tosa.add %1786, %1787 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1789 = tosa.reshape %1788 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1790 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1791 = tosa.add %1742, %1790 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1792 = tosa.reshape %1791 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1793 = tosa.matmul %1789, %1792 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1794 = tosa.reshape %1793 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1795 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1796 = tosa.transpose %1794, %1795 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1797 = tosa.identity %1796 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1798 = tosa.reshape %1797 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1799 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1800 = tosa.transpose %arg151, %1799 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1801 = tosa.reshape %1798 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_321 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1802 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1801, %1800 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_321 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1803 = tosa.reshape %1802 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1804 = tosa.add %1706, %1803 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1805 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_322 = arith.constant 2 : i32
    %1806 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1804 : tensor<1x40x4096xf32>) outs(%1805 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_322 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1807 = tosa.reduce_sum %1806 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1808 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1809 = tosa.reciprocal %1808 : (tensor<1xf32>) -> tensor<1xf32>
    %1810 = tosa.mul %1809, %1807 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1811 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1812 = tosa.add %1810, %1811 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1813 = tosa.rsqrt %1812 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1814 = tosa.mul %1804, %1813 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1815 = tosa.reshape %arg152 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1816 = tosa.mul %1815, %1814 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1817 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1818 = tosa.transpose %arg153, %1817 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1819 = tosa.reshape %1816 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_323 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1820 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1819, %1818 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_323 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1821 = tosa.reshape %1820 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1822 = tosa.sigmoid %1821 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1823 = tosa.mul %1821, %1822 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1824 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1825 = tosa.transpose %arg154, %1824 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1826 = tosa.reshape %1816 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_324 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1827 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1826, %1825 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_324 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1828 = tosa.reshape %1827 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1829 = tosa.mul %1823, %1828 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1830 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1831 = tosa.transpose %arg155, %1830 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1832 = tosa.reshape %1829 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_325 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1833 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1832, %1831 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_325 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1834 = tosa.reshape %1833 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1835 = tosa.add %1804, %1834 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1836 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_326 = arith.constant 2 : i32
    %1837 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1835 : tensor<1x40x4096xf32>) outs(%1836 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_326 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1838 = tosa.reduce_sum %1837 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1839 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1840 = tosa.reciprocal %1839 : (tensor<1xf32>) -> tensor<1xf32>
    %1841 = tosa.mul %1840, %1838 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1842 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1843 = tosa.add %1841, %1842 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1844 = tosa.rsqrt %1843 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1845 = tosa.mul %1835, %1844 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1846 = tosa.reshape %arg156 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1847 = tosa.mul %1846, %1845 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1848 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1849 = tosa.transpose %arg157, %1848 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1850 = tosa.reshape %1847 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_327 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1851 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1850, %1849 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_327 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1852 = tosa.reshape %1851 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1853 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1854 = tosa.transpose %arg158, %1853 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1855 = tosa.reshape %1847 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_328 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1856 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1855, %1854 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_328 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1857 = tosa.reshape %1856 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1858 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1859 = tosa.transpose %arg159, %1858 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1860 = tosa.reshape %1847 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_329 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1861 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1860, %1859 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_329 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1862 = tosa.reshape %1861 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1863 = tosa.reshape %1852 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1864 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1865 = tosa.transpose %1863, %1864 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1866 = tosa.reshape %1857 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1867 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1868 = tosa.transpose %1866, %1867 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1869 = tosa.reshape %1862 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1870 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1871 = tosa.transpose %1869, %1870 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_330 = tensor.extract_slice %arg160[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_331 = tensor.extract_slice %extracted_slice_330[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_332 = tensor.extract_slice %extracted_slice_331[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_333 = tensor.extract_slice %arg161[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_334 = tensor.extract_slice %extracted_slice_333[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_335 = tensor.extract_slice %extracted_slice_334[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1872 = tensor.empty() : tensor<1x40x128xf32>
    %1873 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_332 : tensor<1x1x40x128xf32>) outs(%1872 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1874 = tensor.empty() : tensor<40x128xf32>
    %1875 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1873 : tensor<1x40x128xf32>) outs(%1874 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1876 = tensor.empty() : tensor<1x40x128xf32>
    %1877 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_335 : tensor<1x1x40x128xf32>) outs(%1876 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1878 = tensor.empty() : tensor<40x128xf32>
    %1879 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1877 : tensor<1x40x128xf32>) outs(%1878 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1880 = tensor.empty() : tensor<1x40x128xf32>
    %1881 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1880 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1875[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1882 = tosa.reshape %1881 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1883 = tensor.empty() : tensor<1x40x128xf32>
    %1884 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1883 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %1879[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1885 = tosa.reshape %1884 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1886 = tosa.mul %1865, %1882 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_336 = tensor.extract_slice %1865[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_337 = tensor.extract_slice %1865[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1887 = tosa.negate %extracted_slice_337 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1888 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_338 = tensor.insert_slice %1887 into %1888[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_339 = tensor.insert_slice %extracted_slice_336 into %inserted_slice_338[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1889 = tosa.mul %inserted_slice_339, %1885 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1890 = tosa.add %1886, %1889 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1891 = tosa.mul %1868, %1882 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_340 = tensor.extract_slice %1868[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_341 = tensor.extract_slice %1868[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1892 = tosa.negate %extracted_slice_341 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1893 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_342 = tensor.insert_slice %1892 into %1893[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_343 = tensor.insert_slice %extracted_slice_340 into %inserted_slice_342[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1894 = tosa.mul %inserted_slice_343, %1885 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1895 = tosa.add %1891, %1894 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1896 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1897 = tosa.transpose %1895, %1896 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1898 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1899 = tosa.add %1890, %1898 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1900 = tosa.reshape %1899 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1901 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1902 = tosa.add %1897, %1901 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1903 = tosa.reshape %1902 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1904 = tosa.matmul %1900, %1903 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1905 = tosa.reshape %1904 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1906 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1907 = tosa.reciprocal %1906 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1908 = tosa.mul %1905, %1907 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1909 = tosa.add %1908, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1910 = tosa.reduce_max %1909 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1911 = tosa.sub %1909, %1910 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1912 = tosa.exp %1911 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1913 = tosa.reduce_sum %1912 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1914 = tosa.reciprocal %1913 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1915 = tosa.mul %1912, %1914 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1916 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1917 = tosa.add %1915, %1916 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1918 = tosa.reshape %1917 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1919 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1920 = tosa.add %1871, %1919 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1921 = tosa.reshape %1920 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1922 = tosa.matmul %1918, %1921 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1923 = tosa.reshape %1922 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1924 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1925 = tosa.transpose %1923, %1924 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1926 = tosa.identity %1925 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1927 = tosa.reshape %1926 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1928 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1929 = tosa.transpose %arg162, %1928 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1930 = tosa.reshape %1927 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_344 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1931 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1930, %1929 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_344 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1932 = tosa.reshape %1931 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1933 = tosa.add %1835, %1932 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1934 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_345 = arith.constant 2 : i32
    %1935 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1933 : tensor<1x40x4096xf32>) outs(%1934 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_345 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1936 = tosa.reduce_sum %1935 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1937 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1938 = tosa.reciprocal %1937 : (tensor<1xf32>) -> tensor<1xf32>
    %1939 = tosa.mul %1938, %1936 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1940 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1941 = tosa.add %1939, %1940 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1942 = tosa.rsqrt %1941 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1943 = tosa.mul %1933, %1942 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1944 = tosa.reshape %arg163 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1945 = tosa.mul %1944, %1943 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1946 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1947 = tosa.transpose %arg164, %1946 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1948 = tosa.reshape %1945 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_346 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1949 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1948, %1947 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_346 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1950 = tosa.reshape %1949 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1951 = tosa.sigmoid %1950 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1952 = tosa.mul %1950, %1951 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1953 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1954 = tosa.transpose %arg165, %1953 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1955 = tosa.reshape %1945 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_347 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1956 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1955, %1954 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_347 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1957 = tosa.reshape %1956 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1958 = tosa.mul %1952, %1957 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1959 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1960 = tosa.transpose %arg166, %1959 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1961 = tosa.reshape %1958 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_348 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1962 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1961, %1960 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_348 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1963 = tosa.reshape %1962 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1964 = tosa.add %1933, %1963 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1965 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_349 = arith.constant 2 : i32
    %1966 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1964 : tensor<1x40x4096xf32>) outs(%1965 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_349 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %1967 = tosa.reduce_sum %1966 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %1968 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1969 = tosa.reciprocal %1968 : (tensor<1xf32>) -> tensor<1xf32>
    %1970 = tosa.mul %1969, %1967 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1971 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1972 = tosa.add %1970, %1971 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1973 = tosa.rsqrt %1972 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1974 = tosa.mul %1964, %1973 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1975 = tosa.reshape %arg167 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1976 = tosa.mul %1975, %1974 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1977 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1978 = tosa.transpose %arg168, %1977 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1979 = tosa.reshape %1976 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_350 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1980 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1979, %1978 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_350 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1981 = tosa.reshape %1980 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1982 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1983 = tosa.transpose %arg169, %1982 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1984 = tosa.reshape %1976 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_351 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1985 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1984, %1983 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_351 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1986 = tosa.reshape %1985 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1987 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1988 = tosa.transpose %arg170, %1987 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1989 = tosa.reshape %1976 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_352 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1990 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1989, %1988 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_352 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1991 = tosa.reshape %1990 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1992 = tosa.reshape %1981 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1993 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1994 = tosa.transpose %1992, %1993 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1995 = tosa.reshape %1986 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1996 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1997 = tosa.transpose %1995, %1996 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1998 = tosa.reshape %1991 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1999 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2000 = tosa.transpose %1998, %1999 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_353 = tensor.extract_slice %arg171[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_354 = tensor.extract_slice %extracted_slice_353[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_355 = tensor.extract_slice %extracted_slice_354[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_356 = tensor.extract_slice %arg172[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_357 = tensor.extract_slice %extracted_slice_356[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_358 = tensor.extract_slice %extracted_slice_357[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2001 = tensor.empty() : tensor<1x40x128xf32>
    %2002 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_355 : tensor<1x1x40x128xf32>) outs(%2001 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2003 = tensor.empty() : tensor<40x128xf32>
    %2004 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2002 : tensor<1x40x128xf32>) outs(%2003 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2005 = tensor.empty() : tensor<1x40x128xf32>
    %2006 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_358 : tensor<1x1x40x128xf32>) outs(%2005 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2007 = tensor.empty() : tensor<40x128xf32>
    %2008 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2006 : tensor<1x40x128xf32>) outs(%2007 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2009 = tensor.empty() : tensor<1x40x128xf32>
    %2010 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2009 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2004[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2011 = tosa.reshape %2010 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2012 = tensor.empty() : tensor<1x40x128xf32>
    %2013 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2012 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2008[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2014 = tosa.reshape %2013 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2015 = tosa.mul %1994, %2011 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_359 = tensor.extract_slice %1994[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_360 = tensor.extract_slice %1994[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2016 = tosa.negate %extracted_slice_360 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2017 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_361 = tensor.insert_slice %2016 into %2017[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_362 = tensor.insert_slice %extracted_slice_359 into %inserted_slice_361[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2018 = tosa.mul %inserted_slice_362, %2014 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2019 = tosa.add %2015, %2018 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2020 = tosa.mul %1997, %2011 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_363 = tensor.extract_slice %1997[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_364 = tensor.extract_slice %1997[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2021 = tosa.negate %extracted_slice_364 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2022 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_365 = tensor.insert_slice %2021 into %2022[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_366 = tensor.insert_slice %extracted_slice_363 into %inserted_slice_365[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2023 = tosa.mul %inserted_slice_366, %2014 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2024 = tosa.add %2020, %2023 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2025 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2026 = tosa.transpose %2024, %2025 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2027 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2028 = tosa.add %2019, %2027 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2029 = tosa.reshape %2028 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2030 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2031 = tosa.add %2026, %2030 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2032 = tosa.reshape %2031 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2033 = tosa.matmul %2029, %2032 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2034 = tosa.reshape %2033 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2035 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2036 = tosa.reciprocal %2035 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2037 = tosa.mul %2034, %2036 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2038 = tosa.add %2037, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2039 = tosa.reduce_max %2038 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2040 = tosa.sub %2038, %2039 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2041 = tosa.exp %2040 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2042 = tosa.reduce_sum %2041 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2043 = tosa.reciprocal %2042 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2044 = tosa.mul %2041, %2043 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2045 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2046 = tosa.add %2044, %2045 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2047 = tosa.reshape %2046 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2048 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2049 = tosa.add %2000, %2048 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2050 = tosa.reshape %2049 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2051 = tosa.matmul %2047, %2050 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2052 = tosa.reshape %2051 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2053 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2054 = tosa.transpose %2052, %2053 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2055 = tosa.identity %2054 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2056 = tosa.reshape %2055 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2057 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2058 = tosa.transpose %arg173, %2057 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2059 = tosa.reshape %2056 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_367 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2060 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2059, %2058 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_367 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2061 = tosa.reshape %2060 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2062 = tosa.add %1964, %2061 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2063 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_368 = arith.constant 2 : i32
    %2064 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2062 : tensor<1x40x4096xf32>) outs(%2063 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_368 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2065 = tosa.reduce_sum %2064 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2066 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2067 = tosa.reciprocal %2066 : (tensor<1xf32>) -> tensor<1xf32>
    %2068 = tosa.mul %2067, %2065 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2069 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2070 = tosa.add %2068, %2069 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2071 = tosa.rsqrt %2070 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2072 = tosa.mul %2062, %2071 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2073 = tosa.reshape %arg174 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2074 = tosa.mul %2073, %2072 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2075 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2076 = tosa.transpose %arg175, %2075 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2077 = tosa.reshape %2074 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_369 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2078 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2077, %2076 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_369 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2079 = tosa.reshape %2078 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2080 = tosa.sigmoid %2079 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2081 = tosa.mul %2079, %2080 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2082 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2083 = tosa.transpose %arg176, %2082 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2084 = tosa.reshape %2074 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_370 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2085 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2084, %2083 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_370 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2086 = tosa.reshape %2085 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2087 = tosa.mul %2081, %2086 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2088 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2089 = tosa.transpose %arg177, %2088 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2090 = tosa.reshape %2087 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_371 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2091 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2090, %2089 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_371 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2092 = tosa.reshape %2091 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2093 = tosa.add %2062, %2092 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2094 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_372 = arith.constant 2 : i32
    %2095 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2093 : tensor<1x40x4096xf32>) outs(%2094 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_372 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2096 = tosa.reduce_sum %2095 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2097 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2098 = tosa.reciprocal %2097 : (tensor<1xf32>) -> tensor<1xf32>
    %2099 = tosa.mul %2098, %2096 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2100 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2101 = tosa.add %2099, %2100 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2102 = tosa.rsqrt %2101 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2103 = tosa.mul %2093, %2102 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2104 = tosa.reshape %arg178 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2105 = tosa.mul %2104, %2103 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2106 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2107 = tosa.transpose %arg179, %2106 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2108 = tosa.reshape %2105 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_373 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2109 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2108, %2107 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_373 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2110 = tosa.reshape %2109 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2111 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2112 = tosa.transpose %arg180, %2111 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2113 = tosa.reshape %2105 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_374 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2114 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2113, %2112 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_374 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2115 = tosa.reshape %2114 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2116 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2117 = tosa.transpose %arg181, %2116 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2118 = tosa.reshape %2105 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_375 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2119 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2118, %2117 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_375 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2120 = tosa.reshape %2119 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2121 = tosa.reshape %2110 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2122 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2123 = tosa.transpose %2121, %2122 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2124 = tosa.reshape %2115 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2125 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2126 = tosa.transpose %2124, %2125 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2127 = tosa.reshape %2120 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2128 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2129 = tosa.transpose %2127, %2128 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_376 = tensor.extract_slice %arg182[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_377 = tensor.extract_slice %extracted_slice_376[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_378 = tensor.extract_slice %extracted_slice_377[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_379 = tensor.extract_slice %arg183[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_380 = tensor.extract_slice %extracted_slice_379[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_381 = tensor.extract_slice %extracted_slice_380[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2130 = tensor.empty() : tensor<1x40x128xf32>
    %2131 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_378 : tensor<1x1x40x128xf32>) outs(%2130 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2132 = tensor.empty() : tensor<40x128xf32>
    %2133 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2131 : tensor<1x40x128xf32>) outs(%2132 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2134 = tensor.empty() : tensor<1x40x128xf32>
    %2135 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_381 : tensor<1x1x40x128xf32>) outs(%2134 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2136 = tensor.empty() : tensor<40x128xf32>
    %2137 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2135 : tensor<1x40x128xf32>) outs(%2136 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2138 = tensor.empty() : tensor<1x40x128xf32>
    %2139 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2138 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2133[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2140 = tosa.reshape %2139 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2141 = tensor.empty() : tensor<1x40x128xf32>
    %2142 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2141 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2137[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2143 = tosa.reshape %2142 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2144 = tosa.mul %2123, %2140 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_382 = tensor.extract_slice %2123[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_383 = tensor.extract_slice %2123[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2145 = tosa.negate %extracted_slice_383 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2146 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_384 = tensor.insert_slice %2145 into %2146[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_385 = tensor.insert_slice %extracted_slice_382 into %inserted_slice_384[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2147 = tosa.mul %inserted_slice_385, %2143 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2148 = tosa.add %2144, %2147 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2149 = tosa.mul %2126, %2140 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_386 = tensor.extract_slice %2126[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_387 = tensor.extract_slice %2126[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2150 = tosa.negate %extracted_slice_387 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2151 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_388 = tensor.insert_slice %2150 into %2151[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_389 = tensor.insert_slice %extracted_slice_386 into %inserted_slice_388[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2152 = tosa.mul %inserted_slice_389, %2143 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2153 = tosa.add %2149, %2152 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2154 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2155 = tosa.transpose %2153, %2154 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2156 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2157 = tosa.add %2148, %2156 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2158 = tosa.reshape %2157 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2159 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2160 = tosa.add %2155, %2159 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2161 = tosa.reshape %2160 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2162 = tosa.matmul %2158, %2161 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2163 = tosa.reshape %2162 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2164 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2165 = tosa.reciprocal %2164 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2166 = tosa.mul %2163, %2165 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2167 = tosa.add %2166, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2168 = tosa.reduce_max %2167 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2169 = tosa.sub %2167, %2168 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2170 = tosa.exp %2169 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2171 = tosa.reduce_sum %2170 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2172 = tosa.reciprocal %2171 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2173 = tosa.mul %2170, %2172 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2174 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2175 = tosa.add %2173, %2174 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2176 = tosa.reshape %2175 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2177 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2178 = tosa.add %2129, %2177 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2179 = tosa.reshape %2178 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2180 = tosa.matmul %2176, %2179 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2181 = tosa.reshape %2180 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2182 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2183 = tosa.transpose %2181, %2182 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2184 = tosa.identity %2183 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2185 = tosa.reshape %2184 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2186 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2187 = tosa.transpose %arg184, %2186 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2188 = tosa.reshape %2185 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_390 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2189 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2188, %2187 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_390 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2190 = tosa.reshape %2189 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2191 = tosa.add %2093, %2190 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2192 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_391 = arith.constant 2 : i32
    %2193 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2191 : tensor<1x40x4096xf32>) outs(%2192 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_391 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2194 = tosa.reduce_sum %2193 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2195 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2196 = tosa.reciprocal %2195 : (tensor<1xf32>) -> tensor<1xf32>
    %2197 = tosa.mul %2196, %2194 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2198 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2199 = tosa.add %2197, %2198 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2200 = tosa.rsqrt %2199 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2201 = tosa.mul %2191, %2200 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2202 = tosa.reshape %arg185 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2203 = tosa.mul %2202, %2201 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2204 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2205 = tosa.transpose %arg186, %2204 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2206 = tosa.reshape %2203 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_392 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2207 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2206, %2205 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_392 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2208 = tosa.reshape %2207 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2209 = tosa.sigmoid %2208 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2210 = tosa.mul %2208, %2209 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2211 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2212 = tosa.transpose %arg187, %2211 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2213 = tosa.reshape %2203 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_393 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2214 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2213, %2212 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_393 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2215 = tosa.reshape %2214 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2216 = tosa.mul %2210, %2215 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2217 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2218 = tosa.transpose %arg188, %2217 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2219 = tosa.reshape %2216 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_394 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2220 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2219, %2218 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_394 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2221 = tosa.reshape %2220 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2222 = tosa.add %2191, %2221 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2223 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_395 = arith.constant 2 : i32
    %2224 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2222 : tensor<1x40x4096xf32>) outs(%2223 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_395 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2225 = tosa.reduce_sum %2224 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2226 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2227 = tosa.reciprocal %2226 : (tensor<1xf32>) -> tensor<1xf32>
    %2228 = tosa.mul %2227, %2225 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2229 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2230 = tosa.add %2228, %2229 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2231 = tosa.rsqrt %2230 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2232 = tosa.mul %2222, %2231 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2233 = tosa.reshape %arg189 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2234 = tosa.mul %2233, %2232 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2235 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2236 = tosa.transpose %arg190, %2235 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2237 = tosa.reshape %2234 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_396 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2238 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2237, %2236 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_396 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2239 = tosa.reshape %2238 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2240 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2241 = tosa.transpose %arg191, %2240 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2242 = tosa.reshape %2234 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_397 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2243 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2242, %2241 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_397 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2244 = tosa.reshape %2243 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2245 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2246 = tosa.transpose %arg192, %2245 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2247 = tosa.reshape %2234 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_398 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2248 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2247, %2246 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_398 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2249 = tosa.reshape %2248 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2250 = tosa.reshape %2239 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2251 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2252 = tosa.transpose %2250, %2251 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2253 = tosa.reshape %2244 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2254 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2255 = tosa.transpose %2253, %2254 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2256 = tosa.reshape %2249 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2257 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2258 = tosa.transpose %2256, %2257 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_399 = tensor.extract_slice %arg193[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_400 = tensor.extract_slice %extracted_slice_399[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_401 = tensor.extract_slice %extracted_slice_400[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_402 = tensor.extract_slice %arg194[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_403 = tensor.extract_slice %extracted_slice_402[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_404 = tensor.extract_slice %extracted_slice_403[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2259 = tensor.empty() : tensor<1x40x128xf32>
    %2260 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_401 : tensor<1x1x40x128xf32>) outs(%2259 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2261 = tensor.empty() : tensor<40x128xf32>
    %2262 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2260 : tensor<1x40x128xf32>) outs(%2261 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2263 = tensor.empty() : tensor<1x40x128xf32>
    %2264 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_404 : tensor<1x1x40x128xf32>) outs(%2263 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2265 = tensor.empty() : tensor<40x128xf32>
    %2266 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2264 : tensor<1x40x128xf32>) outs(%2265 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2267 = tensor.empty() : tensor<1x40x128xf32>
    %2268 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2267 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2262[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2269 = tosa.reshape %2268 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2270 = tensor.empty() : tensor<1x40x128xf32>
    %2271 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2270 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2266[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2272 = tosa.reshape %2271 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2273 = tosa.mul %2252, %2269 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_405 = tensor.extract_slice %2252[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_406 = tensor.extract_slice %2252[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2274 = tosa.negate %extracted_slice_406 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2275 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_407 = tensor.insert_slice %2274 into %2275[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_408 = tensor.insert_slice %extracted_slice_405 into %inserted_slice_407[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2276 = tosa.mul %inserted_slice_408, %2272 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2277 = tosa.add %2273, %2276 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2278 = tosa.mul %2255, %2269 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_409 = tensor.extract_slice %2255[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_410 = tensor.extract_slice %2255[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2279 = tosa.negate %extracted_slice_410 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2280 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_411 = tensor.insert_slice %2279 into %2280[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_412 = tensor.insert_slice %extracted_slice_409 into %inserted_slice_411[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2281 = tosa.mul %inserted_slice_412, %2272 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2282 = tosa.add %2278, %2281 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2283 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2284 = tosa.transpose %2282, %2283 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2285 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2286 = tosa.add %2277, %2285 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2287 = tosa.reshape %2286 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2288 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2289 = tosa.add %2284, %2288 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2290 = tosa.reshape %2289 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2291 = tosa.matmul %2287, %2290 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2292 = tosa.reshape %2291 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2293 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2294 = tosa.reciprocal %2293 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2295 = tosa.mul %2292, %2294 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2296 = tosa.add %2295, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2297 = tosa.reduce_max %2296 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2298 = tosa.sub %2296, %2297 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2299 = tosa.exp %2298 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2300 = tosa.reduce_sum %2299 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2301 = tosa.reciprocal %2300 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2302 = tosa.mul %2299, %2301 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2303 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2304 = tosa.add %2302, %2303 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2305 = tosa.reshape %2304 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2306 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2307 = tosa.add %2258, %2306 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2308 = tosa.reshape %2307 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2309 = tosa.matmul %2305, %2308 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2310 = tosa.reshape %2309 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2311 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2312 = tosa.transpose %2310, %2311 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2313 = tosa.identity %2312 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2314 = tosa.reshape %2313 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2315 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2316 = tosa.transpose %arg195, %2315 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2317 = tosa.reshape %2314 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_413 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2318 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2317, %2316 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_413 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2319 = tosa.reshape %2318 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2320 = tosa.add %2222, %2319 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2321 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_414 = arith.constant 2 : i32
    %2322 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2320 : tensor<1x40x4096xf32>) outs(%2321 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_414 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2323 = tosa.reduce_sum %2322 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2324 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2325 = tosa.reciprocal %2324 : (tensor<1xf32>) -> tensor<1xf32>
    %2326 = tosa.mul %2325, %2323 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2327 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2328 = tosa.add %2326, %2327 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2329 = tosa.rsqrt %2328 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2330 = tosa.mul %2320, %2329 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2331 = tosa.reshape %arg196 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2332 = tosa.mul %2331, %2330 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2333 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2334 = tosa.transpose %arg197, %2333 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2335 = tosa.reshape %2332 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_415 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2336 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2335, %2334 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_415 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2337 = tosa.reshape %2336 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2338 = tosa.sigmoid %2337 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2339 = tosa.mul %2337, %2338 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2340 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2341 = tosa.transpose %arg198, %2340 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2342 = tosa.reshape %2332 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_416 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2343 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2342, %2341 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_416 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2344 = tosa.reshape %2343 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2345 = tosa.mul %2339, %2344 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2346 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2347 = tosa.transpose %arg199, %2346 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2348 = tosa.reshape %2345 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_417 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2349 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2348, %2347 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_417 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2350 = tosa.reshape %2349 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2351 = tosa.add %2320, %2350 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2352 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_418 = arith.constant 2 : i32
    %2353 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2351 : tensor<1x40x4096xf32>) outs(%2352 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_418 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2354 = tosa.reduce_sum %2353 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2355 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2356 = tosa.reciprocal %2355 : (tensor<1xf32>) -> tensor<1xf32>
    %2357 = tosa.mul %2356, %2354 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2358 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2359 = tosa.add %2357, %2358 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2360 = tosa.rsqrt %2359 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2361 = tosa.mul %2351, %2360 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2362 = tosa.reshape %arg200 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2363 = tosa.mul %2362, %2361 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2364 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2365 = tosa.transpose %arg201, %2364 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2366 = tosa.reshape %2363 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_419 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2367 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2366, %2365 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_419 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2368 = tosa.reshape %2367 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2369 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2370 = tosa.transpose %arg202, %2369 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2371 = tosa.reshape %2363 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_420 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2372 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2371, %2370 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_420 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2373 = tosa.reshape %2372 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2374 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2375 = tosa.transpose %arg203, %2374 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2376 = tosa.reshape %2363 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_421 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2377 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2376, %2375 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_421 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2378 = tosa.reshape %2377 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2379 = tosa.reshape %2368 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2380 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2381 = tosa.transpose %2379, %2380 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2382 = tosa.reshape %2373 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2383 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2384 = tosa.transpose %2382, %2383 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2385 = tosa.reshape %2378 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2386 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2387 = tosa.transpose %2385, %2386 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_422 = tensor.extract_slice %arg204[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_423 = tensor.extract_slice %extracted_slice_422[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_424 = tensor.extract_slice %extracted_slice_423[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_425 = tensor.extract_slice %arg205[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_426 = tensor.extract_slice %extracted_slice_425[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_427 = tensor.extract_slice %extracted_slice_426[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2388 = tensor.empty() : tensor<1x40x128xf32>
    %2389 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_424 : tensor<1x1x40x128xf32>) outs(%2388 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2390 = tensor.empty() : tensor<40x128xf32>
    %2391 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2389 : tensor<1x40x128xf32>) outs(%2390 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2392 = tensor.empty() : tensor<1x40x128xf32>
    %2393 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_427 : tensor<1x1x40x128xf32>) outs(%2392 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2394 = tensor.empty() : tensor<40x128xf32>
    %2395 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2393 : tensor<1x40x128xf32>) outs(%2394 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2396 = tensor.empty() : tensor<1x40x128xf32>
    %2397 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2396 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2391[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2398 = tosa.reshape %2397 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2399 = tensor.empty() : tensor<1x40x128xf32>
    %2400 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2399 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2395[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2401 = tosa.reshape %2400 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2402 = tosa.mul %2381, %2398 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_428 = tensor.extract_slice %2381[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_429 = tensor.extract_slice %2381[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2403 = tosa.negate %extracted_slice_429 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2404 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_430 = tensor.insert_slice %2403 into %2404[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_431 = tensor.insert_slice %extracted_slice_428 into %inserted_slice_430[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2405 = tosa.mul %inserted_slice_431, %2401 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2406 = tosa.add %2402, %2405 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2407 = tosa.mul %2384, %2398 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_432 = tensor.extract_slice %2384[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_433 = tensor.extract_slice %2384[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2408 = tosa.negate %extracted_slice_433 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2409 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_434 = tensor.insert_slice %2408 into %2409[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_435 = tensor.insert_slice %extracted_slice_432 into %inserted_slice_434[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2410 = tosa.mul %inserted_slice_435, %2401 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2411 = tosa.add %2407, %2410 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2412 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2413 = tosa.transpose %2411, %2412 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2414 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2415 = tosa.add %2406, %2414 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2416 = tosa.reshape %2415 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2417 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2418 = tosa.add %2413, %2417 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2419 = tosa.reshape %2418 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2420 = tosa.matmul %2416, %2419 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2421 = tosa.reshape %2420 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2422 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2423 = tosa.reciprocal %2422 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2424 = tosa.mul %2421, %2423 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2425 = tosa.add %2424, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2426 = tosa.reduce_max %2425 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2427 = tosa.sub %2425, %2426 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2428 = tosa.exp %2427 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2429 = tosa.reduce_sum %2428 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2430 = tosa.reciprocal %2429 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2431 = tosa.mul %2428, %2430 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2432 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2433 = tosa.add %2431, %2432 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2434 = tosa.reshape %2433 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2435 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2436 = tosa.add %2387, %2435 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2437 = tosa.reshape %2436 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2438 = tosa.matmul %2434, %2437 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2439 = tosa.reshape %2438 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2440 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2441 = tosa.transpose %2439, %2440 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2442 = tosa.identity %2441 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2443 = tosa.reshape %2442 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2444 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2445 = tosa.transpose %arg206, %2444 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2446 = tosa.reshape %2443 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_436 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2447 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2446, %2445 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_436 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2448 = tosa.reshape %2447 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2449 = tosa.add %2351, %2448 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2450 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_437 = arith.constant 2 : i32
    %2451 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2449 : tensor<1x40x4096xf32>) outs(%2450 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_437 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2452 = tosa.reduce_sum %2451 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2453 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2454 = tosa.reciprocal %2453 : (tensor<1xf32>) -> tensor<1xf32>
    %2455 = tosa.mul %2454, %2452 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2456 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2457 = tosa.add %2455, %2456 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2458 = tosa.rsqrt %2457 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2459 = tosa.mul %2449, %2458 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2460 = tosa.reshape %arg207 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2461 = tosa.mul %2460, %2459 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2462 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2463 = tosa.transpose %arg208, %2462 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2464 = tosa.reshape %2461 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_438 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2465 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2464, %2463 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_438 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2466 = tosa.reshape %2465 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2467 = tosa.sigmoid %2466 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2468 = tosa.mul %2466, %2467 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2469 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2470 = tosa.transpose %arg209, %2469 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2471 = tosa.reshape %2461 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_439 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2472 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2471, %2470 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_439 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2473 = tosa.reshape %2472 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2474 = tosa.mul %2468, %2473 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2475 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2476 = tosa.transpose %arg210, %2475 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2477 = tosa.reshape %2474 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_440 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2478 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2477, %2476 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_440 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2479 = tosa.reshape %2478 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2480 = tosa.add %2449, %2479 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2481 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_441 = arith.constant 2 : i32
    %2482 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2480 : tensor<1x40x4096xf32>) outs(%2481 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_441 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2483 = tosa.reduce_sum %2482 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2484 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2485 = tosa.reciprocal %2484 : (tensor<1xf32>) -> tensor<1xf32>
    %2486 = tosa.mul %2485, %2483 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2487 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2488 = tosa.add %2486, %2487 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2489 = tosa.rsqrt %2488 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2490 = tosa.mul %2480, %2489 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2491 = tosa.reshape %arg211 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2492 = tosa.mul %2491, %2490 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2493 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2494 = tosa.transpose %arg212, %2493 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2495 = tosa.reshape %2492 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_442 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2496 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2495, %2494 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_442 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2497 = tosa.reshape %2496 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2498 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2499 = tosa.transpose %arg213, %2498 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2500 = tosa.reshape %2492 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_443 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2501 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2500, %2499 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_443 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2502 = tosa.reshape %2501 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2503 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2504 = tosa.transpose %arg214, %2503 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2505 = tosa.reshape %2492 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_444 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2506 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2505, %2504 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_444 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2507 = tosa.reshape %2506 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2508 = tosa.reshape %2497 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2509 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2510 = tosa.transpose %2508, %2509 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2511 = tosa.reshape %2502 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2512 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2513 = tosa.transpose %2511, %2512 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2514 = tosa.reshape %2507 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2515 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2516 = tosa.transpose %2514, %2515 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_445 = tensor.extract_slice %arg215[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_446 = tensor.extract_slice %extracted_slice_445[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_447 = tensor.extract_slice %extracted_slice_446[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_448 = tensor.extract_slice %arg216[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_449 = tensor.extract_slice %extracted_slice_448[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_450 = tensor.extract_slice %extracted_slice_449[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2517 = tensor.empty() : tensor<1x40x128xf32>
    %2518 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_447 : tensor<1x1x40x128xf32>) outs(%2517 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2519 = tensor.empty() : tensor<40x128xf32>
    %2520 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2518 : tensor<1x40x128xf32>) outs(%2519 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2521 = tensor.empty() : tensor<1x40x128xf32>
    %2522 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_450 : tensor<1x1x40x128xf32>) outs(%2521 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2523 = tensor.empty() : tensor<40x128xf32>
    %2524 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2522 : tensor<1x40x128xf32>) outs(%2523 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2525 = tensor.empty() : tensor<1x40x128xf32>
    %2526 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2525 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2520[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2527 = tosa.reshape %2526 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2528 = tensor.empty() : tensor<1x40x128xf32>
    %2529 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2528 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2524[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2530 = tosa.reshape %2529 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2531 = tosa.mul %2510, %2527 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_451 = tensor.extract_slice %2510[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_452 = tensor.extract_slice %2510[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2532 = tosa.negate %extracted_slice_452 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2533 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_453 = tensor.insert_slice %2532 into %2533[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_454 = tensor.insert_slice %extracted_slice_451 into %inserted_slice_453[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2534 = tosa.mul %inserted_slice_454, %2530 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2535 = tosa.add %2531, %2534 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2536 = tosa.mul %2513, %2527 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_455 = tensor.extract_slice %2513[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_456 = tensor.extract_slice %2513[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2537 = tosa.negate %extracted_slice_456 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2538 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_457 = tensor.insert_slice %2537 into %2538[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_458 = tensor.insert_slice %extracted_slice_455 into %inserted_slice_457[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2539 = tosa.mul %inserted_slice_458, %2530 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2540 = tosa.add %2536, %2539 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2541 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2542 = tosa.transpose %2540, %2541 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2543 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2544 = tosa.add %2535, %2543 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2545 = tosa.reshape %2544 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2546 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2547 = tosa.add %2542, %2546 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2548 = tosa.reshape %2547 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2549 = tosa.matmul %2545, %2548 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2550 = tosa.reshape %2549 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2551 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2552 = tosa.reciprocal %2551 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2553 = tosa.mul %2550, %2552 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2554 = tosa.add %2553, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2555 = tosa.reduce_max %2554 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2556 = tosa.sub %2554, %2555 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2557 = tosa.exp %2556 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2558 = tosa.reduce_sum %2557 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2559 = tosa.reciprocal %2558 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2560 = tosa.mul %2557, %2559 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2561 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2562 = tosa.add %2560, %2561 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2563 = tosa.reshape %2562 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2564 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2565 = tosa.add %2516, %2564 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2566 = tosa.reshape %2565 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2567 = tosa.matmul %2563, %2566 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2568 = tosa.reshape %2567 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2569 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2570 = tosa.transpose %2568, %2569 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2571 = tosa.identity %2570 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2572 = tosa.reshape %2571 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2573 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2574 = tosa.transpose %arg217, %2573 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2575 = tosa.reshape %2572 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_459 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2576 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2575, %2574 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_459 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2577 = tosa.reshape %2576 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2578 = tosa.add %2480, %2577 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2579 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_460 = arith.constant 2 : i32
    %2580 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2578 : tensor<1x40x4096xf32>) outs(%2579 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_460 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2581 = tosa.reduce_sum %2580 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2582 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2583 = tosa.reciprocal %2582 : (tensor<1xf32>) -> tensor<1xf32>
    %2584 = tosa.mul %2583, %2581 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2585 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2586 = tosa.add %2584, %2585 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2587 = tosa.rsqrt %2586 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2588 = tosa.mul %2578, %2587 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2589 = tosa.reshape %arg218 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2590 = tosa.mul %2589, %2588 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2591 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2592 = tosa.transpose %arg219, %2591 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2593 = tosa.reshape %2590 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_461 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2594 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2593, %2592 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_461 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2595 = tosa.reshape %2594 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2596 = tosa.sigmoid %2595 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2597 = tosa.mul %2595, %2596 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2598 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2599 = tosa.transpose %arg220, %2598 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2600 = tosa.reshape %2590 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_462 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2601 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2600, %2599 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_462 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2602 = tosa.reshape %2601 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2603 = tosa.mul %2597, %2602 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2604 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2605 = tosa.transpose %arg221, %2604 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2606 = tosa.reshape %2603 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_463 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2607 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2606, %2605 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_463 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2608 = tosa.reshape %2607 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2609 = tosa.add %2578, %2608 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2610 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_464 = arith.constant 2 : i32
    %2611 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2609 : tensor<1x40x4096xf32>) outs(%2610 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_464 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2612 = tosa.reduce_sum %2611 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2613 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2614 = tosa.reciprocal %2613 : (tensor<1xf32>) -> tensor<1xf32>
    %2615 = tosa.mul %2614, %2612 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2616 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2617 = tosa.add %2615, %2616 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2618 = tosa.rsqrt %2617 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2619 = tosa.mul %2609, %2618 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2620 = tosa.reshape %arg222 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2621 = tosa.mul %2620, %2619 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2622 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2623 = tosa.transpose %arg223, %2622 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2624 = tosa.reshape %2621 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_465 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2625 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2624, %2623 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_465 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2626 = tosa.reshape %2625 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2627 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2628 = tosa.transpose %arg224, %2627 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2629 = tosa.reshape %2621 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_466 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2630 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2629, %2628 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_466 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2631 = tosa.reshape %2630 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2632 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2633 = tosa.transpose %arg225, %2632 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2634 = tosa.reshape %2621 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_467 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2635 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2634, %2633 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_467 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2636 = tosa.reshape %2635 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2637 = tosa.reshape %2626 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2638 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2639 = tosa.transpose %2637, %2638 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2640 = tosa.reshape %2631 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2641 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2642 = tosa.transpose %2640, %2641 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2643 = tosa.reshape %2636 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2644 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2645 = tosa.transpose %2643, %2644 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_468 = tensor.extract_slice %arg226[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_469 = tensor.extract_slice %extracted_slice_468[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_470 = tensor.extract_slice %extracted_slice_469[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_471 = tensor.extract_slice %arg227[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_472 = tensor.extract_slice %extracted_slice_471[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_473 = tensor.extract_slice %extracted_slice_472[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2646 = tensor.empty() : tensor<1x40x128xf32>
    %2647 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_470 : tensor<1x1x40x128xf32>) outs(%2646 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2648 = tensor.empty() : tensor<40x128xf32>
    %2649 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2647 : tensor<1x40x128xf32>) outs(%2648 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2650 = tensor.empty() : tensor<1x40x128xf32>
    %2651 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_473 : tensor<1x1x40x128xf32>) outs(%2650 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2652 = tensor.empty() : tensor<40x128xf32>
    %2653 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2651 : tensor<1x40x128xf32>) outs(%2652 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2654 = tensor.empty() : tensor<1x40x128xf32>
    %2655 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2654 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2649[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2656 = tosa.reshape %2655 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2657 = tensor.empty() : tensor<1x40x128xf32>
    %2658 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2657 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2653[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2659 = tosa.reshape %2658 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2660 = tosa.mul %2639, %2656 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_474 = tensor.extract_slice %2639[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_475 = tensor.extract_slice %2639[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2661 = tosa.negate %extracted_slice_475 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2662 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_476 = tensor.insert_slice %2661 into %2662[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_477 = tensor.insert_slice %extracted_slice_474 into %inserted_slice_476[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2663 = tosa.mul %inserted_slice_477, %2659 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2664 = tosa.add %2660, %2663 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2665 = tosa.mul %2642, %2656 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_478 = tensor.extract_slice %2642[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_479 = tensor.extract_slice %2642[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2666 = tosa.negate %extracted_slice_479 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2667 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_480 = tensor.insert_slice %2666 into %2667[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_481 = tensor.insert_slice %extracted_slice_478 into %inserted_slice_480[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2668 = tosa.mul %inserted_slice_481, %2659 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2669 = tosa.add %2665, %2668 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2670 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2671 = tosa.transpose %2669, %2670 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2672 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2673 = tosa.add %2664, %2672 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2674 = tosa.reshape %2673 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2675 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2676 = tosa.add %2671, %2675 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2677 = tosa.reshape %2676 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2678 = tosa.matmul %2674, %2677 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2679 = tosa.reshape %2678 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2680 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2681 = tosa.reciprocal %2680 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2682 = tosa.mul %2679, %2681 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2683 = tosa.add %2682, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2684 = tosa.reduce_max %2683 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2685 = tosa.sub %2683, %2684 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2686 = tosa.exp %2685 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2687 = tosa.reduce_sum %2686 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2688 = tosa.reciprocal %2687 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2689 = tosa.mul %2686, %2688 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2690 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2691 = tosa.add %2689, %2690 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2692 = tosa.reshape %2691 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2693 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2694 = tosa.add %2645, %2693 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2695 = tosa.reshape %2694 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2696 = tosa.matmul %2692, %2695 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2697 = tosa.reshape %2696 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2698 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2699 = tosa.transpose %2697, %2698 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2700 = tosa.identity %2699 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2701 = tosa.reshape %2700 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2702 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2703 = tosa.transpose %arg228, %2702 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2704 = tosa.reshape %2701 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_482 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2705 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2704, %2703 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_482 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2706 = tosa.reshape %2705 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2707 = tosa.add %2609, %2706 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2708 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_483 = arith.constant 2 : i32
    %2709 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2707 : tensor<1x40x4096xf32>) outs(%2708 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_483 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2710 = tosa.reduce_sum %2709 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2711 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2712 = tosa.reciprocal %2711 : (tensor<1xf32>) -> tensor<1xf32>
    %2713 = tosa.mul %2712, %2710 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2714 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2715 = tosa.add %2713, %2714 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2716 = tosa.rsqrt %2715 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2717 = tosa.mul %2707, %2716 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2718 = tosa.reshape %arg229 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2719 = tosa.mul %2718, %2717 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2720 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2721 = tosa.transpose %arg230, %2720 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2722 = tosa.reshape %2719 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_484 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2723 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2722, %2721 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_484 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2724 = tosa.reshape %2723 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2725 = tosa.sigmoid %2724 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2726 = tosa.mul %2724, %2725 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2727 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2728 = tosa.transpose %arg231, %2727 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2729 = tosa.reshape %2719 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_485 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2730 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2729, %2728 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_485 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2731 = tosa.reshape %2730 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2732 = tosa.mul %2726, %2731 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2733 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2734 = tosa.transpose %arg232, %2733 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2735 = tosa.reshape %2732 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_486 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2736 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2735, %2734 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_486 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2737 = tosa.reshape %2736 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2738 = tosa.add %2707, %2737 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2739 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_487 = arith.constant 2 : i32
    %2740 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2738 : tensor<1x40x4096xf32>) outs(%2739 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_487 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2741 = tosa.reduce_sum %2740 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2742 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2743 = tosa.reciprocal %2742 : (tensor<1xf32>) -> tensor<1xf32>
    %2744 = tosa.mul %2743, %2741 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2745 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2746 = tosa.add %2744, %2745 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2747 = tosa.rsqrt %2746 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2748 = tosa.mul %2738, %2747 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2749 = tosa.reshape %arg233 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2750 = tosa.mul %2749, %2748 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2751 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2752 = tosa.transpose %arg234, %2751 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2753 = tosa.reshape %2750 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_488 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2754 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2753, %2752 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_488 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2755 = tosa.reshape %2754 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2756 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2757 = tosa.transpose %arg235, %2756 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2758 = tosa.reshape %2750 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_489 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2759 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2758, %2757 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_489 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2760 = tosa.reshape %2759 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2761 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2762 = tosa.transpose %arg236, %2761 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2763 = tosa.reshape %2750 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_490 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2764 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2763, %2762 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_490 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2765 = tosa.reshape %2764 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2766 = tosa.reshape %2755 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2767 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2768 = tosa.transpose %2766, %2767 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2769 = tosa.reshape %2760 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2770 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2771 = tosa.transpose %2769, %2770 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2772 = tosa.reshape %2765 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2773 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2774 = tosa.transpose %2772, %2773 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_491 = tensor.extract_slice %arg237[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_492 = tensor.extract_slice %extracted_slice_491[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_493 = tensor.extract_slice %extracted_slice_492[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_494 = tensor.extract_slice %arg238[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_495 = tensor.extract_slice %extracted_slice_494[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_496 = tensor.extract_slice %extracted_slice_495[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2775 = tensor.empty() : tensor<1x40x128xf32>
    %2776 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_493 : tensor<1x1x40x128xf32>) outs(%2775 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2777 = tensor.empty() : tensor<40x128xf32>
    %2778 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2776 : tensor<1x40x128xf32>) outs(%2777 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2779 = tensor.empty() : tensor<1x40x128xf32>
    %2780 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_496 : tensor<1x1x40x128xf32>) outs(%2779 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2781 = tensor.empty() : tensor<40x128xf32>
    %2782 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2780 : tensor<1x40x128xf32>) outs(%2781 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2783 = tensor.empty() : tensor<1x40x128xf32>
    %2784 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2783 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2778[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2785 = tosa.reshape %2784 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2786 = tensor.empty() : tensor<1x40x128xf32>
    %2787 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2786 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2782[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2788 = tosa.reshape %2787 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2789 = tosa.mul %2768, %2785 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_497 = tensor.extract_slice %2768[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_498 = tensor.extract_slice %2768[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2790 = tosa.negate %extracted_slice_498 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2791 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_499 = tensor.insert_slice %2790 into %2791[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_500 = tensor.insert_slice %extracted_slice_497 into %inserted_slice_499[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2792 = tosa.mul %inserted_slice_500, %2788 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2793 = tosa.add %2789, %2792 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2794 = tosa.mul %2771, %2785 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_501 = tensor.extract_slice %2771[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_502 = tensor.extract_slice %2771[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2795 = tosa.negate %extracted_slice_502 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2796 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_503 = tensor.insert_slice %2795 into %2796[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_504 = tensor.insert_slice %extracted_slice_501 into %inserted_slice_503[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2797 = tosa.mul %inserted_slice_504, %2788 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2798 = tosa.add %2794, %2797 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2799 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2800 = tosa.transpose %2798, %2799 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2801 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2802 = tosa.add %2793, %2801 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2803 = tosa.reshape %2802 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2804 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2805 = tosa.add %2800, %2804 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2806 = tosa.reshape %2805 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2807 = tosa.matmul %2803, %2806 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2808 = tosa.reshape %2807 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2809 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2810 = tosa.reciprocal %2809 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2811 = tosa.mul %2808, %2810 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2812 = tosa.add %2811, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2813 = tosa.reduce_max %2812 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2814 = tosa.sub %2812, %2813 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2815 = tosa.exp %2814 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2816 = tosa.reduce_sum %2815 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2817 = tosa.reciprocal %2816 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2818 = tosa.mul %2815, %2817 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2819 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2820 = tosa.add %2818, %2819 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2821 = tosa.reshape %2820 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2822 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2823 = tosa.add %2774, %2822 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2824 = tosa.reshape %2823 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2825 = tosa.matmul %2821, %2824 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2826 = tosa.reshape %2825 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2827 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2828 = tosa.transpose %2826, %2827 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2829 = tosa.identity %2828 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2830 = tosa.reshape %2829 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2831 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2832 = tosa.transpose %arg239, %2831 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2833 = tosa.reshape %2830 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_505 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2834 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2833, %2832 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_505 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2835 = tosa.reshape %2834 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2836 = tosa.add %2738, %2835 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2837 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_506 = arith.constant 2 : i32
    %2838 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2836 : tensor<1x40x4096xf32>) outs(%2837 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_506 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2839 = tosa.reduce_sum %2838 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2840 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2841 = tosa.reciprocal %2840 : (tensor<1xf32>) -> tensor<1xf32>
    %2842 = tosa.mul %2841, %2839 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2843 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2844 = tosa.add %2842, %2843 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2845 = tosa.rsqrt %2844 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2846 = tosa.mul %2836, %2845 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2847 = tosa.reshape %arg240 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2848 = tosa.mul %2847, %2846 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2849 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2850 = tosa.transpose %arg241, %2849 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2851 = tosa.reshape %2848 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_507 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2852 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2851, %2850 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_507 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2853 = tosa.reshape %2852 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2854 = tosa.sigmoid %2853 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2855 = tosa.mul %2853, %2854 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2856 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2857 = tosa.transpose %arg242, %2856 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2858 = tosa.reshape %2848 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_508 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2859 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2858, %2857 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_508 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2860 = tosa.reshape %2859 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2861 = tosa.mul %2855, %2860 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2862 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2863 = tosa.transpose %arg243, %2862 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2864 = tosa.reshape %2861 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_509 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2865 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2864, %2863 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_509 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2866 = tosa.reshape %2865 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2867 = tosa.add %2836, %2866 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2868 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_510 = arith.constant 2 : i32
    %2869 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2867 : tensor<1x40x4096xf32>) outs(%2868 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_510 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2870 = tosa.reduce_sum %2869 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2871 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2872 = tosa.reciprocal %2871 : (tensor<1xf32>) -> tensor<1xf32>
    %2873 = tosa.mul %2872, %2870 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2874 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2875 = tosa.add %2873, %2874 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2876 = tosa.rsqrt %2875 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2877 = tosa.mul %2867, %2876 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2878 = tosa.reshape %arg244 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2879 = tosa.mul %2878, %2877 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2880 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2881 = tosa.transpose %arg245, %2880 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2882 = tosa.reshape %2879 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_511 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2883 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2882, %2881 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_511 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2884 = tosa.reshape %2883 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2885 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2886 = tosa.transpose %arg246, %2885 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2887 = tosa.reshape %2879 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_512 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2888 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2887, %2886 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_512 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2889 = tosa.reshape %2888 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2890 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2891 = tosa.transpose %arg247, %2890 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2892 = tosa.reshape %2879 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_513 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2893 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2892, %2891 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_513 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2894 = tosa.reshape %2893 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2895 = tosa.reshape %2884 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2896 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2897 = tosa.transpose %2895, %2896 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2898 = tosa.reshape %2889 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2899 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2900 = tosa.transpose %2898, %2899 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2901 = tosa.reshape %2894 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2902 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2903 = tosa.transpose %2901, %2902 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_514 = tensor.extract_slice %arg248[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_515 = tensor.extract_slice %extracted_slice_514[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_516 = tensor.extract_slice %extracted_slice_515[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_517 = tensor.extract_slice %arg249[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_518 = tensor.extract_slice %extracted_slice_517[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_519 = tensor.extract_slice %extracted_slice_518[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2904 = tensor.empty() : tensor<1x40x128xf32>
    %2905 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_516 : tensor<1x1x40x128xf32>) outs(%2904 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2906 = tensor.empty() : tensor<40x128xf32>
    %2907 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2905 : tensor<1x40x128xf32>) outs(%2906 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2908 = tensor.empty() : tensor<1x40x128xf32>
    %2909 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_519 : tensor<1x1x40x128xf32>) outs(%2908 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2910 = tensor.empty() : tensor<40x128xf32>
    %2911 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2909 : tensor<1x40x128xf32>) outs(%2910 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2912 = tensor.empty() : tensor<1x40x128xf32>
    %2913 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2912 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2907[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2914 = tosa.reshape %2913 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2915 = tensor.empty() : tensor<1x40x128xf32>
    %2916 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2915 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %2911[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2917 = tosa.reshape %2916 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2918 = tosa.mul %2897, %2914 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_520 = tensor.extract_slice %2897[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_521 = tensor.extract_slice %2897[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2919 = tosa.negate %extracted_slice_521 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2920 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_522 = tensor.insert_slice %2919 into %2920[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_523 = tensor.insert_slice %extracted_slice_520 into %inserted_slice_522[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2921 = tosa.mul %inserted_slice_523, %2917 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2922 = tosa.add %2918, %2921 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2923 = tosa.mul %2900, %2914 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_524 = tensor.extract_slice %2900[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_525 = tensor.extract_slice %2900[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2924 = tosa.negate %extracted_slice_525 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2925 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_526 = tensor.insert_slice %2924 into %2925[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_527 = tensor.insert_slice %extracted_slice_524 into %inserted_slice_526[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2926 = tosa.mul %inserted_slice_527, %2917 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2927 = tosa.add %2923, %2926 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2928 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2929 = tosa.transpose %2927, %2928 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2930 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2931 = tosa.add %2922, %2930 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2932 = tosa.reshape %2931 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2933 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2934 = tosa.add %2929, %2933 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2935 = tosa.reshape %2934 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2936 = tosa.matmul %2932, %2935 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2937 = tosa.reshape %2936 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2938 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2939 = tosa.reciprocal %2938 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2940 = tosa.mul %2937, %2939 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2941 = tosa.add %2940, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2942 = tosa.reduce_max %2941 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2943 = tosa.sub %2941, %2942 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2944 = tosa.exp %2943 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2945 = tosa.reduce_sum %2944 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2946 = tosa.reciprocal %2945 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2947 = tosa.mul %2944, %2946 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2948 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2949 = tosa.add %2947, %2948 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2950 = tosa.reshape %2949 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2951 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2952 = tosa.add %2903, %2951 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2953 = tosa.reshape %2952 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2954 = tosa.matmul %2950, %2953 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2955 = tosa.reshape %2954 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2956 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2957 = tosa.transpose %2955, %2956 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2958 = tosa.identity %2957 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2959 = tosa.reshape %2958 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2960 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2961 = tosa.transpose %arg250, %2960 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2962 = tosa.reshape %2959 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_528 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2963 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2962, %2961 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_528 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2964 = tosa.reshape %2963 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2965 = tosa.add %2867, %2964 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2966 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_529 = arith.constant 2 : i32
    %2967 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2965 : tensor<1x40x4096xf32>) outs(%2966 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_529 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2968 = tosa.reduce_sum %2967 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %2969 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2970 = tosa.reciprocal %2969 : (tensor<1xf32>) -> tensor<1xf32>
    %2971 = tosa.mul %2970, %2968 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2972 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2973 = tosa.add %2971, %2972 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2974 = tosa.rsqrt %2973 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2975 = tosa.mul %2965, %2974 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2976 = tosa.reshape %arg251 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2977 = tosa.mul %2976, %2975 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2978 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2979 = tosa.transpose %arg252, %2978 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2980 = tosa.reshape %2977 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_530 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2981 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2980, %2979 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_530 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2982 = tosa.reshape %2981 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2983 = tosa.sigmoid %2982 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2984 = tosa.mul %2982, %2983 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2985 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2986 = tosa.transpose %arg253, %2985 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2987 = tosa.reshape %2977 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_531 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2988 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2987, %2986 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_531 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2989 = tosa.reshape %2988 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2990 = tosa.mul %2984, %2989 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2991 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2992 = tosa.transpose %arg254, %2991 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2993 = tosa.reshape %2990 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_532 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2994 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2993, %2992 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_532 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2995 = tosa.reshape %2994 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2996 = tosa.add %2965, %2995 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2997 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_533 = arith.constant 2 : i32
    %2998 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2996 : tensor<1x40x4096xf32>) outs(%2997 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_533 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %2999 = tosa.reduce_sum %2998 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3000 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3001 = tosa.reciprocal %3000 : (tensor<1xf32>) -> tensor<1xf32>
    %3002 = tosa.mul %3001, %2999 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3003 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3004 = tosa.add %3002, %3003 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3005 = tosa.rsqrt %3004 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3006 = tosa.mul %2996, %3005 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3007 = tosa.reshape %arg255 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3008 = tosa.mul %3007, %3006 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3009 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3010 = tosa.transpose %arg256, %3009 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3011 = tosa.reshape %3008 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_534 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3012 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3011, %3010 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_534 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3013 = tosa.reshape %3012 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3014 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3015 = tosa.transpose %arg257, %3014 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3016 = tosa.reshape %3008 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_535 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3017 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3016, %3015 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_535 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3018 = tosa.reshape %3017 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3019 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3020 = tosa.transpose %arg258, %3019 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3021 = tosa.reshape %3008 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_536 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3022 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3021, %3020 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_536 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3023 = tosa.reshape %3022 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3024 = tosa.reshape %3013 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3025 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3026 = tosa.transpose %3024, %3025 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3027 = tosa.reshape %3018 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3028 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3029 = tosa.transpose %3027, %3028 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3030 = tosa.reshape %3023 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3031 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3032 = tosa.transpose %3030, %3031 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_537 = tensor.extract_slice %arg259[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_538 = tensor.extract_slice %extracted_slice_537[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_539 = tensor.extract_slice %extracted_slice_538[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_540 = tensor.extract_slice %arg260[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_541 = tensor.extract_slice %extracted_slice_540[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_542 = tensor.extract_slice %extracted_slice_541[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3033 = tensor.empty() : tensor<1x40x128xf32>
    %3034 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_539 : tensor<1x1x40x128xf32>) outs(%3033 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3035 = tensor.empty() : tensor<40x128xf32>
    %3036 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3034 : tensor<1x40x128xf32>) outs(%3035 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3037 = tensor.empty() : tensor<1x40x128xf32>
    %3038 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_542 : tensor<1x1x40x128xf32>) outs(%3037 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3039 = tensor.empty() : tensor<40x128xf32>
    %3040 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3038 : tensor<1x40x128xf32>) outs(%3039 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3041 = tensor.empty() : tensor<1x40x128xf32>
    %3042 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3041 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3036[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3043 = tosa.reshape %3042 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3044 = tensor.empty() : tensor<1x40x128xf32>
    %3045 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3044 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3040[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3046 = tosa.reshape %3045 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3047 = tosa.mul %3026, %3043 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_543 = tensor.extract_slice %3026[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_544 = tensor.extract_slice %3026[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3048 = tosa.negate %extracted_slice_544 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3049 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_545 = tensor.insert_slice %3048 into %3049[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_546 = tensor.insert_slice %extracted_slice_543 into %inserted_slice_545[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3050 = tosa.mul %inserted_slice_546, %3046 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3051 = tosa.add %3047, %3050 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3052 = tosa.mul %3029, %3043 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_547 = tensor.extract_slice %3029[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_548 = tensor.extract_slice %3029[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3053 = tosa.negate %extracted_slice_548 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3054 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_549 = tensor.insert_slice %3053 into %3054[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_550 = tensor.insert_slice %extracted_slice_547 into %inserted_slice_549[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3055 = tosa.mul %inserted_slice_550, %3046 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3056 = tosa.add %3052, %3055 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3057 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3058 = tosa.transpose %3056, %3057 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3059 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3060 = tosa.add %3051, %3059 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3061 = tosa.reshape %3060 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3062 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3063 = tosa.add %3058, %3062 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3064 = tosa.reshape %3063 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3065 = tosa.matmul %3061, %3064 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3066 = tosa.reshape %3065 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3067 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3068 = tosa.reciprocal %3067 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3069 = tosa.mul %3066, %3068 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3070 = tosa.add %3069, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3071 = tosa.reduce_max %3070 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3072 = tosa.sub %3070, %3071 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3073 = tosa.exp %3072 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3074 = tosa.reduce_sum %3073 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3075 = tosa.reciprocal %3074 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3076 = tosa.mul %3073, %3075 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3077 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3078 = tosa.add %3076, %3077 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3079 = tosa.reshape %3078 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3080 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3081 = tosa.add %3032, %3080 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3082 = tosa.reshape %3081 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3083 = tosa.matmul %3079, %3082 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3084 = tosa.reshape %3083 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3085 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3086 = tosa.transpose %3084, %3085 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3087 = tosa.identity %3086 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3088 = tosa.reshape %3087 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3089 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3090 = tosa.transpose %arg261, %3089 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3091 = tosa.reshape %3088 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_551 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3092 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3091, %3090 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_551 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3093 = tosa.reshape %3092 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3094 = tosa.add %2996, %3093 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3095 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_552 = arith.constant 2 : i32
    %3096 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3094 : tensor<1x40x4096xf32>) outs(%3095 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_552 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3097 = tosa.reduce_sum %3096 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3098 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3099 = tosa.reciprocal %3098 : (tensor<1xf32>) -> tensor<1xf32>
    %3100 = tosa.mul %3099, %3097 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3101 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3102 = tosa.add %3100, %3101 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3103 = tosa.rsqrt %3102 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3104 = tosa.mul %3094, %3103 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3105 = tosa.reshape %arg262 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3106 = tosa.mul %3105, %3104 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3107 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3108 = tosa.transpose %arg263, %3107 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3109 = tosa.reshape %3106 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_553 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3110 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3109, %3108 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_553 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3111 = tosa.reshape %3110 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3112 = tosa.sigmoid %3111 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3113 = tosa.mul %3111, %3112 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3114 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3115 = tosa.transpose %arg264, %3114 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3116 = tosa.reshape %3106 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_554 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3117 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3116, %3115 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_554 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3118 = tosa.reshape %3117 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3119 = tosa.mul %3113, %3118 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3120 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3121 = tosa.transpose %arg265, %3120 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3122 = tosa.reshape %3119 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_555 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3123 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3122, %3121 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_555 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3124 = tosa.reshape %3123 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3125 = tosa.add %3094, %3124 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3126 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_556 = arith.constant 2 : i32
    %3127 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3125 : tensor<1x40x4096xf32>) outs(%3126 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_556 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3128 = tosa.reduce_sum %3127 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3129 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3130 = tosa.reciprocal %3129 : (tensor<1xf32>) -> tensor<1xf32>
    %3131 = tosa.mul %3130, %3128 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3132 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3133 = tosa.add %3131, %3132 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3134 = tosa.rsqrt %3133 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3135 = tosa.mul %3125, %3134 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3136 = tosa.reshape %arg266 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3137 = tosa.mul %3136, %3135 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3138 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3139 = tosa.transpose %arg267, %3138 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3140 = tosa.reshape %3137 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_557 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3141 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3140, %3139 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_557 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3142 = tosa.reshape %3141 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3143 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3144 = tosa.transpose %arg268, %3143 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3145 = tosa.reshape %3137 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_558 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3146 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3145, %3144 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_558 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3147 = tosa.reshape %3146 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3148 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3149 = tosa.transpose %arg269, %3148 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3150 = tosa.reshape %3137 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_559 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3151 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3150, %3149 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_559 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3152 = tosa.reshape %3151 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3153 = tosa.reshape %3142 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3154 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3155 = tosa.transpose %3153, %3154 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3156 = tosa.reshape %3147 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3157 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3158 = tosa.transpose %3156, %3157 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3159 = tosa.reshape %3152 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3160 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3161 = tosa.transpose %3159, %3160 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_560 = tensor.extract_slice %arg270[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_561 = tensor.extract_slice %extracted_slice_560[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_562 = tensor.extract_slice %extracted_slice_561[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_563 = tensor.extract_slice %arg271[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_564 = tensor.extract_slice %extracted_slice_563[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_565 = tensor.extract_slice %extracted_slice_564[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3162 = tensor.empty() : tensor<1x40x128xf32>
    %3163 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_562 : tensor<1x1x40x128xf32>) outs(%3162 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3164 = tensor.empty() : tensor<40x128xf32>
    %3165 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3163 : tensor<1x40x128xf32>) outs(%3164 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3166 = tensor.empty() : tensor<1x40x128xf32>
    %3167 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_565 : tensor<1x1x40x128xf32>) outs(%3166 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3168 = tensor.empty() : tensor<40x128xf32>
    %3169 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3167 : tensor<1x40x128xf32>) outs(%3168 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3170 = tensor.empty() : tensor<1x40x128xf32>
    %3171 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3170 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3165[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3172 = tosa.reshape %3171 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3173 = tensor.empty() : tensor<1x40x128xf32>
    %3174 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3173 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3169[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3175 = tosa.reshape %3174 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3176 = tosa.mul %3155, %3172 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_566 = tensor.extract_slice %3155[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_567 = tensor.extract_slice %3155[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3177 = tosa.negate %extracted_slice_567 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3178 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_568 = tensor.insert_slice %3177 into %3178[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_569 = tensor.insert_slice %extracted_slice_566 into %inserted_slice_568[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3179 = tosa.mul %inserted_slice_569, %3175 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3180 = tosa.add %3176, %3179 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3181 = tosa.mul %3158, %3172 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_570 = tensor.extract_slice %3158[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_571 = tensor.extract_slice %3158[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3182 = tosa.negate %extracted_slice_571 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3183 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_572 = tensor.insert_slice %3182 into %3183[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_573 = tensor.insert_slice %extracted_slice_570 into %inserted_slice_572[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3184 = tosa.mul %inserted_slice_573, %3175 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3185 = tosa.add %3181, %3184 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3186 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3187 = tosa.transpose %3185, %3186 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3188 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3189 = tosa.add %3180, %3188 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3190 = tosa.reshape %3189 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3191 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3192 = tosa.add %3187, %3191 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3193 = tosa.reshape %3192 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3194 = tosa.matmul %3190, %3193 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3195 = tosa.reshape %3194 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3196 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3197 = tosa.reciprocal %3196 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3198 = tosa.mul %3195, %3197 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3199 = tosa.add %3198, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3200 = tosa.reduce_max %3199 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3201 = tosa.sub %3199, %3200 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3202 = tosa.exp %3201 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3203 = tosa.reduce_sum %3202 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3204 = tosa.reciprocal %3203 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3205 = tosa.mul %3202, %3204 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3206 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3207 = tosa.add %3205, %3206 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3208 = tosa.reshape %3207 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3209 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3210 = tosa.add %3161, %3209 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3211 = tosa.reshape %3210 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3212 = tosa.matmul %3208, %3211 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3213 = tosa.reshape %3212 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3214 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3215 = tosa.transpose %3213, %3214 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3216 = tosa.identity %3215 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3217 = tosa.reshape %3216 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3218 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3219 = tosa.transpose %arg272, %3218 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3220 = tosa.reshape %3217 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_574 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3221 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3220, %3219 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_574 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3222 = tosa.reshape %3221 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3223 = tosa.add %3125, %3222 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3224 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_575 = arith.constant 2 : i32
    %3225 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3223 : tensor<1x40x4096xf32>) outs(%3224 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_575 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3226 = tosa.reduce_sum %3225 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3227 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3228 = tosa.reciprocal %3227 : (tensor<1xf32>) -> tensor<1xf32>
    %3229 = tosa.mul %3228, %3226 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3230 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3231 = tosa.add %3229, %3230 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3232 = tosa.rsqrt %3231 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3233 = tosa.mul %3223, %3232 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3234 = tosa.reshape %arg273 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3235 = tosa.mul %3234, %3233 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3236 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3237 = tosa.transpose %arg274, %3236 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3238 = tosa.reshape %3235 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_576 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3239 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3238, %3237 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_576 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3240 = tosa.reshape %3239 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3241 = tosa.sigmoid %3240 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3242 = tosa.mul %3240, %3241 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3243 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3244 = tosa.transpose %arg275, %3243 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3245 = tosa.reshape %3235 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_577 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3246 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3245, %3244 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_577 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3247 = tosa.reshape %3246 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3248 = tosa.mul %3242, %3247 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3249 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3250 = tosa.transpose %arg276, %3249 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3251 = tosa.reshape %3248 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_578 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3252 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3251, %3250 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_578 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3253 = tosa.reshape %3252 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3254 = tosa.add %3223, %3253 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3255 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_579 = arith.constant 2 : i32
    %3256 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3254 : tensor<1x40x4096xf32>) outs(%3255 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_579 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3257 = tosa.reduce_sum %3256 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3258 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3259 = tosa.reciprocal %3258 : (tensor<1xf32>) -> tensor<1xf32>
    %3260 = tosa.mul %3259, %3257 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3261 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3262 = tosa.add %3260, %3261 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3263 = tosa.rsqrt %3262 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3264 = tosa.mul %3254, %3263 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3265 = tosa.reshape %arg277 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3266 = tosa.mul %3265, %3264 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3267 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3268 = tosa.transpose %arg278, %3267 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3269 = tosa.reshape %3266 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_580 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3270 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3269, %3268 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_580 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3271 = tosa.reshape %3270 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3272 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3273 = tosa.transpose %arg279, %3272 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3274 = tosa.reshape %3266 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_581 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3275 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3274, %3273 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_581 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3276 = tosa.reshape %3275 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3277 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3278 = tosa.transpose %arg280, %3277 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3279 = tosa.reshape %3266 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_582 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3280 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3279, %3278 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_582 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3281 = tosa.reshape %3280 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3282 = tosa.reshape %3271 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3283 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3284 = tosa.transpose %3282, %3283 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3285 = tosa.reshape %3276 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3286 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3287 = tosa.transpose %3285, %3286 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3288 = tosa.reshape %3281 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3289 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3290 = tosa.transpose %3288, %3289 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_583 = tensor.extract_slice %arg281[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_584 = tensor.extract_slice %extracted_slice_583[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_585 = tensor.extract_slice %extracted_slice_584[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_586 = tensor.extract_slice %arg282[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_587 = tensor.extract_slice %extracted_slice_586[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_588 = tensor.extract_slice %extracted_slice_587[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3291 = tensor.empty() : tensor<1x40x128xf32>
    %3292 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_585 : tensor<1x1x40x128xf32>) outs(%3291 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3293 = tensor.empty() : tensor<40x128xf32>
    %3294 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3292 : tensor<1x40x128xf32>) outs(%3293 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3295 = tensor.empty() : tensor<1x40x128xf32>
    %3296 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_588 : tensor<1x1x40x128xf32>) outs(%3295 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3297 = tensor.empty() : tensor<40x128xf32>
    %3298 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3296 : tensor<1x40x128xf32>) outs(%3297 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3299 = tensor.empty() : tensor<1x40x128xf32>
    %3300 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3299 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3294[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3301 = tosa.reshape %3300 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3302 = tensor.empty() : tensor<1x40x128xf32>
    %3303 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3302 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3298[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3304 = tosa.reshape %3303 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3305 = tosa.mul %3284, %3301 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_589 = tensor.extract_slice %3284[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_590 = tensor.extract_slice %3284[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3306 = tosa.negate %extracted_slice_590 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3307 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_591 = tensor.insert_slice %3306 into %3307[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_592 = tensor.insert_slice %extracted_slice_589 into %inserted_slice_591[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3308 = tosa.mul %inserted_slice_592, %3304 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3309 = tosa.add %3305, %3308 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3310 = tosa.mul %3287, %3301 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_593 = tensor.extract_slice %3287[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_594 = tensor.extract_slice %3287[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3311 = tosa.negate %extracted_slice_594 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3312 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_595 = tensor.insert_slice %3311 into %3312[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_596 = tensor.insert_slice %extracted_slice_593 into %inserted_slice_595[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3313 = tosa.mul %inserted_slice_596, %3304 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3314 = tosa.add %3310, %3313 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3315 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3316 = tosa.transpose %3314, %3315 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3317 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3318 = tosa.add %3309, %3317 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3319 = tosa.reshape %3318 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3320 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3321 = tosa.add %3316, %3320 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3322 = tosa.reshape %3321 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3323 = tosa.matmul %3319, %3322 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3324 = tosa.reshape %3323 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3325 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3326 = tosa.reciprocal %3325 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3327 = tosa.mul %3324, %3326 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3328 = tosa.add %3327, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3329 = tosa.reduce_max %3328 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3330 = tosa.sub %3328, %3329 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3331 = tosa.exp %3330 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3332 = tosa.reduce_sum %3331 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3333 = tosa.reciprocal %3332 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3334 = tosa.mul %3331, %3333 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3335 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3336 = tosa.add %3334, %3335 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3337 = tosa.reshape %3336 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3338 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3339 = tosa.add %3290, %3338 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3340 = tosa.reshape %3339 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3341 = tosa.matmul %3337, %3340 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3342 = tosa.reshape %3341 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3343 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3344 = tosa.transpose %3342, %3343 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3345 = tosa.identity %3344 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3346 = tosa.reshape %3345 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3347 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3348 = tosa.transpose %arg283, %3347 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3349 = tosa.reshape %3346 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_597 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3350 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3349, %3348 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_597 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3351 = tosa.reshape %3350 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3352 = tosa.add %3254, %3351 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3353 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_598 = arith.constant 2 : i32
    %3354 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3352 : tensor<1x40x4096xf32>) outs(%3353 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_598 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3355 = tosa.reduce_sum %3354 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3356 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3357 = tosa.reciprocal %3356 : (tensor<1xf32>) -> tensor<1xf32>
    %3358 = tosa.mul %3357, %3355 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3359 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3360 = tosa.add %3358, %3359 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3361 = tosa.rsqrt %3360 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3362 = tosa.mul %3352, %3361 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3363 = tosa.reshape %arg284 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3364 = tosa.mul %3363, %3362 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3365 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3366 = tosa.transpose %arg285, %3365 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3367 = tosa.reshape %3364 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_599 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3368 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3367, %3366 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_599 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3369 = tosa.reshape %3368 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3370 = tosa.sigmoid %3369 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3371 = tosa.mul %3369, %3370 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3372 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3373 = tosa.transpose %arg286, %3372 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3374 = tosa.reshape %3364 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_600 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3375 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3374, %3373 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_600 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3376 = tosa.reshape %3375 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3377 = tosa.mul %3371, %3376 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3378 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3379 = tosa.transpose %arg287, %3378 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3380 = tosa.reshape %3377 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_601 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3381 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3380, %3379 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_601 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3382 = tosa.reshape %3381 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3383 = tosa.add %3352, %3382 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3384 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_602 = arith.constant 2 : i32
    %3385 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3383 : tensor<1x40x4096xf32>) outs(%3384 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_602 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3386 = tosa.reduce_sum %3385 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3387 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3388 = tosa.reciprocal %3387 : (tensor<1xf32>) -> tensor<1xf32>
    %3389 = tosa.mul %3388, %3386 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3390 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3391 = tosa.add %3389, %3390 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3392 = tosa.rsqrt %3391 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3393 = tosa.mul %3383, %3392 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3394 = tosa.reshape %arg288 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3395 = tosa.mul %3394, %3393 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3396 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3397 = tosa.transpose %arg289, %3396 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3398 = tosa.reshape %3395 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_603 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3399 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3398, %3397 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_603 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3400 = tosa.reshape %3399 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3401 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3402 = tosa.transpose %arg290, %3401 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3403 = tosa.reshape %3395 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_604 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3404 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3403, %3402 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_604 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3405 = tosa.reshape %3404 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3406 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3407 = tosa.transpose %arg291, %3406 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3408 = tosa.reshape %3395 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_605 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3409 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3408, %3407 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_605 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3410 = tosa.reshape %3409 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3411 = tosa.reshape %3400 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3412 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3413 = tosa.transpose %3411, %3412 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3414 = tosa.reshape %3405 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3415 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3416 = tosa.transpose %3414, %3415 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3417 = tosa.reshape %3410 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3418 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3419 = tosa.transpose %3417, %3418 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_606 = tensor.extract_slice %arg292[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_607 = tensor.extract_slice %extracted_slice_606[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_608 = tensor.extract_slice %extracted_slice_607[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_609 = tensor.extract_slice %arg293[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_610 = tensor.extract_slice %extracted_slice_609[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_611 = tensor.extract_slice %extracted_slice_610[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3420 = tensor.empty() : tensor<1x40x128xf32>
    %3421 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_608 : tensor<1x1x40x128xf32>) outs(%3420 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3422 = tensor.empty() : tensor<40x128xf32>
    %3423 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3421 : tensor<1x40x128xf32>) outs(%3422 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3424 = tensor.empty() : tensor<1x40x128xf32>
    %3425 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_611 : tensor<1x1x40x128xf32>) outs(%3424 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3426 = tensor.empty() : tensor<40x128xf32>
    %3427 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3425 : tensor<1x40x128xf32>) outs(%3426 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3428 = tensor.empty() : tensor<1x40x128xf32>
    %3429 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3428 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3423[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3430 = tosa.reshape %3429 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3431 = tensor.empty() : tensor<1x40x128xf32>
    %3432 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3431 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3427[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3433 = tosa.reshape %3432 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3434 = tosa.mul %3413, %3430 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_612 = tensor.extract_slice %3413[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_613 = tensor.extract_slice %3413[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3435 = tosa.negate %extracted_slice_613 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3436 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_614 = tensor.insert_slice %3435 into %3436[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_615 = tensor.insert_slice %extracted_slice_612 into %inserted_slice_614[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3437 = tosa.mul %inserted_slice_615, %3433 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3438 = tosa.add %3434, %3437 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3439 = tosa.mul %3416, %3430 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_616 = tensor.extract_slice %3416[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_617 = tensor.extract_slice %3416[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3440 = tosa.negate %extracted_slice_617 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3441 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_618 = tensor.insert_slice %3440 into %3441[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_619 = tensor.insert_slice %extracted_slice_616 into %inserted_slice_618[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3442 = tosa.mul %inserted_slice_619, %3433 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3443 = tosa.add %3439, %3442 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3444 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3445 = tosa.transpose %3443, %3444 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3446 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3447 = tosa.add %3438, %3446 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3448 = tosa.reshape %3447 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3449 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3450 = tosa.add %3445, %3449 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3451 = tosa.reshape %3450 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3452 = tosa.matmul %3448, %3451 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3453 = tosa.reshape %3452 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3454 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3455 = tosa.reciprocal %3454 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3456 = tosa.mul %3453, %3455 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3457 = tosa.add %3456, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3458 = tosa.reduce_max %3457 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3459 = tosa.sub %3457, %3458 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3460 = tosa.exp %3459 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3461 = tosa.reduce_sum %3460 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3462 = tosa.reciprocal %3461 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3463 = tosa.mul %3460, %3462 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3464 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3465 = tosa.add %3463, %3464 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3466 = tosa.reshape %3465 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3467 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3468 = tosa.add %3419, %3467 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3469 = tosa.reshape %3468 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3470 = tosa.matmul %3466, %3469 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3471 = tosa.reshape %3470 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3472 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3473 = tosa.transpose %3471, %3472 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3474 = tosa.identity %3473 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3475 = tosa.reshape %3474 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3476 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3477 = tosa.transpose %arg294, %3476 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3478 = tosa.reshape %3475 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_620 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3479 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3478, %3477 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_620 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3480 = tosa.reshape %3479 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3481 = tosa.add %3383, %3480 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3482 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_621 = arith.constant 2 : i32
    %3483 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3481 : tensor<1x40x4096xf32>) outs(%3482 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_621 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3484 = tosa.reduce_sum %3483 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3485 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3486 = tosa.reciprocal %3485 : (tensor<1xf32>) -> tensor<1xf32>
    %3487 = tosa.mul %3486, %3484 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3488 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3489 = tosa.add %3487, %3488 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3490 = tosa.rsqrt %3489 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3491 = tosa.mul %3481, %3490 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3492 = tosa.reshape %arg295 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3493 = tosa.mul %3492, %3491 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3494 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3495 = tosa.transpose %arg296, %3494 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3496 = tosa.reshape %3493 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_622 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3497 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3496, %3495 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_622 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3498 = tosa.reshape %3497 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3499 = tosa.sigmoid %3498 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3500 = tosa.mul %3498, %3499 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3501 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3502 = tosa.transpose %arg297, %3501 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3503 = tosa.reshape %3493 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_623 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3504 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3503, %3502 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_623 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3505 = tosa.reshape %3504 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3506 = tosa.mul %3500, %3505 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3507 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3508 = tosa.transpose %arg298, %3507 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3509 = tosa.reshape %3506 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_624 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3510 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3509, %3508 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_624 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3511 = tosa.reshape %3510 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3512 = tosa.add %3481, %3511 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3513 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_625 = arith.constant 2 : i32
    %3514 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3512 : tensor<1x40x4096xf32>) outs(%3513 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_625 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3515 = tosa.reduce_sum %3514 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3516 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3517 = tosa.reciprocal %3516 : (tensor<1xf32>) -> tensor<1xf32>
    %3518 = tosa.mul %3517, %3515 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3519 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3520 = tosa.add %3518, %3519 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3521 = tosa.rsqrt %3520 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3522 = tosa.mul %3512, %3521 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3523 = tosa.reshape %arg299 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3524 = tosa.mul %3523, %3522 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3525 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3526 = tosa.transpose %arg300, %3525 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3527 = tosa.reshape %3524 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_626 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3528 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3527, %3526 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_626 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3529 = tosa.reshape %3528 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3530 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3531 = tosa.transpose %arg301, %3530 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3532 = tosa.reshape %3524 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_627 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3533 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3532, %3531 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_627 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3534 = tosa.reshape %3533 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3535 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3536 = tosa.transpose %arg302, %3535 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3537 = tosa.reshape %3524 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_628 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3538 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3537, %3536 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_628 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3539 = tosa.reshape %3538 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3540 = tosa.reshape %3529 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3541 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3542 = tosa.transpose %3540, %3541 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3543 = tosa.reshape %3534 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3544 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3545 = tosa.transpose %3543, %3544 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3546 = tosa.reshape %3539 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3547 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3548 = tosa.transpose %3546, %3547 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_629 = tensor.extract_slice %arg303[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_630 = tensor.extract_slice %extracted_slice_629[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_631 = tensor.extract_slice %extracted_slice_630[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_632 = tensor.extract_slice %arg304[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_633 = tensor.extract_slice %extracted_slice_632[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_634 = tensor.extract_slice %extracted_slice_633[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3549 = tensor.empty() : tensor<1x40x128xf32>
    %3550 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_631 : tensor<1x1x40x128xf32>) outs(%3549 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3551 = tensor.empty() : tensor<40x128xf32>
    %3552 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3550 : tensor<1x40x128xf32>) outs(%3551 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3553 = tensor.empty() : tensor<1x40x128xf32>
    %3554 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_634 : tensor<1x1x40x128xf32>) outs(%3553 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3555 = tensor.empty() : tensor<40x128xf32>
    %3556 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3554 : tensor<1x40x128xf32>) outs(%3555 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3557 = tensor.empty() : tensor<1x40x128xf32>
    %3558 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3557 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3552[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3559 = tosa.reshape %3558 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3560 = tensor.empty() : tensor<1x40x128xf32>
    %3561 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3560 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3556[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3562 = tosa.reshape %3561 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3563 = tosa.mul %3542, %3559 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_635 = tensor.extract_slice %3542[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_636 = tensor.extract_slice %3542[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3564 = tosa.negate %extracted_slice_636 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3565 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_637 = tensor.insert_slice %3564 into %3565[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_638 = tensor.insert_slice %extracted_slice_635 into %inserted_slice_637[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3566 = tosa.mul %inserted_slice_638, %3562 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3567 = tosa.add %3563, %3566 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3568 = tosa.mul %3545, %3559 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_639 = tensor.extract_slice %3545[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_640 = tensor.extract_slice %3545[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3569 = tosa.negate %extracted_slice_640 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3570 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_641 = tensor.insert_slice %3569 into %3570[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_642 = tensor.insert_slice %extracted_slice_639 into %inserted_slice_641[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3571 = tosa.mul %inserted_slice_642, %3562 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3572 = tosa.add %3568, %3571 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3573 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3574 = tosa.transpose %3572, %3573 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3575 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3576 = tosa.add %3567, %3575 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3577 = tosa.reshape %3576 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3578 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3579 = tosa.add %3574, %3578 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3580 = tosa.reshape %3579 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3581 = tosa.matmul %3577, %3580 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3582 = tosa.reshape %3581 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3583 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3584 = tosa.reciprocal %3583 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3585 = tosa.mul %3582, %3584 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3586 = tosa.add %3585, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3587 = tosa.reduce_max %3586 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3588 = tosa.sub %3586, %3587 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3589 = tosa.exp %3588 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3590 = tosa.reduce_sum %3589 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3591 = tosa.reciprocal %3590 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3592 = tosa.mul %3589, %3591 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3593 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3594 = tosa.add %3592, %3593 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3595 = tosa.reshape %3594 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3596 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3597 = tosa.add %3548, %3596 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3598 = tosa.reshape %3597 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3599 = tosa.matmul %3595, %3598 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3600 = tosa.reshape %3599 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3601 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3602 = tosa.transpose %3600, %3601 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3603 = tosa.identity %3602 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3604 = tosa.reshape %3603 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3605 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3606 = tosa.transpose %arg305, %3605 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3607 = tosa.reshape %3604 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_643 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3608 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3607, %3606 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_643 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3609 = tosa.reshape %3608 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3610 = tosa.add %3512, %3609 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3611 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_644 = arith.constant 2 : i32
    %3612 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3610 : tensor<1x40x4096xf32>) outs(%3611 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_644 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3613 = tosa.reduce_sum %3612 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3614 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3615 = tosa.reciprocal %3614 : (tensor<1xf32>) -> tensor<1xf32>
    %3616 = tosa.mul %3615, %3613 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3617 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3618 = tosa.add %3616, %3617 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3619 = tosa.rsqrt %3618 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3620 = tosa.mul %3610, %3619 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3621 = tosa.reshape %arg306 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3622 = tosa.mul %3621, %3620 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3623 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3624 = tosa.transpose %arg307, %3623 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3625 = tosa.reshape %3622 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_645 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3626 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3625, %3624 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_645 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3627 = tosa.reshape %3626 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3628 = tosa.sigmoid %3627 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3629 = tosa.mul %3627, %3628 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3630 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3631 = tosa.transpose %arg308, %3630 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3632 = tosa.reshape %3622 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_646 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3633 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3632, %3631 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_646 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3634 = tosa.reshape %3633 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3635 = tosa.mul %3629, %3634 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3636 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3637 = tosa.transpose %arg309, %3636 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3638 = tosa.reshape %3635 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_647 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3639 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3638, %3637 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_647 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3640 = tosa.reshape %3639 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3641 = tosa.add %3610, %3640 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3642 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_648 = arith.constant 2 : i32
    %3643 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3641 : tensor<1x40x4096xf32>) outs(%3642 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_648 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3644 = tosa.reduce_sum %3643 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3645 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3646 = tosa.reciprocal %3645 : (tensor<1xf32>) -> tensor<1xf32>
    %3647 = tosa.mul %3646, %3644 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3648 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3649 = tosa.add %3647, %3648 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3650 = tosa.rsqrt %3649 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3651 = tosa.mul %3641, %3650 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3652 = tosa.reshape %arg310 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3653 = tosa.mul %3652, %3651 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3654 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3655 = tosa.transpose %arg311, %3654 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3656 = tosa.reshape %3653 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_649 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3657 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3656, %3655 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_649 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3658 = tosa.reshape %3657 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3659 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3660 = tosa.transpose %arg312, %3659 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3661 = tosa.reshape %3653 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_650 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3662 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3661, %3660 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_650 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3663 = tosa.reshape %3662 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3664 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3665 = tosa.transpose %arg313, %3664 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3666 = tosa.reshape %3653 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_651 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3667 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3666, %3665 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_651 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3668 = tosa.reshape %3667 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3669 = tosa.reshape %3658 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3670 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3671 = tosa.transpose %3669, %3670 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3672 = tosa.reshape %3663 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3673 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3674 = tosa.transpose %3672, %3673 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3675 = tosa.reshape %3668 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3676 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3677 = tosa.transpose %3675, %3676 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_652 = tensor.extract_slice %arg314[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_653 = tensor.extract_slice %extracted_slice_652[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_654 = tensor.extract_slice %extracted_slice_653[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_655 = tensor.extract_slice %arg315[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_656 = tensor.extract_slice %extracted_slice_655[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_657 = tensor.extract_slice %extracted_slice_656[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3678 = tensor.empty() : tensor<1x40x128xf32>
    %3679 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_654 : tensor<1x1x40x128xf32>) outs(%3678 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3680 = tensor.empty() : tensor<40x128xf32>
    %3681 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3679 : tensor<1x40x128xf32>) outs(%3680 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3682 = tensor.empty() : tensor<1x40x128xf32>
    %3683 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_657 : tensor<1x1x40x128xf32>) outs(%3682 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3684 = tensor.empty() : tensor<40x128xf32>
    %3685 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3683 : tensor<1x40x128xf32>) outs(%3684 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3686 = tensor.empty() : tensor<1x40x128xf32>
    %3687 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3686 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3681[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3688 = tosa.reshape %3687 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3689 = tensor.empty() : tensor<1x40x128xf32>
    %3690 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3689 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3685[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3691 = tosa.reshape %3690 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3692 = tosa.mul %3671, %3688 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_658 = tensor.extract_slice %3671[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_659 = tensor.extract_slice %3671[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3693 = tosa.negate %extracted_slice_659 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3694 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_660 = tensor.insert_slice %3693 into %3694[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_661 = tensor.insert_slice %extracted_slice_658 into %inserted_slice_660[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3695 = tosa.mul %inserted_slice_661, %3691 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3696 = tosa.add %3692, %3695 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3697 = tosa.mul %3674, %3688 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_662 = tensor.extract_slice %3674[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_663 = tensor.extract_slice %3674[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3698 = tosa.negate %extracted_slice_663 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3699 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_664 = tensor.insert_slice %3698 into %3699[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_665 = tensor.insert_slice %extracted_slice_662 into %inserted_slice_664[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3700 = tosa.mul %inserted_slice_665, %3691 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3701 = tosa.add %3697, %3700 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3702 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3703 = tosa.transpose %3701, %3702 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3704 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3705 = tosa.add %3696, %3704 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3706 = tosa.reshape %3705 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3707 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3708 = tosa.add %3703, %3707 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3709 = tosa.reshape %3708 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3710 = tosa.matmul %3706, %3709 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3711 = tosa.reshape %3710 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3712 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3713 = tosa.reciprocal %3712 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3714 = tosa.mul %3711, %3713 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3715 = tosa.add %3714, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3716 = tosa.reduce_max %3715 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3717 = tosa.sub %3715, %3716 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3718 = tosa.exp %3717 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3719 = tosa.reduce_sum %3718 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3720 = tosa.reciprocal %3719 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3721 = tosa.mul %3718, %3720 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3722 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3723 = tosa.add %3721, %3722 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3724 = tosa.reshape %3723 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3725 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3726 = tosa.add %3677, %3725 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3727 = tosa.reshape %3726 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3728 = tosa.matmul %3724, %3727 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3729 = tosa.reshape %3728 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3730 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3731 = tosa.transpose %3729, %3730 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3732 = tosa.identity %3731 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3733 = tosa.reshape %3732 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3734 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3735 = tosa.transpose %arg316, %3734 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3736 = tosa.reshape %3733 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_666 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3737 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3736, %3735 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_666 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3738 = tosa.reshape %3737 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3739 = tosa.add %3641, %3738 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3740 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_667 = arith.constant 2 : i32
    %3741 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3739 : tensor<1x40x4096xf32>) outs(%3740 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_667 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3742 = tosa.reduce_sum %3741 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3743 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3744 = tosa.reciprocal %3743 : (tensor<1xf32>) -> tensor<1xf32>
    %3745 = tosa.mul %3744, %3742 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3746 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3747 = tosa.add %3745, %3746 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3748 = tosa.rsqrt %3747 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3749 = tosa.mul %3739, %3748 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3750 = tosa.reshape %arg317 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3751 = tosa.mul %3750, %3749 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3752 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3753 = tosa.transpose %arg318, %3752 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3754 = tosa.reshape %3751 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_668 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3755 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3754, %3753 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_668 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3756 = tosa.reshape %3755 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3757 = tosa.sigmoid %3756 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3758 = tosa.mul %3756, %3757 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3759 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3760 = tosa.transpose %arg319, %3759 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3761 = tosa.reshape %3751 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_669 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3762 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3761, %3760 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_669 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3763 = tosa.reshape %3762 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3764 = tosa.mul %3758, %3763 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3765 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3766 = tosa.transpose %arg320, %3765 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3767 = tosa.reshape %3764 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_670 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3768 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3767, %3766 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_670 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3769 = tosa.reshape %3768 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3770 = tosa.add %3739, %3769 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3771 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_671 = arith.constant 2 : i32
    %3772 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3770 : tensor<1x40x4096xf32>) outs(%3771 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_671 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3773 = tosa.reduce_sum %3772 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3774 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3775 = tosa.reciprocal %3774 : (tensor<1xf32>) -> tensor<1xf32>
    %3776 = tosa.mul %3775, %3773 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3777 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3778 = tosa.add %3776, %3777 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3779 = tosa.rsqrt %3778 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3780 = tosa.mul %3770, %3779 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3781 = tosa.reshape %arg321 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3782 = tosa.mul %3781, %3780 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3783 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3784 = tosa.transpose %arg322, %3783 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3785 = tosa.reshape %3782 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_672 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3786 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3785, %3784 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_672 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3787 = tosa.reshape %3786 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3788 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3789 = tosa.transpose %arg323, %3788 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3790 = tosa.reshape %3782 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_673 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3791 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3790, %3789 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_673 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3792 = tosa.reshape %3791 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3793 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3794 = tosa.transpose %arg324, %3793 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3795 = tosa.reshape %3782 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_674 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3796 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3795, %3794 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_674 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3797 = tosa.reshape %3796 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3798 = tosa.reshape %3787 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3799 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3800 = tosa.transpose %3798, %3799 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3801 = tosa.reshape %3792 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3802 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3803 = tosa.transpose %3801, %3802 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3804 = tosa.reshape %3797 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3805 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3806 = tosa.transpose %3804, %3805 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_675 = tensor.extract_slice %arg325[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_676 = tensor.extract_slice %extracted_slice_675[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_677 = tensor.extract_slice %extracted_slice_676[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_678 = tensor.extract_slice %arg326[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_679 = tensor.extract_slice %extracted_slice_678[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_680 = tensor.extract_slice %extracted_slice_679[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3807 = tensor.empty() : tensor<1x40x128xf32>
    %3808 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_677 : tensor<1x1x40x128xf32>) outs(%3807 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3809 = tensor.empty() : tensor<40x128xf32>
    %3810 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3808 : tensor<1x40x128xf32>) outs(%3809 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3811 = tensor.empty() : tensor<1x40x128xf32>
    %3812 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_680 : tensor<1x1x40x128xf32>) outs(%3811 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3813 = tensor.empty() : tensor<40x128xf32>
    %3814 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3812 : tensor<1x40x128xf32>) outs(%3813 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3815 = tensor.empty() : tensor<1x40x128xf32>
    %3816 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3815 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3810[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3817 = tosa.reshape %3816 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3818 = tensor.empty() : tensor<1x40x128xf32>
    %3819 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3818 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3814[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3820 = tosa.reshape %3819 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3821 = tosa.mul %3800, %3817 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_681 = tensor.extract_slice %3800[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_682 = tensor.extract_slice %3800[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3822 = tosa.negate %extracted_slice_682 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3823 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_683 = tensor.insert_slice %3822 into %3823[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_684 = tensor.insert_slice %extracted_slice_681 into %inserted_slice_683[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3824 = tosa.mul %inserted_slice_684, %3820 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3825 = tosa.add %3821, %3824 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3826 = tosa.mul %3803, %3817 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_685 = tensor.extract_slice %3803[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_686 = tensor.extract_slice %3803[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3827 = tosa.negate %extracted_slice_686 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3828 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_687 = tensor.insert_slice %3827 into %3828[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_688 = tensor.insert_slice %extracted_slice_685 into %inserted_slice_687[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3829 = tosa.mul %inserted_slice_688, %3820 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3830 = tosa.add %3826, %3829 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3831 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3832 = tosa.transpose %3830, %3831 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3833 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3834 = tosa.add %3825, %3833 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3835 = tosa.reshape %3834 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3836 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3837 = tosa.add %3832, %3836 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3838 = tosa.reshape %3837 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3839 = tosa.matmul %3835, %3838 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3840 = tosa.reshape %3839 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3841 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3842 = tosa.reciprocal %3841 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3843 = tosa.mul %3840, %3842 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3844 = tosa.add %3843, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3845 = tosa.reduce_max %3844 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3846 = tosa.sub %3844, %3845 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3847 = tosa.exp %3846 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3848 = tosa.reduce_sum %3847 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3849 = tosa.reciprocal %3848 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3850 = tosa.mul %3847, %3849 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3851 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3852 = tosa.add %3850, %3851 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3853 = tosa.reshape %3852 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3854 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3855 = tosa.add %3806, %3854 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3856 = tosa.reshape %3855 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3857 = tosa.matmul %3853, %3856 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3858 = tosa.reshape %3857 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3859 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3860 = tosa.transpose %3858, %3859 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3861 = tosa.identity %3860 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3862 = tosa.reshape %3861 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3863 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3864 = tosa.transpose %arg327, %3863 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3865 = tosa.reshape %3862 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_689 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3866 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3865, %3864 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_689 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3867 = tosa.reshape %3866 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3868 = tosa.add %3770, %3867 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3869 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_690 = arith.constant 2 : i32
    %3870 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3868 : tensor<1x40x4096xf32>) outs(%3869 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_690 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3871 = tosa.reduce_sum %3870 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3872 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3873 = tosa.reciprocal %3872 : (tensor<1xf32>) -> tensor<1xf32>
    %3874 = tosa.mul %3873, %3871 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3875 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3876 = tosa.add %3874, %3875 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3877 = tosa.rsqrt %3876 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3878 = tosa.mul %3868, %3877 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3879 = tosa.reshape %arg328 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3880 = tosa.mul %3879, %3878 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3881 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3882 = tosa.transpose %arg329, %3881 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3883 = tosa.reshape %3880 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_691 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3884 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3883, %3882 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_691 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3885 = tosa.reshape %3884 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3886 = tosa.sigmoid %3885 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3887 = tosa.mul %3885, %3886 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3888 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3889 = tosa.transpose %arg330, %3888 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3890 = tosa.reshape %3880 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_692 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3891 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3890, %3889 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_692 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3892 = tosa.reshape %3891 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3893 = tosa.mul %3887, %3892 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3894 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3895 = tosa.transpose %arg331, %3894 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3896 = tosa.reshape %3893 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_693 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3897 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3896, %3895 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_693 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3898 = tosa.reshape %3897 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3899 = tosa.add %3868, %3898 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3900 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_694 = arith.constant 2 : i32
    %3901 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3899 : tensor<1x40x4096xf32>) outs(%3900 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_694 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %3902 = tosa.reduce_sum %3901 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %3903 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3904 = tosa.reciprocal %3903 : (tensor<1xf32>) -> tensor<1xf32>
    %3905 = tosa.mul %3904, %3902 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3906 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3907 = tosa.add %3905, %3906 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3908 = tosa.rsqrt %3907 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3909 = tosa.mul %3899, %3908 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3910 = tosa.reshape %arg332 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3911 = tosa.mul %3910, %3909 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3912 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3913 = tosa.transpose %arg333, %3912 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3914 = tosa.reshape %3911 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_695 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3915 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3914, %3913 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_695 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3916 = tosa.reshape %3915 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3917 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3918 = tosa.transpose %arg334, %3917 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3919 = tosa.reshape %3911 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_696 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3920 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3919, %3918 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_696 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3921 = tosa.reshape %3920 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3922 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3923 = tosa.transpose %arg335, %3922 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3924 = tosa.reshape %3911 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_697 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3925 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3924, %3923 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_697 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3926 = tosa.reshape %3925 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3927 = tosa.reshape %3916 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3928 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3929 = tosa.transpose %3927, %3928 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3930 = tosa.reshape %3921 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3931 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3932 = tosa.transpose %3930, %3931 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3933 = tosa.reshape %3926 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3934 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3935 = tosa.transpose %3933, %3934 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_698 = tensor.extract_slice %arg336[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_699 = tensor.extract_slice %extracted_slice_698[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_700 = tensor.extract_slice %extracted_slice_699[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_701 = tensor.extract_slice %arg337[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_702 = tensor.extract_slice %extracted_slice_701[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_703 = tensor.extract_slice %extracted_slice_702[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3936 = tensor.empty() : tensor<1x40x128xf32>
    %3937 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_700 : tensor<1x1x40x128xf32>) outs(%3936 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3938 = tensor.empty() : tensor<40x128xf32>
    %3939 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3937 : tensor<1x40x128xf32>) outs(%3938 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3940 = tensor.empty() : tensor<1x40x128xf32>
    %3941 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_703 : tensor<1x1x40x128xf32>) outs(%3940 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3942 = tensor.empty() : tensor<40x128xf32>
    %3943 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3941 : tensor<1x40x128xf32>) outs(%3942 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3944 = tensor.empty() : tensor<1x40x128xf32>
    %3945 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3944 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3939[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3946 = tosa.reshape %3945 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3947 = tensor.empty() : tensor<1x40x128xf32>
    %3948 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3947 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %3943[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3949 = tosa.reshape %3948 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3950 = tosa.mul %3929, %3946 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_704 = tensor.extract_slice %3929[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_705 = tensor.extract_slice %3929[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3951 = tosa.negate %extracted_slice_705 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3952 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_706 = tensor.insert_slice %3951 into %3952[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_707 = tensor.insert_slice %extracted_slice_704 into %inserted_slice_706[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3953 = tosa.mul %inserted_slice_707, %3949 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3954 = tosa.add %3950, %3953 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3955 = tosa.mul %3932, %3946 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_708 = tensor.extract_slice %3932[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_709 = tensor.extract_slice %3932[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3956 = tosa.negate %extracted_slice_709 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3957 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_710 = tensor.insert_slice %3956 into %3957[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_711 = tensor.insert_slice %extracted_slice_708 into %inserted_slice_710[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3958 = tosa.mul %inserted_slice_711, %3949 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3959 = tosa.add %3955, %3958 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3960 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3961 = tosa.transpose %3959, %3960 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3962 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3963 = tosa.add %3954, %3962 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3964 = tosa.reshape %3963 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3965 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3966 = tosa.add %3961, %3965 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3967 = tosa.reshape %3966 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3968 = tosa.matmul %3964, %3967 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3969 = tosa.reshape %3968 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3970 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3971 = tosa.reciprocal %3970 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3972 = tosa.mul %3969, %3971 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3973 = tosa.add %3972, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3974 = tosa.reduce_max %3973 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3975 = tosa.sub %3973, %3974 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3976 = tosa.exp %3975 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3977 = tosa.reduce_sum %3976 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3978 = tosa.reciprocal %3977 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3979 = tosa.mul %3976, %3978 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3980 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3981 = tosa.add %3979, %3980 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3982 = tosa.reshape %3981 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3983 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3984 = tosa.add %3935, %3983 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3985 = tosa.reshape %3984 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3986 = tosa.matmul %3982, %3985 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3987 = tosa.reshape %3986 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3988 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3989 = tosa.transpose %3987, %3988 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3990 = tosa.identity %3989 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3991 = tosa.reshape %3990 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3992 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3993 = tosa.transpose %arg338, %3992 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3994 = tosa.reshape %3991 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_712 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3995 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3994, %3993 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_712 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3996 = tosa.reshape %3995 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3997 = tosa.add %3899, %3996 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3998 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_713 = arith.constant 2 : i32
    %3999 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3997 : tensor<1x40x4096xf32>) outs(%3998 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_713 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %4000 = tosa.reduce_sum %3999 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %4001 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4002 = tosa.reciprocal %4001 : (tensor<1xf32>) -> tensor<1xf32>
    %4003 = tosa.mul %4002, %4000 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4004 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4005 = tosa.add %4003, %4004 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4006 = tosa.rsqrt %4005 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4007 = tosa.mul %3997, %4006 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4008 = tosa.reshape %arg339 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4009 = tosa.mul %4008, %4007 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %4010 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4011 = tosa.transpose %arg340, %4010 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %4012 = tosa.reshape %4009 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_714 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %4013 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4012, %4011 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_714 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %4014 = tosa.reshape %4013 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %4015 = tosa.sigmoid %4014 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %4016 = tosa.mul %4014, %4015 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %4017 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4018 = tosa.transpose %arg341, %4017 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %4019 = tosa.reshape %4009 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_715 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %4020 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4019, %4018 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_715 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %4021 = tosa.reshape %4020 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %4022 = tosa.mul %4016, %4021 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %4023 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4024 = tosa.transpose %arg342, %4023 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %4025 = tosa.reshape %4022 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_716 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4026 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4025, %4024 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_716 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %4027 = tosa.reshape %4026 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %4028 = tosa.add %3997, %4027 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %4029 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_717 = arith.constant 2 : i32
    %4030 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4028 : tensor<1x40x4096xf32>) outs(%4029 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_717 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %4031 = tosa.reduce_sum %4030 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %4032 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4033 = tosa.reciprocal %4032 : (tensor<1xf32>) -> tensor<1xf32>
    %4034 = tosa.mul %4033, %4031 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4035 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4036 = tosa.add %4034, %4035 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4037 = tosa.rsqrt %4036 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4038 = tosa.mul %4028, %4037 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4039 = tosa.reshape %arg343 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4040 = tosa.mul %4039, %4038 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %4041 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4042 = tosa.transpose %arg344, %4041 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %4043 = tosa.reshape %4040 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_718 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4044 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4043, %4042 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_718 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %4045 = tosa.reshape %4044 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %4046 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4047 = tosa.transpose %arg345, %4046 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %4048 = tosa.reshape %4040 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_719 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4049 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4048, %4047 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_719 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %4050 = tosa.reshape %4049 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %4051 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4052 = tosa.transpose %arg346, %4051 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %4053 = tosa.reshape %4040 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_720 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4054 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4053, %4052 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_720 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %4055 = tosa.reshape %4054 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %4056 = tosa.reshape %4045 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %4057 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4058 = tosa.transpose %4056, %4057 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %4059 = tosa.reshape %4050 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %4060 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4061 = tosa.transpose %4059, %4060 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %4062 = tosa.reshape %4055 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %4063 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4064 = tosa.transpose %4062, %4063 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_721 = tensor.extract_slice %arg347[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_722 = tensor.extract_slice %extracted_slice_721[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_723 = tensor.extract_slice %extracted_slice_722[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_724 = tensor.extract_slice %arg348[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_725 = tensor.extract_slice %extracted_slice_724[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_726 = tensor.extract_slice %extracted_slice_725[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %4065 = tensor.empty() : tensor<1x40x128xf32>
    %4066 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_723 : tensor<1x1x40x128xf32>) outs(%4065 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %4067 = tensor.empty() : tensor<40x128xf32>
    %4068 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%4066 : tensor<1x40x128xf32>) outs(%4067 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %4069 = tensor.empty() : tensor<1x40x128xf32>
    %4070 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_726 : tensor<1x1x40x128xf32>) outs(%4069 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %4071 = tensor.empty() : tensor<40x128xf32>
    %4072 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%4070 : tensor<1x40x128xf32>) outs(%4071 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %4073 = tensor.empty() : tensor<1x40x128xf32>
    %4074 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%4073 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %4068[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %4075 = tosa.reshape %4074 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %4076 = tensor.empty() : tensor<1x40x128xf32>
    %4077 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%4076 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4175 = arith.index_cast %in : i64 to index
      %4176 = linalg.index 2 : index
      %extracted = tensor.extract %4072[%4175, %4176] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %4078 = tosa.reshape %4077 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %4079 = tosa.mul %4058, %4075 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_727 = tensor.extract_slice %4058[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_728 = tensor.extract_slice %4058[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %4080 = tosa.negate %extracted_slice_728 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %4081 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_729 = tensor.insert_slice %4080 into %4081[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_730 = tensor.insert_slice %extracted_slice_727 into %inserted_slice_729[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %4082 = tosa.mul %inserted_slice_730, %4078 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %4083 = tosa.add %4079, %4082 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %4084 = tosa.mul %4061, %4075 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_731 = tensor.extract_slice %4061[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_732 = tensor.extract_slice %4061[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %4085 = tosa.negate %extracted_slice_732 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %4086 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_733 = tensor.insert_slice %4085 into %4086[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_734 = tensor.insert_slice %extracted_slice_731 into %inserted_slice_733[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %4087 = tosa.mul %inserted_slice_734, %4078 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %4088 = tosa.add %4084, %4087 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %4089 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4090 = tosa.transpose %4088, %4089 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %4091 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %4092 = tosa.add %4083, %4091 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %4093 = tosa.reshape %4092 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %4094 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %4095 = tosa.add %4090, %4094 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %4096 = tosa.reshape %4095 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %4097 = tosa.matmul %4093, %4096 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %4098 = tosa.reshape %4097 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %4099 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %4100 = tosa.reciprocal %4099 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %4101 = tosa.mul %4098, %4100 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %4102 = tosa.add %4101, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %4103 = tosa.reduce_max %4102 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %4104 = tosa.sub %4102, %4103 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %4105 = tosa.exp %4104 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %4106 = tosa.reduce_sum %4105 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %4107 = tosa.reciprocal %4106 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %4108 = tosa.mul %4105, %4107 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %4109 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %4110 = tosa.add %4108, %4109 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %4111 = tosa.reshape %4110 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %4112 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %4113 = tosa.add %4064, %4112 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %4114 = tosa.reshape %4113 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %4115 = tosa.matmul %4111, %4114 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %4116 = tosa.reshape %4115 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %4117 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4118 = tosa.transpose %4116, %4117 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %4119 = tosa.identity %4118 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %4120 = tosa.reshape %4119 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %4121 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4122 = tosa.transpose %arg349, %4121 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %4123 = tosa.reshape %4120 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_735 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4124 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4123, %4122 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_735 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %4125 = tosa.reshape %4124 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %4126 = tosa.add %4028, %4125 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %4127 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_736 = arith.constant 2 : i32
    %4128 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4126 : tensor<1x40x4096xf32>) outs(%4127 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_736 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %4129 = tosa.reduce_sum %4128 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %4130 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4131 = tosa.reciprocal %4130 : (tensor<1xf32>) -> tensor<1xf32>
    %4132 = tosa.mul %4131, %4129 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4133 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4134 = tosa.add %4132, %4133 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4135 = tosa.rsqrt %4134 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4136 = tosa.mul %4126, %4135 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4137 = tosa.reshape %arg350 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4138 = tosa.mul %4137, %4136 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %4139 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4140 = tosa.transpose %arg351, %4139 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %4141 = tosa.reshape %4138 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_737 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %4142 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4141, %4140 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_737 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %4143 = tosa.reshape %4142 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %4144 = tosa.sigmoid %4143 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %4145 = tosa.mul %4143, %4144 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %4146 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4147 = tosa.transpose %arg352, %4146 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %4148 = tosa.reshape %4138 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_738 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %4149 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4148, %4147 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_738 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %4150 = tosa.reshape %4149 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %4151 = tosa.mul %4145, %4150 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %4152 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4153 = tosa.transpose %arg353, %4152 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %4154 = tosa.reshape %4151 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_739 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4155 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4154, %4153 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_739 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %4156 = tosa.reshape %4155 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %4157 = tosa.add %4126, %4156 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %4158 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_740 = arith.constant 2 : i32
    %4159 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4157 : tensor<1x40x4096xf32>) outs(%4158 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4175 = math.fpowi %in, %c2_i32_740 : f32, i32
      linalg.yield %4175 : f32
    } -> tensor<1x40x4096xf32>
    %4160 = tosa.reduce_sum %4159 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    %4161 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4162 = tosa.reciprocal %4161 : (tensor<1xf32>) -> tensor<1xf32>
    %4163 = tosa.mul %4162, %4160 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4164 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4165 = tosa.add %4163, %4164 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4166 = tosa.rsqrt %4165 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4167 = tosa.mul %4157, %4166 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4168 = tosa.reshape %arg354 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4169 = tosa.mul %4168, %4167 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %4170 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4171 = tosa.transpose %arg355, %4170 : (tensor<32000x4096xf32>, tensor<2xi32>) -> tensor<4096x32000xf32>
    %4172 = tosa.reshape %4169 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_741 = arith.constant dense<0.000000e+00> : tensor<40x32000xf32>
    %4173 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4172, %4171 : tensor<40x4096xf32>, tensor<4096x32000xf32>) outs(%cst_741 : tensor<40x32000xf32>) -> tensor<40x32000xf32>
    %4174 = tosa.reshape %4173 {new_shape = array<i64: 1, 40, 32000>} : (tensor<40x32000xf32>) -> tensor<1x40x32000xf32>
    return %4169, %4174 : tensor<1x40x4096xf32>, tensor<1x40x32000xf32>
  }
}

