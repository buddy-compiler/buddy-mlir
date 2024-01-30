#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map8 = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
#map9 = affine_map<(d0, d1) -> (0, d0, d1)>
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
    ^bb0(%in: i64, %in_807: i64, %out: i1):
      %4044 = arith.cmpi slt, %in, %in_807 : i64
      linalg.yield %4044 : i1
    } -> tensor<40x40xi1>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %13 = tensor.empty() : tensor<40x40xf32>
    %14 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%12, %cst_0 : tensor<40x40xi1>, tensor<40x40xf32>) outs(%13 : tensor<40x40xf32>) {
    ^bb0(%in: i1, %in_807: f32, %out: f32):
      %4044 = arith.select %in, %cst_1, %in_807 : f32
      linalg.yield %4044 : f32
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
    ^bb0(%in: i1, %in_807: f32, %out: f32):
      %4044 = arith.select %in, %cst_3, %in_807 : f32
      linalg.yield %4044 : f32
    } -> tensor<1x1x40x40xf32>
    %25 = tosa.reshape %14 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %26 = tosa.reshape %25 {new_shape = array<i64: 1, 1, 40, 40>} : (tensor<1x40x40xf32>) -> tensor<1x1x40x40xf32>
    %extracted_slice_4 = tensor.extract_slice %26[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_5 = tensor.extract_slice %extracted_slice_4[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %27 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x40xf32>}> : () -> tensor<1x1x40x40xf32>
    %28 = tosa.add %extracted_slice_5, %27 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
    %29 = tosa.add %24, %28 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
    %30 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32 = arith.constant 2 : i32
    %31 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<1x40x4096xf32>) outs(%30 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %32 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%31 : tensor<1x40x4096xf32>) outs(%cst_6 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %33 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %34 = tosa.add %32, %33 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %35 = tosa.rsqrt %34 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %36 = tosa.mul %6, %35 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %37 = tosa.reshape %arg2 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %38 = tosa.mul %37, %36 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %39 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %40 = tosa.transpose %arg3, %39 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %41 = tosa.reshape %38 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %42 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%41, %40 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_7 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %43 = tosa.reshape %42 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %44 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %45 = tosa.transpose %arg4, %44 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %46 = tosa.reshape %38 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %47 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%46, %45 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_8 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %48 = tosa.reshape %47 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %49 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %50 = tosa.transpose %arg5, %49 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %51 = tosa.reshape %38 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %52 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%51, %50 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_9 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %53 = tosa.reshape %52 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %54 = tosa.reshape %43 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %55 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %56 = tosa.transpose %54, %55 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %57 = tosa.reshape %48 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %58 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %59 = tosa.transpose %57, %58 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %60 = tosa.reshape %53 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %61 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %62 = tosa.transpose %60, %61 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_10 = tensor.extract_slice %arg6[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_11 = tensor.extract_slice %extracted_slice_10[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_12 = tensor.extract_slice %extracted_slice_11[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_13 = tensor.extract_slice %arg7[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_14 = tensor.extract_slice %extracted_slice_13[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_15 = tensor.extract_slice %extracted_slice_14[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %63 = tensor.empty() : tensor<1x40x128xf32>
    %64 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_12 : tensor<1x1x40x128xf32>) outs(%63 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %65 = tensor.empty() : tensor<40x128xf32>
    %66 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%64 : tensor<1x40x128xf32>) outs(%65 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %67 = tensor.empty() : tensor<1x40x128xf32>
    %68 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_15 : tensor<1x1x40x128xf32>) outs(%67 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %69 = tensor.empty() : tensor<40x128xf32>
    %70 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%68 : tensor<1x40x128xf32>) outs(%69 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %71 = tensor.empty() : tensor<1x40x128xf32>
    %72 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%71 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %66[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %73 = tosa.reshape %72 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %74 = tensor.empty() : tensor<1x40x128xf32>
    %75 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%74 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %70[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %76 = tosa.reshape %75 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %77 = tosa.mul %56, %73 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_16 = tensor.extract_slice %56[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_17 = tensor.extract_slice %56[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %78 = tensor.empty() : tensor<1x32x40x64xf32>
    %79 = linalg.negf ins(%extracted_slice_17 : tensor<1x32x40x64xf32>) outs(%78 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %80 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice = tensor.insert_slice %79 into %80[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_18 = tensor.insert_slice %extracted_slice_16 into %inserted_slice[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %81 = tosa.mul %inserted_slice_18, %76 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %82 = tosa.add %77, %81 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %83 = tosa.mul %59, %73 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_19 = tensor.extract_slice %59[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_20 = tensor.extract_slice %59[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %84 = tensor.empty() : tensor<1x32x40x64xf32>
    %85 = linalg.negf ins(%extracted_slice_20 : tensor<1x32x40x64xf32>) outs(%84 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %86 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_21 = tensor.insert_slice %85 into %86[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_22 = tensor.insert_slice %extracted_slice_19 into %inserted_slice_21[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %87 = tosa.mul %inserted_slice_22, %76 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %88 = tosa.add %83, %87 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %89 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %90 = tosa.transpose %88, %89 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %91 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %92 = tosa.add %82, %91 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %93 = tosa.reshape %92 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %94 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %95 = tosa.add %90, %94 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %96 = tosa.reshape %95 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %97 = tosa.matmul %93, %96 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %98 = tosa.reshape %97 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %99 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %100 = tosa.reciprocal %99 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %101 = tosa.mul %98, %100 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %102 = tosa.add %101, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %103 = tosa.reduce_max %102 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %104 = tosa.sub %102, %103 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %105 = tosa.exp %104 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %106 = tosa.reduce_sum %105 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %107 = tosa.reciprocal %106 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %108 = tosa.mul %105, %107 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %109 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %110 = tosa.add %108, %109 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %111 = tosa.reshape %110 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %112 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %113 = tosa.add %62, %112 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %114 = tosa.reshape %113 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %115 = tosa.matmul %111, %114 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %116 = tosa.reshape %115 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %117 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %118 = tosa.transpose %116, %117 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %119 = tosa.identity %118 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %120 = tosa.reshape %119 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %121 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %122 = tosa.transpose %arg8, %121 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %123 = tosa.reshape %120 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_23 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %124 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%123, %122 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_23 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %125 = tosa.reshape %124 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %126 = tosa.add %6, %125 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %127 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_24 = arith.constant 2 : i32
    %128 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%126 : tensor<1x40x4096xf32>) outs(%127 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_24 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_25 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %129 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%128 : tensor<1x40x4096xf32>) outs(%cst_25 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %130 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %131 = tosa.add %129, %130 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %132 = tosa.rsqrt %131 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %133 = tosa.mul %126, %132 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %134 = tosa.reshape %arg9 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %135 = tosa.mul %134, %133 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %136 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %137 = tosa.transpose %arg10, %136 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %138 = tosa.reshape %135 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_26 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %139 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%138, %137 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_26 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %140 = tosa.reshape %139 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %141 = tosa.sigmoid %140 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %142 = tosa.mul %140, %141 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %143 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %144 = tosa.transpose %arg11, %143 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %145 = tosa.reshape %135 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_27 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %146 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%145, %144 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_27 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %147 = tosa.reshape %146 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %148 = tosa.mul %142, %147 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %149 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %150 = tosa.transpose %arg12, %149 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %151 = tosa.reshape %148 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_28 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %152 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%151, %150 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_28 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %153 = tosa.reshape %152 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %154 = tosa.add %126, %153 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %155 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_29 = arith.constant 2 : i32
    %156 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%154 : tensor<1x40x4096xf32>) outs(%155 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_29 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_30 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %157 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%156 : tensor<1x40x4096xf32>) outs(%cst_30 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %158 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %159 = tosa.add %157, %158 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %160 = tosa.rsqrt %159 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %161 = tosa.mul %154, %160 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %162 = tosa.reshape %arg13 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %163 = tosa.mul %162, %161 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %164 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %165 = tosa.transpose %arg14, %164 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %166 = tosa.reshape %163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_31 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %167 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%166, %165 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_31 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %168 = tosa.reshape %167 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %169 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %170 = tosa.transpose %arg15, %169 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %171 = tosa.reshape %163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_32 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %172 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%171, %170 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_32 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %173 = tosa.reshape %172 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %174 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %175 = tosa.transpose %arg16, %174 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %176 = tosa.reshape %163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_33 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %177 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%176, %175 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_33 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %178 = tosa.reshape %177 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %179 = tosa.reshape %168 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %180 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %181 = tosa.transpose %179, %180 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %182 = tosa.reshape %173 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %183 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %184 = tosa.transpose %182, %183 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %185 = tosa.reshape %178 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %186 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %187 = tosa.transpose %185, %186 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_34 = tensor.extract_slice %arg17[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_35 = tensor.extract_slice %extracted_slice_34[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_36 = tensor.extract_slice %extracted_slice_35[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_37 = tensor.extract_slice %arg18[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_38 = tensor.extract_slice %extracted_slice_37[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_39 = tensor.extract_slice %extracted_slice_38[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %188 = tensor.empty() : tensor<1x40x128xf32>
    %189 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_36 : tensor<1x1x40x128xf32>) outs(%188 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %190 = tensor.empty() : tensor<40x128xf32>
    %191 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%189 : tensor<1x40x128xf32>) outs(%190 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %192 = tensor.empty() : tensor<1x40x128xf32>
    %193 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_39 : tensor<1x1x40x128xf32>) outs(%192 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %194 = tensor.empty() : tensor<40x128xf32>
    %195 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%193 : tensor<1x40x128xf32>) outs(%194 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %196 = tensor.empty() : tensor<1x40x128xf32>
    %197 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%196 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %191[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %198 = tosa.reshape %197 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %199 = tensor.empty() : tensor<1x40x128xf32>
    %200 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%199 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %195[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %201 = tosa.reshape %200 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %202 = tosa.mul %181, %198 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_40 = tensor.extract_slice %181[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_41 = tensor.extract_slice %181[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %203 = tensor.empty() : tensor<1x32x40x64xf32>
    %204 = linalg.negf ins(%extracted_slice_41 : tensor<1x32x40x64xf32>) outs(%203 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %205 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_42 = tensor.insert_slice %204 into %205[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_43 = tensor.insert_slice %extracted_slice_40 into %inserted_slice_42[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %206 = tosa.mul %inserted_slice_43, %201 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %207 = tosa.add %202, %206 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %208 = tosa.mul %184, %198 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_44 = tensor.extract_slice %184[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_45 = tensor.extract_slice %184[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %209 = tensor.empty() : tensor<1x32x40x64xf32>
    %210 = linalg.negf ins(%extracted_slice_45 : tensor<1x32x40x64xf32>) outs(%209 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %211 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_46 = tensor.insert_slice %210 into %211[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_47 = tensor.insert_slice %extracted_slice_44 into %inserted_slice_46[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %212 = tosa.mul %inserted_slice_47, %201 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %213 = tosa.add %208, %212 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %214 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %215 = tosa.transpose %213, %214 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %216 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %217 = tosa.add %207, %216 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %218 = tosa.reshape %217 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %219 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %220 = tosa.add %215, %219 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %221 = tosa.reshape %220 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %222 = tosa.matmul %218, %221 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %223 = tosa.reshape %222 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %224 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %225 = tosa.reciprocal %224 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %226 = tosa.mul %223, %225 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %227 = tosa.add %226, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %228 = tosa.reduce_max %227 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %229 = tosa.sub %227, %228 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %230 = tosa.exp %229 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %231 = tosa.reduce_sum %230 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %232 = tosa.reciprocal %231 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %233 = tosa.mul %230, %232 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %234 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %235 = tosa.add %233, %234 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %236 = tosa.reshape %235 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %237 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %238 = tosa.add %187, %237 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %239 = tosa.reshape %238 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %240 = tosa.matmul %236, %239 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %241 = tosa.reshape %240 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %242 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %243 = tosa.transpose %241, %242 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %244 = tosa.identity %243 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %245 = tosa.reshape %244 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %246 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %247 = tosa.transpose %arg19, %246 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %248 = tosa.reshape %245 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_48 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %249 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%248, %247 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_48 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %250 = tosa.reshape %249 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %251 = tosa.add %154, %250 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %252 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_49 = arith.constant 2 : i32
    %253 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%251 : tensor<1x40x4096xf32>) outs(%252 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_49 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_50 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %254 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%253 : tensor<1x40x4096xf32>) outs(%cst_50 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %255 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %256 = tosa.add %254, %255 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %257 = tosa.rsqrt %256 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %258 = tosa.mul %251, %257 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %259 = tosa.reshape %arg20 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %260 = tosa.mul %259, %258 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %261 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %262 = tosa.transpose %arg21, %261 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %263 = tosa.reshape %260 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_51 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %264 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%263, %262 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_51 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %265 = tosa.reshape %264 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %266 = tosa.sigmoid %265 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %267 = tosa.mul %265, %266 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %268 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %269 = tosa.transpose %arg22, %268 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %270 = tosa.reshape %260 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_52 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %271 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%270, %269 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_52 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %272 = tosa.reshape %271 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %273 = tosa.mul %267, %272 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %274 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %275 = tosa.transpose %arg23, %274 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %276 = tosa.reshape %273 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_53 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %277 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%276, %275 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_53 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %278 = tosa.reshape %277 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %279 = tosa.add %251, %278 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %280 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_54 = arith.constant 2 : i32
    %281 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%279 : tensor<1x40x4096xf32>) outs(%280 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_54 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_55 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %282 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%281 : tensor<1x40x4096xf32>) outs(%cst_55 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %283 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %284 = tosa.add %282, %283 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %285 = tosa.rsqrt %284 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %286 = tosa.mul %279, %285 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %287 = tosa.reshape %arg24 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %288 = tosa.mul %287, %286 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %289 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %290 = tosa.transpose %arg25, %289 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %291 = tosa.reshape %288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_56 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %292 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%291, %290 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_56 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %293 = tosa.reshape %292 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %294 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %295 = tosa.transpose %arg26, %294 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %296 = tosa.reshape %288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_57 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %297 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%296, %295 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_57 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %298 = tosa.reshape %297 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %299 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %300 = tosa.transpose %arg27, %299 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %301 = tosa.reshape %288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_58 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %302 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%301, %300 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_58 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %303 = tosa.reshape %302 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %304 = tosa.reshape %293 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %305 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %306 = tosa.transpose %304, %305 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %307 = tosa.reshape %298 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %308 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %309 = tosa.transpose %307, %308 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %310 = tosa.reshape %303 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %311 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %312 = tosa.transpose %310, %311 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_59 = tensor.extract_slice %arg28[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_60 = tensor.extract_slice %extracted_slice_59[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_61 = tensor.extract_slice %extracted_slice_60[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_62 = tensor.extract_slice %arg29[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_63 = tensor.extract_slice %extracted_slice_62[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_64 = tensor.extract_slice %extracted_slice_63[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %313 = tensor.empty() : tensor<1x40x128xf32>
    %314 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_61 : tensor<1x1x40x128xf32>) outs(%313 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %315 = tensor.empty() : tensor<40x128xf32>
    %316 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%314 : tensor<1x40x128xf32>) outs(%315 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %317 = tensor.empty() : tensor<1x40x128xf32>
    %318 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_64 : tensor<1x1x40x128xf32>) outs(%317 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %319 = tensor.empty() : tensor<40x128xf32>
    %320 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%318 : tensor<1x40x128xf32>) outs(%319 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %321 = tensor.empty() : tensor<1x40x128xf32>
    %322 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%321 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %316[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %323 = tosa.reshape %322 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %324 = tensor.empty() : tensor<1x40x128xf32>
    %325 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%324 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %320[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %326 = tosa.reshape %325 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %327 = tosa.mul %306, %323 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_65 = tensor.extract_slice %306[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_66 = tensor.extract_slice %306[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %328 = tensor.empty() : tensor<1x32x40x64xf32>
    %329 = linalg.negf ins(%extracted_slice_66 : tensor<1x32x40x64xf32>) outs(%328 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %330 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_67 = tensor.insert_slice %329 into %330[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_68 = tensor.insert_slice %extracted_slice_65 into %inserted_slice_67[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %331 = tosa.mul %inserted_slice_68, %326 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %332 = tosa.add %327, %331 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %333 = tosa.mul %309, %323 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_69 = tensor.extract_slice %309[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_70 = tensor.extract_slice %309[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %334 = tensor.empty() : tensor<1x32x40x64xf32>
    %335 = linalg.negf ins(%extracted_slice_70 : tensor<1x32x40x64xf32>) outs(%334 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %336 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_71 = tensor.insert_slice %335 into %336[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_72 = tensor.insert_slice %extracted_slice_69 into %inserted_slice_71[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %337 = tosa.mul %inserted_slice_72, %326 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %338 = tosa.add %333, %337 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %339 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %340 = tosa.transpose %338, %339 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %341 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %342 = tosa.add %332, %341 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %343 = tosa.reshape %342 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %344 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %345 = tosa.add %340, %344 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %346 = tosa.reshape %345 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %347 = tosa.matmul %343, %346 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %348 = tosa.reshape %347 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %349 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %350 = tosa.reciprocal %349 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %351 = tosa.mul %348, %350 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %352 = tosa.add %351, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %353 = tosa.reduce_max %352 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %354 = tosa.sub %352, %353 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %355 = tosa.exp %354 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %356 = tosa.reduce_sum %355 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %357 = tosa.reciprocal %356 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %358 = tosa.mul %355, %357 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %359 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %360 = tosa.add %358, %359 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %361 = tosa.reshape %360 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %362 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %363 = tosa.add %312, %362 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %364 = tosa.reshape %363 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %365 = tosa.matmul %361, %364 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %366 = tosa.reshape %365 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %367 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %368 = tosa.transpose %366, %367 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %369 = tosa.identity %368 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %370 = tosa.reshape %369 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %371 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %372 = tosa.transpose %arg30, %371 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %373 = tosa.reshape %370 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_73 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %374 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%373, %372 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_73 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %375 = tosa.reshape %374 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %376 = tosa.add %279, %375 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %377 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_74 = arith.constant 2 : i32
    %378 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%376 : tensor<1x40x4096xf32>) outs(%377 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_74 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_75 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %379 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%378 : tensor<1x40x4096xf32>) outs(%cst_75 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %380 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %381 = tosa.add %379, %380 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %382 = tosa.rsqrt %381 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %383 = tosa.mul %376, %382 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %384 = tosa.reshape %arg31 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %385 = tosa.mul %384, %383 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %386 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %387 = tosa.transpose %arg32, %386 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %388 = tosa.reshape %385 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_76 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %389 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%388, %387 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_76 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %390 = tosa.reshape %389 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %391 = tosa.sigmoid %390 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %392 = tosa.mul %390, %391 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %393 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %394 = tosa.transpose %arg33, %393 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %395 = tosa.reshape %385 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_77 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %396 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%395, %394 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_77 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %397 = tosa.reshape %396 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %398 = tosa.mul %392, %397 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %399 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %400 = tosa.transpose %arg34, %399 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %401 = tosa.reshape %398 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_78 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %402 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%401, %400 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_78 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %403 = tosa.reshape %402 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %404 = tosa.add %376, %403 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %405 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_79 = arith.constant 2 : i32
    %406 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%404 : tensor<1x40x4096xf32>) outs(%405 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_79 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_80 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %407 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%406 : tensor<1x40x4096xf32>) outs(%cst_80 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %408 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %409 = tosa.add %407, %408 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %410 = tosa.rsqrt %409 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %411 = tosa.mul %404, %410 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %412 = tosa.reshape %arg35 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %413 = tosa.mul %412, %411 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %414 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %415 = tosa.transpose %arg36, %414 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %416 = tosa.reshape %413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_81 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %417 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%416, %415 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_81 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %418 = tosa.reshape %417 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %419 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %420 = tosa.transpose %arg37, %419 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %421 = tosa.reshape %413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_82 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %422 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%421, %420 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_82 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %423 = tosa.reshape %422 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %424 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %425 = tosa.transpose %arg38, %424 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %426 = tosa.reshape %413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_83 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %427 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%426, %425 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_83 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %428 = tosa.reshape %427 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %429 = tosa.reshape %418 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %430 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %431 = tosa.transpose %429, %430 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %432 = tosa.reshape %423 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %433 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %434 = tosa.transpose %432, %433 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %435 = tosa.reshape %428 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %436 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %437 = tosa.transpose %435, %436 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_84 = tensor.extract_slice %arg39[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_85 = tensor.extract_slice %extracted_slice_84[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_86 = tensor.extract_slice %extracted_slice_85[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_87 = tensor.extract_slice %arg40[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_88 = tensor.extract_slice %extracted_slice_87[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_89 = tensor.extract_slice %extracted_slice_88[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %438 = tensor.empty() : tensor<1x40x128xf32>
    %439 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_86 : tensor<1x1x40x128xf32>) outs(%438 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %440 = tensor.empty() : tensor<40x128xf32>
    %441 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%439 : tensor<1x40x128xf32>) outs(%440 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %442 = tensor.empty() : tensor<1x40x128xf32>
    %443 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_89 : tensor<1x1x40x128xf32>) outs(%442 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %444 = tensor.empty() : tensor<40x128xf32>
    %445 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%443 : tensor<1x40x128xf32>) outs(%444 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %446 = tensor.empty() : tensor<1x40x128xf32>
    %447 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%446 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %441[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %448 = tosa.reshape %447 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %449 = tensor.empty() : tensor<1x40x128xf32>
    %450 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%449 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %445[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %451 = tosa.reshape %450 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %452 = tosa.mul %431, %448 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_90 = tensor.extract_slice %431[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_91 = tensor.extract_slice %431[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %453 = tensor.empty() : tensor<1x32x40x64xf32>
    %454 = linalg.negf ins(%extracted_slice_91 : tensor<1x32x40x64xf32>) outs(%453 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %455 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_92 = tensor.insert_slice %454 into %455[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_93 = tensor.insert_slice %extracted_slice_90 into %inserted_slice_92[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %456 = tosa.mul %inserted_slice_93, %451 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %457 = tosa.add %452, %456 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %458 = tosa.mul %434, %448 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_94 = tensor.extract_slice %434[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_95 = tensor.extract_slice %434[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %459 = tensor.empty() : tensor<1x32x40x64xf32>
    %460 = linalg.negf ins(%extracted_slice_95 : tensor<1x32x40x64xf32>) outs(%459 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %461 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_96 = tensor.insert_slice %460 into %461[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_97 = tensor.insert_slice %extracted_slice_94 into %inserted_slice_96[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %462 = tosa.mul %inserted_slice_97, %451 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %463 = tosa.add %458, %462 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %464 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %465 = tosa.transpose %463, %464 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %466 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %467 = tosa.add %457, %466 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %468 = tosa.reshape %467 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %469 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %470 = tosa.add %465, %469 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %471 = tosa.reshape %470 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %472 = tosa.matmul %468, %471 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %473 = tosa.reshape %472 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %474 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %475 = tosa.reciprocal %474 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %476 = tosa.mul %473, %475 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %477 = tosa.add %476, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %478 = tosa.reduce_max %477 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %479 = tosa.sub %477, %478 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %480 = tosa.exp %479 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %481 = tosa.reduce_sum %480 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %482 = tosa.reciprocal %481 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %483 = tosa.mul %480, %482 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %484 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %485 = tosa.add %483, %484 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %486 = tosa.reshape %485 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %487 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %488 = tosa.add %437, %487 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %489 = tosa.reshape %488 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %490 = tosa.matmul %486, %489 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %491 = tosa.reshape %490 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %492 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %493 = tosa.transpose %491, %492 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %494 = tosa.identity %493 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %495 = tosa.reshape %494 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %496 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %497 = tosa.transpose %arg41, %496 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %498 = tosa.reshape %495 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_98 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %499 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%498, %497 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_98 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %500 = tosa.reshape %499 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %501 = tosa.add %404, %500 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %502 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_99 = arith.constant 2 : i32
    %503 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%501 : tensor<1x40x4096xf32>) outs(%502 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_99 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_100 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %504 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%503 : tensor<1x40x4096xf32>) outs(%cst_100 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %505 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %506 = tosa.add %504, %505 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %507 = tosa.rsqrt %506 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %508 = tosa.mul %501, %507 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %509 = tosa.reshape %arg42 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %510 = tosa.mul %509, %508 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %511 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %512 = tosa.transpose %arg43, %511 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %513 = tosa.reshape %510 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_101 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %514 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%513, %512 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_101 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %515 = tosa.reshape %514 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %516 = tosa.sigmoid %515 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %517 = tosa.mul %515, %516 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %518 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %519 = tosa.transpose %arg44, %518 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %520 = tosa.reshape %510 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_102 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %521 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%520, %519 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_102 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %522 = tosa.reshape %521 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %523 = tosa.mul %517, %522 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %524 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %525 = tosa.transpose %arg45, %524 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %526 = tosa.reshape %523 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_103 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %527 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%526, %525 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_103 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %528 = tosa.reshape %527 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %529 = tosa.add %501, %528 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %530 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_104 = arith.constant 2 : i32
    %531 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%529 : tensor<1x40x4096xf32>) outs(%530 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_104 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_105 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %532 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%531 : tensor<1x40x4096xf32>) outs(%cst_105 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %533 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %534 = tosa.add %532, %533 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %535 = tosa.rsqrt %534 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %536 = tosa.mul %529, %535 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %537 = tosa.reshape %arg46 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %538 = tosa.mul %537, %536 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %539 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %540 = tosa.transpose %arg47, %539 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %541 = tosa.reshape %538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_106 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %542 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%541, %540 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_106 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %543 = tosa.reshape %542 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %544 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %545 = tosa.transpose %arg48, %544 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %546 = tosa.reshape %538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_107 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %547 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%546, %545 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_107 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %548 = tosa.reshape %547 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %549 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %550 = tosa.transpose %arg49, %549 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %551 = tosa.reshape %538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_108 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %552 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%551, %550 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_108 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %553 = tosa.reshape %552 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %554 = tosa.reshape %543 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %555 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %556 = tosa.transpose %554, %555 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %557 = tosa.reshape %548 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %558 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %559 = tosa.transpose %557, %558 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %560 = tosa.reshape %553 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %561 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %562 = tosa.transpose %560, %561 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_109 = tensor.extract_slice %arg50[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_110 = tensor.extract_slice %extracted_slice_109[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_111 = tensor.extract_slice %extracted_slice_110[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_112 = tensor.extract_slice %arg51[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_113 = tensor.extract_slice %extracted_slice_112[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_114 = tensor.extract_slice %extracted_slice_113[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %563 = tensor.empty() : tensor<1x40x128xf32>
    %564 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_111 : tensor<1x1x40x128xf32>) outs(%563 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %565 = tensor.empty() : tensor<40x128xf32>
    %566 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%564 : tensor<1x40x128xf32>) outs(%565 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %567 = tensor.empty() : tensor<1x40x128xf32>
    %568 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_114 : tensor<1x1x40x128xf32>) outs(%567 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %569 = tensor.empty() : tensor<40x128xf32>
    %570 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%568 : tensor<1x40x128xf32>) outs(%569 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %571 = tensor.empty() : tensor<1x40x128xf32>
    %572 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%571 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %566[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %573 = tosa.reshape %572 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %574 = tensor.empty() : tensor<1x40x128xf32>
    %575 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%574 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %570[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %576 = tosa.reshape %575 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %577 = tosa.mul %556, %573 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_115 = tensor.extract_slice %556[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_116 = tensor.extract_slice %556[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %578 = tensor.empty() : tensor<1x32x40x64xf32>
    %579 = linalg.negf ins(%extracted_slice_116 : tensor<1x32x40x64xf32>) outs(%578 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %580 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_117 = tensor.insert_slice %579 into %580[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_118 = tensor.insert_slice %extracted_slice_115 into %inserted_slice_117[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %581 = tosa.mul %inserted_slice_118, %576 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %582 = tosa.add %577, %581 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %583 = tosa.mul %559, %573 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_119 = tensor.extract_slice %559[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_120 = tensor.extract_slice %559[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %584 = tensor.empty() : tensor<1x32x40x64xf32>
    %585 = linalg.negf ins(%extracted_slice_120 : tensor<1x32x40x64xf32>) outs(%584 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %586 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_121 = tensor.insert_slice %585 into %586[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_122 = tensor.insert_slice %extracted_slice_119 into %inserted_slice_121[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %587 = tosa.mul %inserted_slice_122, %576 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %588 = tosa.add %583, %587 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %589 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %590 = tosa.transpose %588, %589 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %591 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %592 = tosa.add %582, %591 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %593 = tosa.reshape %592 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %594 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %595 = tosa.add %590, %594 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %596 = tosa.reshape %595 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %597 = tosa.matmul %593, %596 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %598 = tosa.reshape %597 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %599 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %600 = tosa.reciprocal %599 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %601 = tosa.mul %598, %600 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %602 = tosa.add %601, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %603 = tosa.reduce_max %602 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %604 = tosa.sub %602, %603 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %605 = tosa.exp %604 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %606 = tosa.reduce_sum %605 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %607 = tosa.reciprocal %606 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %608 = tosa.mul %605, %607 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %609 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %610 = tosa.add %608, %609 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %611 = tosa.reshape %610 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %612 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %613 = tosa.add %562, %612 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %614 = tosa.reshape %613 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %615 = tosa.matmul %611, %614 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %616 = tosa.reshape %615 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %617 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %618 = tosa.transpose %616, %617 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %619 = tosa.identity %618 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %620 = tosa.reshape %619 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %621 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %622 = tosa.transpose %arg52, %621 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %623 = tosa.reshape %620 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_123 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %624 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%623, %622 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_123 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %625 = tosa.reshape %624 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %626 = tosa.add %529, %625 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %627 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_124 = arith.constant 2 : i32
    %628 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%626 : tensor<1x40x4096xf32>) outs(%627 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_124 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_125 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %629 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%628 : tensor<1x40x4096xf32>) outs(%cst_125 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %630 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %631 = tosa.add %629, %630 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %632 = tosa.rsqrt %631 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %633 = tosa.mul %626, %632 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %634 = tosa.reshape %arg53 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %635 = tosa.mul %634, %633 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %636 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %637 = tosa.transpose %arg54, %636 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %638 = tosa.reshape %635 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_126 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %639 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%638, %637 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_126 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %640 = tosa.reshape %639 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %641 = tosa.sigmoid %640 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %642 = tosa.mul %640, %641 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %643 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %644 = tosa.transpose %arg55, %643 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %645 = tosa.reshape %635 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_127 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %646 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%645, %644 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_127 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %647 = tosa.reshape %646 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %648 = tosa.mul %642, %647 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %649 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %650 = tosa.transpose %arg56, %649 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %651 = tosa.reshape %648 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_128 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %652 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%651, %650 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_128 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %653 = tosa.reshape %652 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %654 = tosa.add %626, %653 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %655 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_129 = arith.constant 2 : i32
    %656 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%654 : tensor<1x40x4096xf32>) outs(%655 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_129 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_130 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %657 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%656 : tensor<1x40x4096xf32>) outs(%cst_130 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %658 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %659 = tosa.add %657, %658 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %660 = tosa.rsqrt %659 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %661 = tosa.mul %654, %660 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %662 = tosa.reshape %arg57 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %663 = tosa.mul %662, %661 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %664 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %665 = tosa.transpose %arg58, %664 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %666 = tosa.reshape %663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_131 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %667 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%666, %665 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_131 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %668 = tosa.reshape %667 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %669 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %670 = tosa.transpose %arg59, %669 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %671 = tosa.reshape %663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_132 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %672 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%671, %670 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_132 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %673 = tosa.reshape %672 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %674 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %675 = tosa.transpose %arg60, %674 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %676 = tosa.reshape %663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_133 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %677 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%676, %675 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_133 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %678 = tosa.reshape %677 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %679 = tosa.reshape %668 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %680 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %681 = tosa.transpose %679, %680 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %682 = tosa.reshape %673 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %683 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %684 = tosa.transpose %682, %683 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %685 = tosa.reshape %678 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %686 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %687 = tosa.transpose %685, %686 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_134 = tensor.extract_slice %arg61[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_135 = tensor.extract_slice %extracted_slice_134[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_136 = tensor.extract_slice %extracted_slice_135[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_137 = tensor.extract_slice %arg62[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_138 = tensor.extract_slice %extracted_slice_137[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_139 = tensor.extract_slice %extracted_slice_138[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %688 = tensor.empty() : tensor<1x40x128xf32>
    %689 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_136 : tensor<1x1x40x128xf32>) outs(%688 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %690 = tensor.empty() : tensor<40x128xf32>
    %691 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%689 : tensor<1x40x128xf32>) outs(%690 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %692 = tensor.empty() : tensor<1x40x128xf32>
    %693 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_139 : tensor<1x1x40x128xf32>) outs(%692 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %694 = tensor.empty() : tensor<40x128xf32>
    %695 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%693 : tensor<1x40x128xf32>) outs(%694 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %696 = tensor.empty() : tensor<1x40x128xf32>
    %697 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%696 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %691[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %698 = tosa.reshape %697 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %699 = tensor.empty() : tensor<1x40x128xf32>
    %700 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%699 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %695[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %701 = tosa.reshape %700 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %702 = tosa.mul %681, %698 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_140 = tensor.extract_slice %681[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_141 = tensor.extract_slice %681[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %703 = tensor.empty() : tensor<1x32x40x64xf32>
    %704 = linalg.negf ins(%extracted_slice_141 : tensor<1x32x40x64xf32>) outs(%703 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %705 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_142 = tensor.insert_slice %704 into %705[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_143 = tensor.insert_slice %extracted_slice_140 into %inserted_slice_142[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %706 = tosa.mul %inserted_slice_143, %701 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %707 = tosa.add %702, %706 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %708 = tosa.mul %684, %698 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_144 = tensor.extract_slice %684[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_145 = tensor.extract_slice %684[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %709 = tensor.empty() : tensor<1x32x40x64xf32>
    %710 = linalg.negf ins(%extracted_slice_145 : tensor<1x32x40x64xf32>) outs(%709 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %711 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_146 = tensor.insert_slice %710 into %711[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_147 = tensor.insert_slice %extracted_slice_144 into %inserted_slice_146[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %712 = tosa.mul %inserted_slice_147, %701 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %713 = tosa.add %708, %712 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %714 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %715 = tosa.transpose %713, %714 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %716 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %717 = tosa.add %707, %716 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %718 = tosa.reshape %717 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %719 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %720 = tosa.add %715, %719 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %721 = tosa.reshape %720 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %722 = tosa.matmul %718, %721 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %723 = tosa.reshape %722 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %724 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %725 = tosa.reciprocal %724 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %726 = tosa.mul %723, %725 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %727 = tosa.add %726, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %728 = tosa.reduce_max %727 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %729 = tosa.sub %727, %728 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %730 = tosa.exp %729 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %731 = tosa.reduce_sum %730 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %732 = tosa.reciprocal %731 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %733 = tosa.mul %730, %732 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %734 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %735 = tosa.add %733, %734 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %736 = tosa.reshape %735 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %737 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %738 = tosa.add %687, %737 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %739 = tosa.reshape %738 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %740 = tosa.matmul %736, %739 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %741 = tosa.reshape %740 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %742 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %743 = tosa.transpose %741, %742 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %744 = tosa.identity %743 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %745 = tosa.reshape %744 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %746 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %747 = tosa.transpose %arg63, %746 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %748 = tosa.reshape %745 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_148 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %749 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%748, %747 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_148 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %750 = tosa.reshape %749 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %751 = tosa.add %654, %750 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %752 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_149 = arith.constant 2 : i32
    %753 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%751 : tensor<1x40x4096xf32>) outs(%752 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_149 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_150 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %754 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%753 : tensor<1x40x4096xf32>) outs(%cst_150 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %755 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %756 = tosa.add %754, %755 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %757 = tosa.rsqrt %756 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %758 = tosa.mul %751, %757 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %759 = tosa.reshape %arg64 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %760 = tosa.mul %759, %758 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %761 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %762 = tosa.transpose %arg65, %761 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %763 = tosa.reshape %760 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_151 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %764 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%763, %762 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_151 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %765 = tosa.reshape %764 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %766 = tosa.sigmoid %765 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %767 = tosa.mul %765, %766 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %768 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %769 = tosa.transpose %arg66, %768 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %770 = tosa.reshape %760 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_152 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %771 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%770, %769 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_152 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %772 = tosa.reshape %771 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %773 = tosa.mul %767, %772 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %774 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %775 = tosa.transpose %arg67, %774 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %776 = tosa.reshape %773 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_153 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %777 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%776, %775 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_153 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %778 = tosa.reshape %777 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %779 = tosa.add %751, %778 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %780 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_154 = arith.constant 2 : i32
    %781 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%779 : tensor<1x40x4096xf32>) outs(%780 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_154 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_155 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %782 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%781 : tensor<1x40x4096xf32>) outs(%cst_155 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %783 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %784 = tosa.add %782, %783 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %785 = tosa.rsqrt %784 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %786 = tosa.mul %779, %785 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %787 = tosa.reshape %arg68 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %788 = tosa.mul %787, %786 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %789 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %790 = tosa.transpose %arg69, %789 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %791 = tosa.reshape %788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_156 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %792 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%791, %790 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_156 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %793 = tosa.reshape %792 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %794 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %795 = tosa.transpose %arg70, %794 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %796 = tosa.reshape %788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_157 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %797 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%796, %795 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_157 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %798 = tosa.reshape %797 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %799 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %800 = tosa.transpose %arg71, %799 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %801 = tosa.reshape %788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_158 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %802 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%801, %800 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_158 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %803 = tosa.reshape %802 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %804 = tosa.reshape %793 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %805 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %806 = tosa.transpose %804, %805 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %807 = tosa.reshape %798 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %808 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %809 = tosa.transpose %807, %808 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %810 = tosa.reshape %803 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %811 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %812 = tosa.transpose %810, %811 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_159 = tensor.extract_slice %arg72[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_160 = tensor.extract_slice %extracted_slice_159[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_161 = tensor.extract_slice %extracted_slice_160[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_162 = tensor.extract_slice %arg73[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_163 = tensor.extract_slice %extracted_slice_162[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_164 = tensor.extract_slice %extracted_slice_163[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %813 = tensor.empty() : tensor<1x40x128xf32>
    %814 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_161 : tensor<1x1x40x128xf32>) outs(%813 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %815 = tensor.empty() : tensor<40x128xf32>
    %816 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%814 : tensor<1x40x128xf32>) outs(%815 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %817 = tensor.empty() : tensor<1x40x128xf32>
    %818 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_164 : tensor<1x1x40x128xf32>) outs(%817 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %819 = tensor.empty() : tensor<40x128xf32>
    %820 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%818 : tensor<1x40x128xf32>) outs(%819 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %821 = tensor.empty() : tensor<1x40x128xf32>
    %822 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%821 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %816[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %823 = tosa.reshape %822 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %824 = tensor.empty() : tensor<1x40x128xf32>
    %825 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%824 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %820[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %826 = tosa.reshape %825 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %827 = tosa.mul %806, %823 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_165 = tensor.extract_slice %806[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_166 = tensor.extract_slice %806[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %828 = tensor.empty() : tensor<1x32x40x64xf32>
    %829 = linalg.negf ins(%extracted_slice_166 : tensor<1x32x40x64xf32>) outs(%828 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %830 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_167 = tensor.insert_slice %829 into %830[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_168 = tensor.insert_slice %extracted_slice_165 into %inserted_slice_167[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %831 = tosa.mul %inserted_slice_168, %826 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %832 = tosa.add %827, %831 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %833 = tosa.mul %809, %823 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_169 = tensor.extract_slice %809[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_170 = tensor.extract_slice %809[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %834 = tensor.empty() : tensor<1x32x40x64xf32>
    %835 = linalg.negf ins(%extracted_slice_170 : tensor<1x32x40x64xf32>) outs(%834 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %836 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_171 = tensor.insert_slice %835 into %836[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_172 = tensor.insert_slice %extracted_slice_169 into %inserted_slice_171[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %837 = tosa.mul %inserted_slice_172, %826 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %838 = tosa.add %833, %837 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %839 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %840 = tosa.transpose %838, %839 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %841 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %842 = tosa.add %832, %841 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %843 = tosa.reshape %842 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %844 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %845 = tosa.add %840, %844 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %846 = tosa.reshape %845 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %847 = tosa.matmul %843, %846 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %848 = tosa.reshape %847 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %849 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %850 = tosa.reciprocal %849 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %851 = tosa.mul %848, %850 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %852 = tosa.add %851, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %853 = tosa.reduce_max %852 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %854 = tosa.sub %852, %853 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %855 = tosa.exp %854 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %856 = tosa.reduce_sum %855 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %857 = tosa.reciprocal %856 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %858 = tosa.mul %855, %857 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %859 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %860 = tosa.add %858, %859 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %861 = tosa.reshape %860 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %862 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %863 = tosa.add %812, %862 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %864 = tosa.reshape %863 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %865 = tosa.matmul %861, %864 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %866 = tosa.reshape %865 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %867 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %868 = tosa.transpose %866, %867 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %869 = tosa.identity %868 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %870 = tosa.reshape %869 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %871 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %872 = tosa.transpose %arg74, %871 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %873 = tosa.reshape %870 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_173 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %874 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%873, %872 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_173 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %875 = tosa.reshape %874 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %876 = tosa.add %779, %875 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %877 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_174 = arith.constant 2 : i32
    %878 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%876 : tensor<1x40x4096xf32>) outs(%877 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_174 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_175 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %879 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%878 : tensor<1x40x4096xf32>) outs(%cst_175 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %880 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %881 = tosa.add %879, %880 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %882 = tosa.rsqrt %881 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %883 = tosa.mul %876, %882 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %884 = tosa.reshape %arg75 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %885 = tosa.mul %884, %883 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %886 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %887 = tosa.transpose %arg76, %886 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %888 = tosa.reshape %885 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_176 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %889 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%888, %887 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_176 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %890 = tosa.reshape %889 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %891 = tosa.sigmoid %890 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %892 = tosa.mul %890, %891 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %893 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %894 = tosa.transpose %arg77, %893 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %895 = tosa.reshape %885 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_177 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %896 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%895, %894 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_177 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %897 = tosa.reshape %896 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %898 = tosa.mul %892, %897 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %899 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %900 = tosa.transpose %arg78, %899 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %901 = tosa.reshape %898 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_178 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %902 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%901, %900 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_178 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %903 = tosa.reshape %902 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %904 = tosa.add %876, %903 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %905 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_179 = arith.constant 2 : i32
    %906 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%904 : tensor<1x40x4096xf32>) outs(%905 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_179 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_180 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %907 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%906 : tensor<1x40x4096xf32>) outs(%cst_180 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %908 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %909 = tosa.add %907, %908 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %910 = tosa.rsqrt %909 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %911 = tosa.mul %904, %910 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %912 = tosa.reshape %arg79 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %913 = tosa.mul %912, %911 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %914 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %915 = tosa.transpose %arg80, %914 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %916 = tosa.reshape %913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_181 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %917 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%916, %915 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_181 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %918 = tosa.reshape %917 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %919 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %920 = tosa.transpose %arg81, %919 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %921 = tosa.reshape %913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_182 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %922 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%921, %920 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_182 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %923 = tosa.reshape %922 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %924 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %925 = tosa.transpose %arg82, %924 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %926 = tosa.reshape %913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_183 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %927 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%926, %925 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_183 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %928 = tosa.reshape %927 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %929 = tosa.reshape %918 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %930 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %931 = tosa.transpose %929, %930 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %932 = tosa.reshape %923 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %933 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %934 = tosa.transpose %932, %933 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %935 = tosa.reshape %928 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %936 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %937 = tosa.transpose %935, %936 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_184 = tensor.extract_slice %arg83[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_185 = tensor.extract_slice %extracted_slice_184[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_186 = tensor.extract_slice %extracted_slice_185[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_187 = tensor.extract_slice %arg84[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_188 = tensor.extract_slice %extracted_slice_187[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_189 = tensor.extract_slice %extracted_slice_188[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %938 = tensor.empty() : tensor<1x40x128xf32>
    %939 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_186 : tensor<1x1x40x128xf32>) outs(%938 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %940 = tensor.empty() : tensor<40x128xf32>
    %941 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%939 : tensor<1x40x128xf32>) outs(%940 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %942 = tensor.empty() : tensor<1x40x128xf32>
    %943 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_189 : tensor<1x1x40x128xf32>) outs(%942 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %944 = tensor.empty() : tensor<40x128xf32>
    %945 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%943 : tensor<1x40x128xf32>) outs(%944 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %946 = tensor.empty() : tensor<1x40x128xf32>
    %947 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%946 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %941[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %948 = tosa.reshape %947 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %949 = tensor.empty() : tensor<1x40x128xf32>
    %950 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%949 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %945[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %951 = tosa.reshape %950 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %952 = tosa.mul %931, %948 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_190 = tensor.extract_slice %931[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_191 = tensor.extract_slice %931[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %953 = tensor.empty() : tensor<1x32x40x64xf32>
    %954 = linalg.negf ins(%extracted_slice_191 : tensor<1x32x40x64xf32>) outs(%953 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %955 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_192 = tensor.insert_slice %954 into %955[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_193 = tensor.insert_slice %extracted_slice_190 into %inserted_slice_192[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %956 = tosa.mul %inserted_slice_193, %951 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %957 = tosa.add %952, %956 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %958 = tosa.mul %934, %948 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_194 = tensor.extract_slice %934[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_195 = tensor.extract_slice %934[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %959 = tensor.empty() : tensor<1x32x40x64xf32>
    %960 = linalg.negf ins(%extracted_slice_195 : tensor<1x32x40x64xf32>) outs(%959 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %961 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_196 = tensor.insert_slice %960 into %961[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_197 = tensor.insert_slice %extracted_slice_194 into %inserted_slice_196[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %962 = tosa.mul %inserted_slice_197, %951 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %963 = tosa.add %958, %962 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %964 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %965 = tosa.transpose %963, %964 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %966 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %967 = tosa.add %957, %966 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %968 = tosa.reshape %967 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %969 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %970 = tosa.add %965, %969 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %971 = tosa.reshape %970 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %972 = tosa.matmul %968, %971 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %973 = tosa.reshape %972 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %974 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %975 = tosa.reciprocal %974 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %976 = tosa.mul %973, %975 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %977 = tosa.add %976, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %978 = tosa.reduce_max %977 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %979 = tosa.sub %977, %978 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %980 = tosa.exp %979 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %981 = tosa.reduce_sum %980 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %982 = tosa.reciprocal %981 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %983 = tosa.mul %980, %982 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %984 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %985 = tosa.add %983, %984 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %986 = tosa.reshape %985 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %987 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %988 = tosa.add %937, %987 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %989 = tosa.reshape %988 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %990 = tosa.matmul %986, %989 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %991 = tosa.reshape %990 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %992 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %993 = tosa.transpose %991, %992 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %994 = tosa.identity %993 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %995 = tosa.reshape %994 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %996 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %997 = tosa.transpose %arg85, %996 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %998 = tosa.reshape %995 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_198 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %999 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%998, %997 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_198 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1000 = tosa.reshape %999 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1001 = tosa.add %904, %1000 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1002 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_199 = arith.constant 2 : i32
    %1003 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1001 : tensor<1x40x4096xf32>) outs(%1002 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_199 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_200 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1004 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1003 : tensor<1x40x4096xf32>) outs(%cst_200 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1005 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1006 = tosa.add %1004, %1005 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1007 = tosa.rsqrt %1006 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1008 = tosa.mul %1001, %1007 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1009 = tosa.reshape %arg86 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1010 = tosa.mul %1009, %1008 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1011 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1012 = tosa.transpose %arg87, %1011 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1013 = tosa.reshape %1010 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_201 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1014 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1013, %1012 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_201 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1015 = tosa.reshape %1014 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1016 = tosa.sigmoid %1015 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1017 = tosa.mul %1015, %1016 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1018 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1019 = tosa.transpose %arg88, %1018 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1020 = tosa.reshape %1010 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_202 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1021 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1020, %1019 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_202 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1022 = tosa.reshape %1021 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1023 = tosa.mul %1017, %1022 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1024 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1025 = tosa.transpose %arg89, %1024 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1026 = tosa.reshape %1023 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_203 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1027 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1026, %1025 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_203 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1028 = tosa.reshape %1027 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1029 = tosa.add %1001, %1028 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1030 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_204 = arith.constant 2 : i32
    %1031 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1029 : tensor<1x40x4096xf32>) outs(%1030 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_204 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_205 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1032 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1031 : tensor<1x40x4096xf32>) outs(%cst_205 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1033 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1034 = tosa.add %1032, %1033 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1035 = tosa.rsqrt %1034 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1036 = tosa.mul %1029, %1035 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1037 = tosa.reshape %arg90 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1038 = tosa.mul %1037, %1036 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1039 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1040 = tosa.transpose %arg91, %1039 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1041 = tosa.reshape %1038 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_206 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1042 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1041, %1040 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_206 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1043 = tosa.reshape %1042 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1044 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1045 = tosa.transpose %arg92, %1044 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1046 = tosa.reshape %1038 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_207 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1047 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1046, %1045 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_207 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1048 = tosa.reshape %1047 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1049 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1050 = tosa.transpose %arg93, %1049 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1051 = tosa.reshape %1038 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_208 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1052 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1051, %1050 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_208 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1053 = tosa.reshape %1052 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1054 = tosa.reshape %1043 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1055 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1056 = tosa.transpose %1054, %1055 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1057 = tosa.reshape %1048 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1058 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1059 = tosa.transpose %1057, %1058 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1060 = tosa.reshape %1053 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1061 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1062 = tosa.transpose %1060, %1061 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_209 = tensor.extract_slice %arg94[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_210 = tensor.extract_slice %extracted_slice_209[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_211 = tensor.extract_slice %extracted_slice_210[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_212 = tensor.extract_slice %arg95[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_213 = tensor.extract_slice %extracted_slice_212[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_214 = tensor.extract_slice %extracted_slice_213[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1063 = tensor.empty() : tensor<1x40x128xf32>
    %1064 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_211 : tensor<1x1x40x128xf32>) outs(%1063 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1065 = tensor.empty() : tensor<40x128xf32>
    %1066 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1064 : tensor<1x40x128xf32>) outs(%1065 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1067 = tensor.empty() : tensor<1x40x128xf32>
    %1068 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_214 : tensor<1x1x40x128xf32>) outs(%1067 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1069 = tensor.empty() : tensor<40x128xf32>
    %1070 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1068 : tensor<1x40x128xf32>) outs(%1069 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1071 = tensor.empty() : tensor<1x40x128xf32>
    %1072 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1071 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1066[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1073 = tosa.reshape %1072 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1074 = tensor.empty() : tensor<1x40x128xf32>
    %1075 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1074 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1070[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1076 = tosa.reshape %1075 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1077 = tosa.mul %1056, %1073 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_215 = tensor.extract_slice %1056[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_216 = tensor.extract_slice %1056[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1078 = tensor.empty() : tensor<1x32x40x64xf32>
    %1079 = linalg.negf ins(%extracted_slice_216 : tensor<1x32x40x64xf32>) outs(%1078 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1080 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_217 = tensor.insert_slice %1079 into %1080[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_218 = tensor.insert_slice %extracted_slice_215 into %inserted_slice_217[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1081 = tosa.mul %inserted_slice_218, %1076 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1082 = tosa.add %1077, %1081 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1083 = tosa.mul %1059, %1073 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_219 = tensor.extract_slice %1059[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_220 = tensor.extract_slice %1059[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1084 = tensor.empty() : tensor<1x32x40x64xf32>
    %1085 = linalg.negf ins(%extracted_slice_220 : tensor<1x32x40x64xf32>) outs(%1084 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1086 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_221 = tensor.insert_slice %1085 into %1086[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_222 = tensor.insert_slice %extracted_slice_219 into %inserted_slice_221[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1087 = tosa.mul %inserted_slice_222, %1076 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1088 = tosa.add %1083, %1087 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1089 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1090 = tosa.transpose %1088, %1089 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1091 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1092 = tosa.add %1082, %1091 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1093 = tosa.reshape %1092 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1094 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1095 = tosa.add %1090, %1094 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1096 = tosa.reshape %1095 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1097 = tosa.matmul %1093, %1096 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1098 = tosa.reshape %1097 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1099 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1100 = tosa.reciprocal %1099 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1101 = tosa.mul %1098, %1100 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1102 = tosa.add %1101, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1103 = tosa.reduce_max %1102 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1104 = tosa.sub %1102, %1103 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1105 = tosa.exp %1104 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1106 = tosa.reduce_sum %1105 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1107 = tosa.reciprocal %1106 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1108 = tosa.mul %1105, %1107 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1109 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1110 = tosa.add %1108, %1109 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1111 = tosa.reshape %1110 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1112 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1113 = tosa.add %1062, %1112 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1114 = tosa.reshape %1113 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1115 = tosa.matmul %1111, %1114 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1116 = tosa.reshape %1115 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1117 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1118 = tosa.transpose %1116, %1117 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1119 = tosa.identity %1118 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1120 = tosa.reshape %1119 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1121 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1122 = tosa.transpose %arg96, %1121 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1123 = tosa.reshape %1120 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_223 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1124 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1123, %1122 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_223 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1125 = tosa.reshape %1124 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1126 = tosa.add %1029, %1125 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1127 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_224 = arith.constant 2 : i32
    %1128 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1126 : tensor<1x40x4096xf32>) outs(%1127 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_224 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_225 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1129 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1128 : tensor<1x40x4096xf32>) outs(%cst_225 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1130 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1131 = tosa.add %1129, %1130 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1132 = tosa.rsqrt %1131 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1133 = tosa.mul %1126, %1132 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1134 = tosa.reshape %arg97 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1135 = tosa.mul %1134, %1133 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1136 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1137 = tosa.transpose %arg98, %1136 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1138 = tosa.reshape %1135 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_226 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1139 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1138, %1137 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_226 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1140 = tosa.reshape %1139 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1141 = tosa.sigmoid %1140 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1142 = tosa.mul %1140, %1141 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1143 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1144 = tosa.transpose %arg99, %1143 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1145 = tosa.reshape %1135 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_227 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1146 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1145, %1144 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_227 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1147 = tosa.reshape %1146 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1148 = tosa.mul %1142, %1147 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1149 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1150 = tosa.transpose %arg100, %1149 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1151 = tosa.reshape %1148 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_228 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1152 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1151, %1150 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_228 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1153 = tosa.reshape %1152 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1154 = tosa.add %1126, %1153 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1155 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_229 = arith.constant 2 : i32
    %1156 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1154 : tensor<1x40x4096xf32>) outs(%1155 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_229 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_230 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1157 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1156 : tensor<1x40x4096xf32>) outs(%cst_230 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1158 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1159 = tosa.add %1157, %1158 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1160 = tosa.rsqrt %1159 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1161 = tosa.mul %1154, %1160 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1162 = tosa.reshape %arg101 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1163 = tosa.mul %1162, %1161 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1164 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1165 = tosa.transpose %arg102, %1164 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1166 = tosa.reshape %1163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_231 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1167 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1166, %1165 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_231 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1168 = tosa.reshape %1167 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1169 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1170 = tosa.transpose %arg103, %1169 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1171 = tosa.reshape %1163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_232 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1172 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1171, %1170 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_232 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1173 = tosa.reshape %1172 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1174 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1175 = tosa.transpose %arg104, %1174 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1176 = tosa.reshape %1163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_233 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1177 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1176, %1175 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_233 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1178 = tosa.reshape %1177 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1179 = tosa.reshape %1168 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1180 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1181 = tosa.transpose %1179, %1180 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1182 = tosa.reshape %1173 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1183 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1184 = tosa.transpose %1182, %1183 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1185 = tosa.reshape %1178 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1186 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1187 = tosa.transpose %1185, %1186 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_234 = tensor.extract_slice %arg105[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_235 = tensor.extract_slice %extracted_slice_234[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_236 = tensor.extract_slice %extracted_slice_235[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_237 = tensor.extract_slice %arg106[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_238 = tensor.extract_slice %extracted_slice_237[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_239 = tensor.extract_slice %extracted_slice_238[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1188 = tensor.empty() : tensor<1x40x128xf32>
    %1189 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_236 : tensor<1x1x40x128xf32>) outs(%1188 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1190 = tensor.empty() : tensor<40x128xf32>
    %1191 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1189 : tensor<1x40x128xf32>) outs(%1190 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1192 = tensor.empty() : tensor<1x40x128xf32>
    %1193 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_239 : tensor<1x1x40x128xf32>) outs(%1192 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1194 = tensor.empty() : tensor<40x128xf32>
    %1195 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1193 : tensor<1x40x128xf32>) outs(%1194 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1196 = tensor.empty() : tensor<1x40x128xf32>
    %1197 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1196 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1191[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1198 = tosa.reshape %1197 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1199 = tensor.empty() : tensor<1x40x128xf32>
    %1200 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1199 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1195[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1201 = tosa.reshape %1200 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1202 = tosa.mul %1181, %1198 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_240 = tensor.extract_slice %1181[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_241 = tensor.extract_slice %1181[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1203 = tensor.empty() : tensor<1x32x40x64xf32>
    %1204 = linalg.negf ins(%extracted_slice_241 : tensor<1x32x40x64xf32>) outs(%1203 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1205 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_242 = tensor.insert_slice %1204 into %1205[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_243 = tensor.insert_slice %extracted_slice_240 into %inserted_slice_242[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1206 = tosa.mul %inserted_slice_243, %1201 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1207 = tosa.add %1202, %1206 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1208 = tosa.mul %1184, %1198 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_244 = tensor.extract_slice %1184[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_245 = tensor.extract_slice %1184[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1209 = tensor.empty() : tensor<1x32x40x64xf32>
    %1210 = linalg.negf ins(%extracted_slice_245 : tensor<1x32x40x64xf32>) outs(%1209 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1211 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_246 = tensor.insert_slice %1210 into %1211[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_247 = tensor.insert_slice %extracted_slice_244 into %inserted_slice_246[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1212 = tosa.mul %inserted_slice_247, %1201 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1213 = tosa.add %1208, %1212 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1214 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1215 = tosa.transpose %1213, %1214 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1216 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1217 = tosa.add %1207, %1216 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1218 = tosa.reshape %1217 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1219 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1220 = tosa.add %1215, %1219 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1221 = tosa.reshape %1220 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1222 = tosa.matmul %1218, %1221 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1223 = tosa.reshape %1222 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1224 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1225 = tosa.reciprocal %1224 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1226 = tosa.mul %1223, %1225 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1227 = tosa.add %1226, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1228 = tosa.reduce_max %1227 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1229 = tosa.sub %1227, %1228 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1230 = tosa.exp %1229 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1231 = tosa.reduce_sum %1230 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1232 = tosa.reciprocal %1231 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1233 = tosa.mul %1230, %1232 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1234 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1235 = tosa.add %1233, %1234 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1236 = tosa.reshape %1235 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1237 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1238 = tosa.add %1187, %1237 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1239 = tosa.reshape %1238 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1240 = tosa.matmul %1236, %1239 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1241 = tosa.reshape %1240 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1242 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1243 = tosa.transpose %1241, %1242 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1244 = tosa.identity %1243 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1245 = tosa.reshape %1244 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1246 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1247 = tosa.transpose %arg107, %1246 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1248 = tosa.reshape %1245 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_248 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1249 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1248, %1247 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_248 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1250 = tosa.reshape %1249 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1251 = tosa.add %1154, %1250 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1252 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_249 = arith.constant 2 : i32
    %1253 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1251 : tensor<1x40x4096xf32>) outs(%1252 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_249 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_250 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1254 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1253 : tensor<1x40x4096xf32>) outs(%cst_250 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1255 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1256 = tosa.add %1254, %1255 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1257 = tosa.rsqrt %1256 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1258 = tosa.mul %1251, %1257 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1259 = tosa.reshape %arg108 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1260 = tosa.mul %1259, %1258 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1261 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1262 = tosa.transpose %arg109, %1261 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1263 = tosa.reshape %1260 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_251 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1264 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1263, %1262 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_251 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1265 = tosa.reshape %1264 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1266 = tosa.sigmoid %1265 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1267 = tosa.mul %1265, %1266 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1268 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1269 = tosa.transpose %arg110, %1268 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1270 = tosa.reshape %1260 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_252 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1271 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1270, %1269 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_252 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1272 = tosa.reshape %1271 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1273 = tosa.mul %1267, %1272 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1274 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1275 = tosa.transpose %arg111, %1274 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1276 = tosa.reshape %1273 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_253 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1277 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1276, %1275 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_253 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1278 = tosa.reshape %1277 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1279 = tosa.add %1251, %1278 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1280 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_254 = arith.constant 2 : i32
    %1281 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1279 : tensor<1x40x4096xf32>) outs(%1280 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_254 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_255 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1282 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1281 : tensor<1x40x4096xf32>) outs(%cst_255 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1283 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1284 = tosa.add %1282, %1283 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1285 = tosa.rsqrt %1284 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1286 = tosa.mul %1279, %1285 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1287 = tosa.reshape %arg112 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1288 = tosa.mul %1287, %1286 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1289 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1290 = tosa.transpose %arg113, %1289 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1291 = tosa.reshape %1288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_256 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1292 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1291, %1290 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_256 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1293 = tosa.reshape %1292 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1294 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1295 = tosa.transpose %arg114, %1294 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1296 = tosa.reshape %1288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_257 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1297 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1296, %1295 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_257 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1298 = tosa.reshape %1297 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1299 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1300 = tosa.transpose %arg115, %1299 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1301 = tosa.reshape %1288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_258 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1302 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1301, %1300 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_258 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1303 = tosa.reshape %1302 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1304 = tosa.reshape %1293 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1305 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1306 = tosa.transpose %1304, %1305 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1307 = tosa.reshape %1298 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1308 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1309 = tosa.transpose %1307, %1308 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1310 = tosa.reshape %1303 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1311 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1312 = tosa.transpose %1310, %1311 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_259 = tensor.extract_slice %arg116[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_260 = tensor.extract_slice %extracted_slice_259[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_261 = tensor.extract_slice %extracted_slice_260[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_262 = tensor.extract_slice %arg117[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_263 = tensor.extract_slice %extracted_slice_262[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_264 = tensor.extract_slice %extracted_slice_263[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1313 = tensor.empty() : tensor<1x40x128xf32>
    %1314 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_261 : tensor<1x1x40x128xf32>) outs(%1313 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1315 = tensor.empty() : tensor<40x128xf32>
    %1316 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1314 : tensor<1x40x128xf32>) outs(%1315 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1317 = tensor.empty() : tensor<1x40x128xf32>
    %1318 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_264 : tensor<1x1x40x128xf32>) outs(%1317 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1319 = tensor.empty() : tensor<40x128xf32>
    %1320 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1318 : tensor<1x40x128xf32>) outs(%1319 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1321 = tensor.empty() : tensor<1x40x128xf32>
    %1322 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1321 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1316[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1323 = tosa.reshape %1322 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1324 = tensor.empty() : tensor<1x40x128xf32>
    %1325 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1324 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1320[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1326 = tosa.reshape %1325 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1327 = tosa.mul %1306, %1323 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_265 = tensor.extract_slice %1306[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_266 = tensor.extract_slice %1306[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1328 = tensor.empty() : tensor<1x32x40x64xf32>
    %1329 = linalg.negf ins(%extracted_slice_266 : tensor<1x32x40x64xf32>) outs(%1328 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1330 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_267 = tensor.insert_slice %1329 into %1330[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_268 = tensor.insert_slice %extracted_slice_265 into %inserted_slice_267[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1331 = tosa.mul %inserted_slice_268, %1326 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1332 = tosa.add %1327, %1331 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1333 = tosa.mul %1309, %1323 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_269 = tensor.extract_slice %1309[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_270 = tensor.extract_slice %1309[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1334 = tensor.empty() : tensor<1x32x40x64xf32>
    %1335 = linalg.negf ins(%extracted_slice_270 : tensor<1x32x40x64xf32>) outs(%1334 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1336 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_271 = tensor.insert_slice %1335 into %1336[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_272 = tensor.insert_slice %extracted_slice_269 into %inserted_slice_271[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1337 = tosa.mul %inserted_slice_272, %1326 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1338 = tosa.add %1333, %1337 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1339 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1340 = tosa.transpose %1338, %1339 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1341 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1342 = tosa.add %1332, %1341 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1343 = tosa.reshape %1342 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1344 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1345 = tosa.add %1340, %1344 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1346 = tosa.reshape %1345 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1347 = tosa.matmul %1343, %1346 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1348 = tosa.reshape %1347 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1349 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1350 = tosa.reciprocal %1349 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1351 = tosa.mul %1348, %1350 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1352 = tosa.add %1351, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1353 = tosa.reduce_max %1352 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1354 = tosa.sub %1352, %1353 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1355 = tosa.exp %1354 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1356 = tosa.reduce_sum %1355 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1357 = tosa.reciprocal %1356 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1358 = tosa.mul %1355, %1357 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1359 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1360 = tosa.add %1358, %1359 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1361 = tosa.reshape %1360 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1362 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1363 = tosa.add %1312, %1362 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1364 = tosa.reshape %1363 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1365 = tosa.matmul %1361, %1364 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1366 = tosa.reshape %1365 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1367 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1368 = tosa.transpose %1366, %1367 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1369 = tosa.identity %1368 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1370 = tosa.reshape %1369 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1371 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1372 = tosa.transpose %arg118, %1371 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1373 = tosa.reshape %1370 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_273 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1374 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1373, %1372 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_273 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1375 = tosa.reshape %1374 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1376 = tosa.add %1279, %1375 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1377 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_274 = arith.constant 2 : i32
    %1378 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1376 : tensor<1x40x4096xf32>) outs(%1377 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_274 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_275 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1379 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1378 : tensor<1x40x4096xf32>) outs(%cst_275 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1380 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1381 = tosa.add %1379, %1380 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1382 = tosa.rsqrt %1381 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1383 = tosa.mul %1376, %1382 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1384 = tosa.reshape %arg119 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1385 = tosa.mul %1384, %1383 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1386 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1387 = tosa.transpose %arg120, %1386 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1388 = tosa.reshape %1385 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_276 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1389 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1388, %1387 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_276 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1390 = tosa.reshape %1389 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1391 = tosa.sigmoid %1390 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1392 = tosa.mul %1390, %1391 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1393 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1394 = tosa.transpose %arg121, %1393 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1395 = tosa.reshape %1385 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_277 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1396 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1395, %1394 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_277 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1397 = tosa.reshape %1396 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1398 = tosa.mul %1392, %1397 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1399 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1400 = tosa.transpose %arg122, %1399 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1401 = tosa.reshape %1398 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_278 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1402 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1401, %1400 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_278 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1403 = tosa.reshape %1402 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1404 = tosa.add %1376, %1403 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1405 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_279 = arith.constant 2 : i32
    %1406 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1404 : tensor<1x40x4096xf32>) outs(%1405 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_279 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_280 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1407 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1406 : tensor<1x40x4096xf32>) outs(%cst_280 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1408 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1409 = tosa.add %1407, %1408 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1410 = tosa.rsqrt %1409 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1411 = tosa.mul %1404, %1410 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1412 = tosa.reshape %arg123 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1413 = tosa.mul %1412, %1411 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1414 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1415 = tosa.transpose %arg124, %1414 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1416 = tosa.reshape %1413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_281 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1417 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1416, %1415 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_281 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1418 = tosa.reshape %1417 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1419 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1420 = tosa.transpose %arg125, %1419 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1421 = tosa.reshape %1413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_282 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1422 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1421, %1420 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_282 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1423 = tosa.reshape %1422 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1424 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1425 = tosa.transpose %arg126, %1424 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1426 = tosa.reshape %1413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_283 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1427 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1426, %1425 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_283 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1428 = tosa.reshape %1427 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1429 = tosa.reshape %1418 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1430 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1431 = tosa.transpose %1429, %1430 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1432 = tosa.reshape %1423 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1433 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1434 = tosa.transpose %1432, %1433 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1435 = tosa.reshape %1428 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1436 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1437 = tosa.transpose %1435, %1436 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_284 = tensor.extract_slice %arg127[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_285 = tensor.extract_slice %extracted_slice_284[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_286 = tensor.extract_slice %extracted_slice_285[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_287 = tensor.extract_slice %arg128[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_288 = tensor.extract_slice %extracted_slice_287[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_289 = tensor.extract_slice %extracted_slice_288[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1438 = tensor.empty() : tensor<1x40x128xf32>
    %1439 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_286 : tensor<1x1x40x128xf32>) outs(%1438 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1440 = tensor.empty() : tensor<40x128xf32>
    %1441 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1439 : tensor<1x40x128xf32>) outs(%1440 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1442 = tensor.empty() : tensor<1x40x128xf32>
    %1443 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_289 : tensor<1x1x40x128xf32>) outs(%1442 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1444 = tensor.empty() : tensor<40x128xf32>
    %1445 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1443 : tensor<1x40x128xf32>) outs(%1444 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1446 = tensor.empty() : tensor<1x40x128xf32>
    %1447 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1446 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1441[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1448 = tosa.reshape %1447 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1449 = tensor.empty() : tensor<1x40x128xf32>
    %1450 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1449 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1445[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1451 = tosa.reshape %1450 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1452 = tosa.mul %1431, %1448 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_290 = tensor.extract_slice %1431[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_291 = tensor.extract_slice %1431[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1453 = tensor.empty() : tensor<1x32x40x64xf32>
    %1454 = linalg.negf ins(%extracted_slice_291 : tensor<1x32x40x64xf32>) outs(%1453 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1455 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_292 = tensor.insert_slice %1454 into %1455[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_293 = tensor.insert_slice %extracted_slice_290 into %inserted_slice_292[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1456 = tosa.mul %inserted_slice_293, %1451 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1457 = tosa.add %1452, %1456 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1458 = tosa.mul %1434, %1448 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_294 = tensor.extract_slice %1434[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_295 = tensor.extract_slice %1434[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1459 = tensor.empty() : tensor<1x32x40x64xf32>
    %1460 = linalg.negf ins(%extracted_slice_295 : tensor<1x32x40x64xf32>) outs(%1459 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1461 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_296 = tensor.insert_slice %1460 into %1461[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_297 = tensor.insert_slice %extracted_slice_294 into %inserted_slice_296[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1462 = tosa.mul %inserted_slice_297, %1451 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1463 = tosa.add %1458, %1462 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1464 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1465 = tosa.transpose %1463, %1464 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1466 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1467 = tosa.add %1457, %1466 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1468 = tosa.reshape %1467 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1469 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1470 = tosa.add %1465, %1469 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1471 = tosa.reshape %1470 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1472 = tosa.matmul %1468, %1471 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1473 = tosa.reshape %1472 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1474 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1475 = tosa.reciprocal %1474 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1476 = tosa.mul %1473, %1475 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1477 = tosa.add %1476, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1478 = tosa.reduce_max %1477 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1479 = tosa.sub %1477, %1478 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1480 = tosa.exp %1479 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1481 = tosa.reduce_sum %1480 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1482 = tosa.reciprocal %1481 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1483 = tosa.mul %1480, %1482 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1484 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1485 = tosa.add %1483, %1484 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1486 = tosa.reshape %1485 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1487 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1488 = tosa.add %1437, %1487 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1489 = tosa.reshape %1488 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1490 = tosa.matmul %1486, %1489 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1491 = tosa.reshape %1490 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1492 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1493 = tosa.transpose %1491, %1492 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1494 = tosa.identity %1493 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1495 = tosa.reshape %1494 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1496 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1497 = tosa.transpose %arg129, %1496 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1498 = tosa.reshape %1495 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_298 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1499 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1498, %1497 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_298 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1500 = tosa.reshape %1499 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1501 = tosa.add %1404, %1500 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1502 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_299 = arith.constant 2 : i32
    %1503 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1501 : tensor<1x40x4096xf32>) outs(%1502 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_299 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_300 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1504 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1503 : tensor<1x40x4096xf32>) outs(%cst_300 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1505 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1506 = tosa.add %1504, %1505 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1507 = tosa.rsqrt %1506 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1508 = tosa.mul %1501, %1507 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1509 = tosa.reshape %arg130 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1510 = tosa.mul %1509, %1508 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1511 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1512 = tosa.transpose %arg131, %1511 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1513 = tosa.reshape %1510 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_301 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1514 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1513, %1512 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_301 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1515 = tosa.reshape %1514 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1516 = tosa.sigmoid %1515 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1517 = tosa.mul %1515, %1516 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1518 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1519 = tosa.transpose %arg132, %1518 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1520 = tosa.reshape %1510 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_302 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1521 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1520, %1519 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_302 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1522 = tosa.reshape %1521 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1523 = tosa.mul %1517, %1522 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1524 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1525 = tosa.transpose %arg133, %1524 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1526 = tosa.reshape %1523 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_303 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1527 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1526, %1525 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_303 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1528 = tosa.reshape %1527 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1529 = tosa.add %1501, %1528 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1530 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_304 = arith.constant 2 : i32
    %1531 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1529 : tensor<1x40x4096xf32>) outs(%1530 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_304 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_305 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1532 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1531 : tensor<1x40x4096xf32>) outs(%cst_305 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1533 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1534 = tosa.add %1532, %1533 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1535 = tosa.rsqrt %1534 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1536 = tosa.mul %1529, %1535 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1537 = tosa.reshape %arg134 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1538 = tosa.mul %1537, %1536 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1539 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1540 = tosa.transpose %arg135, %1539 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1541 = tosa.reshape %1538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_306 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1542 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1541, %1540 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_306 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1543 = tosa.reshape %1542 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1544 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1545 = tosa.transpose %arg136, %1544 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1546 = tosa.reshape %1538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_307 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1547 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1546, %1545 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_307 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1548 = tosa.reshape %1547 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1549 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1550 = tosa.transpose %arg137, %1549 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1551 = tosa.reshape %1538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_308 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1552 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1551, %1550 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_308 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1553 = tosa.reshape %1552 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1554 = tosa.reshape %1543 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1555 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1556 = tosa.transpose %1554, %1555 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1557 = tosa.reshape %1548 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1558 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1559 = tosa.transpose %1557, %1558 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1560 = tosa.reshape %1553 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1561 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1562 = tosa.transpose %1560, %1561 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_309 = tensor.extract_slice %arg138[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_310 = tensor.extract_slice %extracted_slice_309[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_311 = tensor.extract_slice %extracted_slice_310[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_312 = tensor.extract_slice %arg139[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_313 = tensor.extract_slice %extracted_slice_312[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_314 = tensor.extract_slice %extracted_slice_313[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1563 = tensor.empty() : tensor<1x40x128xf32>
    %1564 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_311 : tensor<1x1x40x128xf32>) outs(%1563 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1565 = tensor.empty() : tensor<40x128xf32>
    %1566 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1564 : tensor<1x40x128xf32>) outs(%1565 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1567 = tensor.empty() : tensor<1x40x128xf32>
    %1568 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_314 : tensor<1x1x40x128xf32>) outs(%1567 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1569 = tensor.empty() : tensor<40x128xf32>
    %1570 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1568 : tensor<1x40x128xf32>) outs(%1569 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1571 = tensor.empty() : tensor<1x40x128xf32>
    %1572 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1571 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1566[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1573 = tosa.reshape %1572 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1574 = tensor.empty() : tensor<1x40x128xf32>
    %1575 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1574 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1570[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1576 = tosa.reshape %1575 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1577 = tosa.mul %1556, %1573 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_315 = tensor.extract_slice %1556[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_316 = tensor.extract_slice %1556[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1578 = tensor.empty() : tensor<1x32x40x64xf32>
    %1579 = linalg.negf ins(%extracted_slice_316 : tensor<1x32x40x64xf32>) outs(%1578 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1580 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_317 = tensor.insert_slice %1579 into %1580[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_318 = tensor.insert_slice %extracted_slice_315 into %inserted_slice_317[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1581 = tosa.mul %inserted_slice_318, %1576 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1582 = tosa.add %1577, %1581 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1583 = tosa.mul %1559, %1573 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_319 = tensor.extract_slice %1559[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_320 = tensor.extract_slice %1559[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1584 = tensor.empty() : tensor<1x32x40x64xf32>
    %1585 = linalg.negf ins(%extracted_slice_320 : tensor<1x32x40x64xf32>) outs(%1584 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1586 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_321 = tensor.insert_slice %1585 into %1586[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_322 = tensor.insert_slice %extracted_slice_319 into %inserted_slice_321[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1587 = tosa.mul %inserted_slice_322, %1576 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1588 = tosa.add %1583, %1587 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1589 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1590 = tosa.transpose %1588, %1589 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1591 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1592 = tosa.add %1582, %1591 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1593 = tosa.reshape %1592 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1594 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1595 = tosa.add %1590, %1594 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1596 = tosa.reshape %1595 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1597 = tosa.matmul %1593, %1596 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1598 = tosa.reshape %1597 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1599 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1600 = tosa.reciprocal %1599 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1601 = tosa.mul %1598, %1600 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1602 = tosa.add %1601, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1603 = tosa.reduce_max %1602 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1604 = tosa.sub %1602, %1603 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1605 = tosa.exp %1604 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1606 = tosa.reduce_sum %1605 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1607 = tosa.reciprocal %1606 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1608 = tosa.mul %1605, %1607 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1609 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1610 = tosa.add %1608, %1609 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1611 = tosa.reshape %1610 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1612 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1613 = tosa.add %1562, %1612 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1614 = tosa.reshape %1613 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1615 = tosa.matmul %1611, %1614 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1616 = tosa.reshape %1615 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1617 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1618 = tosa.transpose %1616, %1617 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1619 = tosa.identity %1618 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1620 = tosa.reshape %1619 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1621 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1622 = tosa.transpose %arg140, %1621 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1623 = tosa.reshape %1620 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_323 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1624 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1623, %1622 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_323 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1625 = tosa.reshape %1624 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1626 = tosa.add %1529, %1625 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1627 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_324 = arith.constant 2 : i32
    %1628 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1626 : tensor<1x40x4096xf32>) outs(%1627 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_324 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_325 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1629 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1628 : tensor<1x40x4096xf32>) outs(%cst_325 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1630 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1631 = tosa.add %1629, %1630 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1632 = tosa.rsqrt %1631 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1633 = tosa.mul %1626, %1632 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1634 = tosa.reshape %arg141 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1635 = tosa.mul %1634, %1633 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1636 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1637 = tosa.transpose %arg142, %1636 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1638 = tosa.reshape %1635 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_326 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1639 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1638, %1637 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_326 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1640 = tosa.reshape %1639 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1641 = tosa.sigmoid %1640 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1642 = tosa.mul %1640, %1641 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1643 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1644 = tosa.transpose %arg143, %1643 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1645 = tosa.reshape %1635 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_327 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1646 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1645, %1644 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_327 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1647 = tosa.reshape %1646 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1648 = tosa.mul %1642, %1647 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1649 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1650 = tosa.transpose %arg144, %1649 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1651 = tosa.reshape %1648 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_328 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1652 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1651, %1650 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_328 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1653 = tosa.reshape %1652 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1654 = tosa.add %1626, %1653 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1655 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_329 = arith.constant 2 : i32
    %1656 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1654 : tensor<1x40x4096xf32>) outs(%1655 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_329 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_330 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1657 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1656 : tensor<1x40x4096xf32>) outs(%cst_330 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1658 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1659 = tosa.add %1657, %1658 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1660 = tosa.rsqrt %1659 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1661 = tosa.mul %1654, %1660 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1662 = tosa.reshape %arg145 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1663 = tosa.mul %1662, %1661 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1664 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1665 = tosa.transpose %arg146, %1664 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1666 = tosa.reshape %1663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_331 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1667 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1666, %1665 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_331 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1668 = tosa.reshape %1667 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1669 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1670 = tosa.transpose %arg147, %1669 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1671 = tosa.reshape %1663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_332 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1672 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1671, %1670 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_332 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1673 = tosa.reshape %1672 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1674 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1675 = tosa.transpose %arg148, %1674 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1676 = tosa.reshape %1663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_333 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1677 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1676, %1675 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_333 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1678 = tosa.reshape %1677 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1679 = tosa.reshape %1668 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1680 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1681 = tosa.transpose %1679, %1680 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1682 = tosa.reshape %1673 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1683 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1684 = tosa.transpose %1682, %1683 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1685 = tosa.reshape %1678 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1686 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1687 = tosa.transpose %1685, %1686 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_334 = tensor.extract_slice %arg149[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_335 = tensor.extract_slice %extracted_slice_334[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_336 = tensor.extract_slice %extracted_slice_335[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_337 = tensor.extract_slice %arg150[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_338 = tensor.extract_slice %extracted_slice_337[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_339 = tensor.extract_slice %extracted_slice_338[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1688 = tensor.empty() : tensor<1x40x128xf32>
    %1689 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_336 : tensor<1x1x40x128xf32>) outs(%1688 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1690 = tensor.empty() : tensor<40x128xf32>
    %1691 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1689 : tensor<1x40x128xf32>) outs(%1690 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1692 = tensor.empty() : tensor<1x40x128xf32>
    %1693 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_339 : tensor<1x1x40x128xf32>) outs(%1692 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1694 = tensor.empty() : tensor<40x128xf32>
    %1695 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1693 : tensor<1x40x128xf32>) outs(%1694 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1696 = tensor.empty() : tensor<1x40x128xf32>
    %1697 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1696 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1691[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1698 = tosa.reshape %1697 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1699 = tensor.empty() : tensor<1x40x128xf32>
    %1700 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1699 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1695[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1701 = tosa.reshape %1700 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1702 = tosa.mul %1681, %1698 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_340 = tensor.extract_slice %1681[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_341 = tensor.extract_slice %1681[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1703 = tensor.empty() : tensor<1x32x40x64xf32>
    %1704 = linalg.negf ins(%extracted_slice_341 : tensor<1x32x40x64xf32>) outs(%1703 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1705 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_342 = tensor.insert_slice %1704 into %1705[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_343 = tensor.insert_slice %extracted_slice_340 into %inserted_slice_342[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1706 = tosa.mul %inserted_slice_343, %1701 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1707 = tosa.add %1702, %1706 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1708 = tosa.mul %1684, %1698 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_344 = tensor.extract_slice %1684[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_345 = tensor.extract_slice %1684[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1709 = tensor.empty() : tensor<1x32x40x64xf32>
    %1710 = linalg.negf ins(%extracted_slice_345 : tensor<1x32x40x64xf32>) outs(%1709 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1711 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_346 = tensor.insert_slice %1710 into %1711[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_347 = tensor.insert_slice %extracted_slice_344 into %inserted_slice_346[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1712 = tosa.mul %inserted_slice_347, %1701 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1713 = tosa.add %1708, %1712 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1714 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1715 = tosa.transpose %1713, %1714 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1716 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1717 = tosa.add %1707, %1716 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1718 = tosa.reshape %1717 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1719 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1720 = tosa.add %1715, %1719 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1721 = tosa.reshape %1720 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1722 = tosa.matmul %1718, %1721 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1723 = tosa.reshape %1722 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1724 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1725 = tosa.reciprocal %1724 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1726 = tosa.mul %1723, %1725 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1727 = tosa.add %1726, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1728 = tosa.reduce_max %1727 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1729 = tosa.sub %1727, %1728 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1730 = tosa.exp %1729 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1731 = tosa.reduce_sum %1730 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1732 = tosa.reciprocal %1731 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1733 = tosa.mul %1730, %1732 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1734 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1735 = tosa.add %1733, %1734 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1736 = tosa.reshape %1735 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1737 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1738 = tosa.add %1687, %1737 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1739 = tosa.reshape %1738 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1740 = tosa.matmul %1736, %1739 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1741 = tosa.reshape %1740 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1742 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1743 = tosa.transpose %1741, %1742 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1744 = tosa.identity %1743 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1745 = tosa.reshape %1744 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1746 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1747 = tosa.transpose %arg151, %1746 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1748 = tosa.reshape %1745 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_348 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1749 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1748, %1747 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_348 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1750 = tosa.reshape %1749 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1751 = tosa.add %1654, %1750 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1752 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_349 = arith.constant 2 : i32
    %1753 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1751 : tensor<1x40x4096xf32>) outs(%1752 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_349 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_350 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1754 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1753 : tensor<1x40x4096xf32>) outs(%cst_350 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1755 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1756 = tosa.add %1754, %1755 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1757 = tosa.rsqrt %1756 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1758 = tosa.mul %1751, %1757 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1759 = tosa.reshape %arg152 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1760 = tosa.mul %1759, %1758 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1761 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1762 = tosa.transpose %arg153, %1761 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1763 = tosa.reshape %1760 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_351 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1764 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1763, %1762 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_351 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1765 = tosa.reshape %1764 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1766 = tosa.sigmoid %1765 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1767 = tosa.mul %1765, %1766 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1768 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1769 = tosa.transpose %arg154, %1768 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1770 = tosa.reshape %1760 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_352 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1771 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1770, %1769 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_352 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1772 = tosa.reshape %1771 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1773 = tosa.mul %1767, %1772 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1774 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1775 = tosa.transpose %arg155, %1774 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1776 = tosa.reshape %1773 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_353 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1777 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1776, %1775 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_353 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1778 = tosa.reshape %1777 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1779 = tosa.add %1751, %1778 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1780 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_354 = arith.constant 2 : i32
    %1781 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1779 : tensor<1x40x4096xf32>) outs(%1780 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_354 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_355 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1782 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1781 : tensor<1x40x4096xf32>) outs(%cst_355 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1783 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1784 = tosa.add %1782, %1783 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1785 = tosa.rsqrt %1784 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1786 = tosa.mul %1779, %1785 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1787 = tosa.reshape %arg156 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1788 = tosa.mul %1787, %1786 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1789 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1790 = tosa.transpose %arg157, %1789 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1791 = tosa.reshape %1788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_356 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1792 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1791, %1790 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_356 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1793 = tosa.reshape %1792 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1794 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1795 = tosa.transpose %arg158, %1794 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1796 = tosa.reshape %1788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_357 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1797 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1796, %1795 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_357 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1798 = tosa.reshape %1797 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1799 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1800 = tosa.transpose %arg159, %1799 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1801 = tosa.reshape %1788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_358 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1802 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1801, %1800 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_358 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1803 = tosa.reshape %1802 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1804 = tosa.reshape %1793 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1805 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1806 = tosa.transpose %1804, %1805 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1807 = tosa.reshape %1798 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1808 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1809 = tosa.transpose %1807, %1808 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1810 = tosa.reshape %1803 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1811 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1812 = tosa.transpose %1810, %1811 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_359 = tensor.extract_slice %arg160[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_360 = tensor.extract_slice %extracted_slice_359[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_361 = tensor.extract_slice %extracted_slice_360[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_362 = tensor.extract_slice %arg161[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_363 = tensor.extract_slice %extracted_slice_362[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_364 = tensor.extract_slice %extracted_slice_363[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1813 = tensor.empty() : tensor<1x40x128xf32>
    %1814 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_361 : tensor<1x1x40x128xf32>) outs(%1813 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1815 = tensor.empty() : tensor<40x128xf32>
    %1816 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1814 : tensor<1x40x128xf32>) outs(%1815 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1817 = tensor.empty() : tensor<1x40x128xf32>
    %1818 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_364 : tensor<1x1x40x128xf32>) outs(%1817 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1819 = tensor.empty() : tensor<40x128xf32>
    %1820 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1818 : tensor<1x40x128xf32>) outs(%1819 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1821 = tensor.empty() : tensor<1x40x128xf32>
    %1822 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1821 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1816[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1823 = tosa.reshape %1822 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1824 = tensor.empty() : tensor<1x40x128xf32>
    %1825 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1824 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1820[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1826 = tosa.reshape %1825 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1827 = tosa.mul %1806, %1823 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_365 = tensor.extract_slice %1806[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_366 = tensor.extract_slice %1806[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1828 = tensor.empty() : tensor<1x32x40x64xf32>
    %1829 = linalg.negf ins(%extracted_slice_366 : tensor<1x32x40x64xf32>) outs(%1828 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1830 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_367 = tensor.insert_slice %1829 into %1830[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_368 = tensor.insert_slice %extracted_slice_365 into %inserted_slice_367[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1831 = tosa.mul %inserted_slice_368, %1826 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1832 = tosa.add %1827, %1831 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1833 = tosa.mul %1809, %1823 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_369 = tensor.extract_slice %1809[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_370 = tensor.extract_slice %1809[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1834 = tensor.empty() : tensor<1x32x40x64xf32>
    %1835 = linalg.negf ins(%extracted_slice_370 : tensor<1x32x40x64xf32>) outs(%1834 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1836 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_371 = tensor.insert_slice %1835 into %1836[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_372 = tensor.insert_slice %extracted_slice_369 into %inserted_slice_371[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1837 = tosa.mul %inserted_slice_372, %1826 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1838 = tosa.add %1833, %1837 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1839 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1840 = tosa.transpose %1838, %1839 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1841 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1842 = tosa.add %1832, %1841 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1843 = tosa.reshape %1842 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1844 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1845 = tosa.add %1840, %1844 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1846 = tosa.reshape %1845 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1847 = tosa.matmul %1843, %1846 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1848 = tosa.reshape %1847 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1849 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1850 = tosa.reciprocal %1849 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1851 = tosa.mul %1848, %1850 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1852 = tosa.add %1851, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1853 = tosa.reduce_max %1852 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1854 = tosa.sub %1852, %1853 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1855 = tosa.exp %1854 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1856 = tosa.reduce_sum %1855 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1857 = tosa.reciprocal %1856 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1858 = tosa.mul %1855, %1857 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1859 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1860 = tosa.add %1858, %1859 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1861 = tosa.reshape %1860 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1862 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1863 = tosa.add %1812, %1862 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1864 = tosa.reshape %1863 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1865 = tosa.matmul %1861, %1864 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1866 = tosa.reshape %1865 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1867 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1868 = tosa.transpose %1866, %1867 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1869 = tosa.identity %1868 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1870 = tosa.reshape %1869 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1871 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1872 = tosa.transpose %arg162, %1871 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1873 = tosa.reshape %1870 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_373 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1874 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1873, %1872 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_373 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1875 = tosa.reshape %1874 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1876 = tosa.add %1779, %1875 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1877 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_374 = arith.constant 2 : i32
    %1878 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1876 : tensor<1x40x4096xf32>) outs(%1877 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_374 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_375 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1879 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1878 : tensor<1x40x4096xf32>) outs(%cst_375 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1880 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1881 = tosa.add %1879, %1880 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1882 = tosa.rsqrt %1881 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1883 = tosa.mul %1876, %1882 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1884 = tosa.reshape %arg163 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1885 = tosa.mul %1884, %1883 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1886 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1887 = tosa.transpose %arg164, %1886 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1888 = tosa.reshape %1885 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_376 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1889 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1888, %1887 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_376 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1890 = tosa.reshape %1889 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1891 = tosa.sigmoid %1890 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1892 = tosa.mul %1890, %1891 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1893 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1894 = tosa.transpose %arg165, %1893 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %1895 = tosa.reshape %1885 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_377 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1896 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1895, %1894 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_377 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %1897 = tosa.reshape %1896 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %1898 = tosa.mul %1892, %1897 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %1899 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1900 = tosa.transpose %arg166, %1899 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %1901 = tosa.reshape %1898 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_378 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1902 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1901, %1900 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_378 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1903 = tosa.reshape %1902 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1904 = tosa.add %1876, %1903 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1905 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_379 = arith.constant 2 : i32
    %1906 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1904 : tensor<1x40x4096xf32>) outs(%1905 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_379 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_380 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1907 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1906 : tensor<1x40x4096xf32>) outs(%cst_380 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %1908 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1909 = tosa.add %1907, %1908 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1910 = tosa.rsqrt %1909 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1911 = tosa.mul %1904, %1910 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1912 = tosa.reshape %arg167 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1913 = tosa.mul %1912, %1911 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %1914 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1915 = tosa.transpose %arg168, %1914 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1916 = tosa.reshape %1913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_381 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1917 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1916, %1915 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_381 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1918 = tosa.reshape %1917 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1919 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1920 = tosa.transpose %arg169, %1919 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1921 = tosa.reshape %1913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_382 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1922 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1921, %1920 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_382 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1923 = tosa.reshape %1922 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1924 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1925 = tosa.transpose %arg170, %1924 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1926 = tosa.reshape %1913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_383 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1927 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1926, %1925 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_383 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %1928 = tosa.reshape %1927 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %1929 = tosa.reshape %1918 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1930 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1931 = tosa.transpose %1929, %1930 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1932 = tosa.reshape %1923 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1933 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1934 = tosa.transpose %1932, %1933 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %1935 = tosa.reshape %1928 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %1936 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1937 = tosa.transpose %1935, %1936 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_384 = tensor.extract_slice %arg171[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_385 = tensor.extract_slice %extracted_slice_384[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_386 = tensor.extract_slice %extracted_slice_385[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_387 = tensor.extract_slice %arg172[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_388 = tensor.extract_slice %extracted_slice_387[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_389 = tensor.extract_slice %extracted_slice_388[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1938 = tensor.empty() : tensor<1x40x128xf32>
    %1939 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_386 : tensor<1x1x40x128xf32>) outs(%1938 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1940 = tensor.empty() : tensor<40x128xf32>
    %1941 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1939 : tensor<1x40x128xf32>) outs(%1940 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1942 = tensor.empty() : tensor<1x40x128xf32>
    %1943 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_389 : tensor<1x1x40x128xf32>) outs(%1942 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1944 = tensor.empty() : tensor<40x128xf32>
    %1945 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%1943 : tensor<1x40x128xf32>) outs(%1944 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1946 = tensor.empty() : tensor<1x40x128xf32>
    %1947 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1946 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1941[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1948 = tosa.reshape %1947 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1949 = tensor.empty() : tensor<1x40x128xf32>
    %1950 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1949 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %1945[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %1951 = tosa.reshape %1950 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1952 = tosa.mul %1931, %1948 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_390 = tensor.extract_slice %1931[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_391 = tensor.extract_slice %1931[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1953 = tensor.empty() : tensor<1x32x40x64xf32>
    %1954 = linalg.negf ins(%extracted_slice_391 : tensor<1x32x40x64xf32>) outs(%1953 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1955 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_392 = tensor.insert_slice %1954 into %1955[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_393 = tensor.insert_slice %extracted_slice_390 into %inserted_slice_392[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1956 = tosa.mul %inserted_slice_393, %1951 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1957 = tosa.add %1952, %1956 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1958 = tosa.mul %1934, %1948 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_394 = tensor.extract_slice %1934[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_395 = tensor.extract_slice %1934[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1959 = tensor.empty() : tensor<1x32x40x64xf32>
    %1960 = linalg.negf ins(%extracted_slice_395 : tensor<1x32x40x64xf32>) outs(%1959 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1961 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_396 = tensor.insert_slice %1960 into %1961[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_397 = tensor.insert_slice %extracted_slice_394 into %inserted_slice_396[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %1962 = tosa.mul %inserted_slice_397, %1951 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1963 = tosa.add %1958, %1962 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1964 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1965 = tosa.transpose %1963, %1964 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %1966 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1967 = tosa.add %1957, %1966 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1968 = tosa.reshape %1967 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1969 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %1970 = tosa.add %1965, %1969 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %1971 = tosa.reshape %1970 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %1972 = tosa.matmul %1968, %1971 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %1973 = tosa.reshape %1972 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1974 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1975 = tosa.reciprocal %1974 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1976 = tosa.mul %1973, %1975 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1977 = tosa.add %1976, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1978 = tosa.reduce_max %1977 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1979 = tosa.sub %1977, %1978 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1980 = tosa.exp %1979 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1981 = tosa.reduce_sum %1980 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %1982 = tosa.reciprocal %1981 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %1983 = tosa.mul %1980, %1982 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %1984 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %1985 = tosa.add %1983, %1984 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %1986 = tosa.reshape %1985 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %1987 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %1988 = tosa.add %1937, %1987 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1989 = tosa.reshape %1988 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %1990 = tosa.matmul %1986, %1989 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %1991 = tosa.reshape %1990 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %1992 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1993 = tosa.transpose %1991, %1992 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %1994 = tosa.identity %1993 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %1995 = tosa.reshape %1994 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %1996 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1997 = tosa.transpose %arg173, %1996 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %1998 = tosa.reshape %1995 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_398 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1999 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1998, %1997 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_398 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2000 = tosa.reshape %1999 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2001 = tosa.add %1904, %2000 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2002 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_399 = arith.constant 2 : i32
    %2003 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2001 : tensor<1x40x4096xf32>) outs(%2002 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_399 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_400 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2004 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2003 : tensor<1x40x4096xf32>) outs(%cst_400 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2005 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2006 = tosa.add %2004, %2005 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2007 = tosa.rsqrt %2006 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2008 = tosa.mul %2001, %2007 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2009 = tosa.reshape %arg174 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2010 = tosa.mul %2009, %2008 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2011 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2012 = tosa.transpose %arg175, %2011 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2013 = tosa.reshape %2010 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_401 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2014 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2013, %2012 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_401 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2015 = tosa.reshape %2014 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2016 = tosa.sigmoid %2015 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2017 = tosa.mul %2015, %2016 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2018 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2019 = tosa.transpose %arg176, %2018 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2020 = tosa.reshape %2010 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_402 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2021 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2020, %2019 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_402 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2022 = tosa.reshape %2021 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2023 = tosa.mul %2017, %2022 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2024 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2025 = tosa.transpose %arg177, %2024 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2026 = tosa.reshape %2023 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_403 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2027 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2026, %2025 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_403 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2028 = tosa.reshape %2027 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2029 = tosa.add %2001, %2028 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2030 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_404 = arith.constant 2 : i32
    %2031 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2029 : tensor<1x40x4096xf32>) outs(%2030 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_404 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_405 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2032 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2031 : tensor<1x40x4096xf32>) outs(%cst_405 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2033 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2034 = tosa.add %2032, %2033 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2035 = tosa.rsqrt %2034 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2036 = tosa.mul %2029, %2035 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2037 = tosa.reshape %arg178 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2038 = tosa.mul %2037, %2036 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2039 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2040 = tosa.transpose %arg179, %2039 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2041 = tosa.reshape %2038 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_406 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2042 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2041, %2040 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_406 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2043 = tosa.reshape %2042 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2044 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2045 = tosa.transpose %arg180, %2044 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2046 = tosa.reshape %2038 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_407 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2047 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2046, %2045 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_407 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2048 = tosa.reshape %2047 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2049 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2050 = tosa.transpose %arg181, %2049 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2051 = tosa.reshape %2038 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_408 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2052 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2051, %2050 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_408 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2053 = tosa.reshape %2052 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2054 = tosa.reshape %2043 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2055 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2056 = tosa.transpose %2054, %2055 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2057 = tosa.reshape %2048 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2058 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2059 = tosa.transpose %2057, %2058 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2060 = tosa.reshape %2053 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2061 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2062 = tosa.transpose %2060, %2061 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_409 = tensor.extract_slice %arg182[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_410 = tensor.extract_slice %extracted_slice_409[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_411 = tensor.extract_slice %extracted_slice_410[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_412 = tensor.extract_slice %arg183[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_413 = tensor.extract_slice %extracted_slice_412[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_414 = tensor.extract_slice %extracted_slice_413[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2063 = tensor.empty() : tensor<1x40x128xf32>
    %2064 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_411 : tensor<1x1x40x128xf32>) outs(%2063 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2065 = tensor.empty() : tensor<40x128xf32>
    %2066 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2064 : tensor<1x40x128xf32>) outs(%2065 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2067 = tensor.empty() : tensor<1x40x128xf32>
    %2068 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_414 : tensor<1x1x40x128xf32>) outs(%2067 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2069 = tensor.empty() : tensor<40x128xf32>
    %2070 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2068 : tensor<1x40x128xf32>) outs(%2069 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2071 = tensor.empty() : tensor<1x40x128xf32>
    %2072 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2071 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2066[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2073 = tosa.reshape %2072 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2074 = tensor.empty() : tensor<1x40x128xf32>
    %2075 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2074 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2070[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2076 = tosa.reshape %2075 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2077 = tosa.mul %2056, %2073 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_415 = tensor.extract_slice %2056[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_416 = tensor.extract_slice %2056[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2078 = tensor.empty() : tensor<1x32x40x64xf32>
    %2079 = linalg.negf ins(%extracted_slice_416 : tensor<1x32x40x64xf32>) outs(%2078 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2080 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_417 = tensor.insert_slice %2079 into %2080[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_418 = tensor.insert_slice %extracted_slice_415 into %inserted_slice_417[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2081 = tosa.mul %inserted_slice_418, %2076 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2082 = tosa.add %2077, %2081 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2083 = tosa.mul %2059, %2073 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_419 = tensor.extract_slice %2059[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_420 = tensor.extract_slice %2059[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2084 = tensor.empty() : tensor<1x32x40x64xf32>
    %2085 = linalg.negf ins(%extracted_slice_420 : tensor<1x32x40x64xf32>) outs(%2084 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2086 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_421 = tensor.insert_slice %2085 into %2086[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_422 = tensor.insert_slice %extracted_slice_419 into %inserted_slice_421[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2087 = tosa.mul %inserted_slice_422, %2076 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2088 = tosa.add %2083, %2087 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2089 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2090 = tosa.transpose %2088, %2089 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2091 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2092 = tosa.add %2082, %2091 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2093 = tosa.reshape %2092 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2094 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2095 = tosa.add %2090, %2094 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2096 = tosa.reshape %2095 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2097 = tosa.matmul %2093, %2096 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2098 = tosa.reshape %2097 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2099 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2100 = tosa.reciprocal %2099 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2101 = tosa.mul %2098, %2100 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2102 = tosa.add %2101, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2103 = tosa.reduce_max %2102 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2104 = tosa.sub %2102, %2103 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2105 = tosa.exp %2104 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2106 = tosa.reduce_sum %2105 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2107 = tosa.reciprocal %2106 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2108 = tosa.mul %2105, %2107 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2109 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2110 = tosa.add %2108, %2109 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2111 = tosa.reshape %2110 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2112 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2113 = tosa.add %2062, %2112 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2114 = tosa.reshape %2113 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2115 = tosa.matmul %2111, %2114 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2116 = tosa.reshape %2115 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2117 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2118 = tosa.transpose %2116, %2117 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2119 = tosa.identity %2118 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2120 = tosa.reshape %2119 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2121 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2122 = tosa.transpose %arg184, %2121 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2123 = tosa.reshape %2120 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_423 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2124 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2123, %2122 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_423 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2125 = tosa.reshape %2124 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2126 = tosa.add %2029, %2125 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2127 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_424 = arith.constant 2 : i32
    %2128 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2126 : tensor<1x40x4096xf32>) outs(%2127 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_424 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_425 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2129 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2128 : tensor<1x40x4096xf32>) outs(%cst_425 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2130 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2131 = tosa.add %2129, %2130 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2132 = tosa.rsqrt %2131 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2133 = tosa.mul %2126, %2132 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2134 = tosa.reshape %arg185 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2135 = tosa.mul %2134, %2133 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2136 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2137 = tosa.transpose %arg186, %2136 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2138 = tosa.reshape %2135 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_426 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2139 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2138, %2137 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_426 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2140 = tosa.reshape %2139 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2141 = tosa.sigmoid %2140 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2142 = tosa.mul %2140, %2141 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2143 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2144 = tosa.transpose %arg187, %2143 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2145 = tosa.reshape %2135 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_427 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2146 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2145, %2144 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_427 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2147 = tosa.reshape %2146 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2148 = tosa.mul %2142, %2147 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2149 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2150 = tosa.transpose %arg188, %2149 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2151 = tosa.reshape %2148 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_428 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2152 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2151, %2150 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_428 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2153 = tosa.reshape %2152 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2154 = tosa.add %2126, %2153 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2155 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_429 = arith.constant 2 : i32
    %2156 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2154 : tensor<1x40x4096xf32>) outs(%2155 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_429 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_430 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2157 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2156 : tensor<1x40x4096xf32>) outs(%cst_430 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2158 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2159 = tosa.add %2157, %2158 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2160 = tosa.rsqrt %2159 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2161 = tosa.mul %2154, %2160 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2162 = tosa.reshape %arg189 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2163 = tosa.mul %2162, %2161 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2164 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2165 = tosa.transpose %arg190, %2164 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2166 = tosa.reshape %2163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_431 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2167 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2166, %2165 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_431 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2168 = tosa.reshape %2167 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2169 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2170 = tosa.transpose %arg191, %2169 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2171 = tosa.reshape %2163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_432 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2172 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2171, %2170 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_432 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2173 = tosa.reshape %2172 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2174 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2175 = tosa.transpose %arg192, %2174 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2176 = tosa.reshape %2163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_433 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2177 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2176, %2175 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_433 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2178 = tosa.reshape %2177 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2179 = tosa.reshape %2168 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2180 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2181 = tosa.transpose %2179, %2180 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2182 = tosa.reshape %2173 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2183 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2184 = tosa.transpose %2182, %2183 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2185 = tosa.reshape %2178 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2186 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2187 = tosa.transpose %2185, %2186 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_434 = tensor.extract_slice %arg193[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_435 = tensor.extract_slice %extracted_slice_434[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_436 = tensor.extract_slice %extracted_slice_435[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_437 = tensor.extract_slice %arg194[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_438 = tensor.extract_slice %extracted_slice_437[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_439 = tensor.extract_slice %extracted_slice_438[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2188 = tensor.empty() : tensor<1x40x128xf32>
    %2189 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_436 : tensor<1x1x40x128xf32>) outs(%2188 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2190 = tensor.empty() : tensor<40x128xf32>
    %2191 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2189 : tensor<1x40x128xf32>) outs(%2190 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2192 = tensor.empty() : tensor<1x40x128xf32>
    %2193 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_439 : tensor<1x1x40x128xf32>) outs(%2192 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2194 = tensor.empty() : tensor<40x128xf32>
    %2195 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2193 : tensor<1x40x128xf32>) outs(%2194 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2196 = tensor.empty() : tensor<1x40x128xf32>
    %2197 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2196 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2191[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2198 = tosa.reshape %2197 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2199 = tensor.empty() : tensor<1x40x128xf32>
    %2200 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2199 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2195[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2201 = tosa.reshape %2200 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2202 = tosa.mul %2181, %2198 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_440 = tensor.extract_slice %2181[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_441 = tensor.extract_slice %2181[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2203 = tensor.empty() : tensor<1x32x40x64xf32>
    %2204 = linalg.negf ins(%extracted_slice_441 : tensor<1x32x40x64xf32>) outs(%2203 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2205 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_442 = tensor.insert_slice %2204 into %2205[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_443 = tensor.insert_slice %extracted_slice_440 into %inserted_slice_442[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2206 = tosa.mul %inserted_slice_443, %2201 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2207 = tosa.add %2202, %2206 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2208 = tosa.mul %2184, %2198 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_444 = tensor.extract_slice %2184[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_445 = tensor.extract_slice %2184[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2209 = tensor.empty() : tensor<1x32x40x64xf32>
    %2210 = linalg.negf ins(%extracted_slice_445 : tensor<1x32x40x64xf32>) outs(%2209 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2211 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_446 = tensor.insert_slice %2210 into %2211[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_447 = tensor.insert_slice %extracted_slice_444 into %inserted_slice_446[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2212 = tosa.mul %inserted_slice_447, %2201 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2213 = tosa.add %2208, %2212 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2214 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2215 = tosa.transpose %2213, %2214 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2216 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2217 = tosa.add %2207, %2216 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2218 = tosa.reshape %2217 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2219 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2220 = tosa.add %2215, %2219 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2221 = tosa.reshape %2220 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2222 = tosa.matmul %2218, %2221 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2223 = tosa.reshape %2222 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2224 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2225 = tosa.reciprocal %2224 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2226 = tosa.mul %2223, %2225 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2227 = tosa.add %2226, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2228 = tosa.reduce_max %2227 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2229 = tosa.sub %2227, %2228 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2230 = tosa.exp %2229 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2231 = tosa.reduce_sum %2230 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2232 = tosa.reciprocal %2231 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2233 = tosa.mul %2230, %2232 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2234 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2235 = tosa.add %2233, %2234 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2236 = tosa.reshape %2235 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2237 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2238 = tosa.add %2187, %2237 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2239 = tosa.reshape %2238 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2240 = tosa.matmul %2236, %2239 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2241 = tosa.reshape %2240 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2242 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2243 = tosa.transpose %2241, %2242 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2244 = tosa.identity %2243 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2245 = tosa.reshape %2244 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2246 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2247 = tosa.transpose %arg195, %2246 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2248 = tosa.reshape %2245 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_448 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2249 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2248, %2247 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_448 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2250 = tosa.reshape %2249 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2251 = tosa.add %2154, %2250 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2252 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_449 = arith.constant 2 : i32
    %2253 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2251 : tensor<1x40x4096xf32>) outs(%2252 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_449 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_450 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2254 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2253 : tensor<1x40x4096xf32>) outs(%cst_450 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2255 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2256 = tosa.add %2254, %2255 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2257 = tosa.rsqrt %2256 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2258 = tosa.mul %2251, %2257 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2259 = tosa.reshape %arg196 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2260 = tosa.mul %2259, %2258 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2261 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2262 = tosa.transpose %arg197, %2261 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2263 = tosa.reshape %2260 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_451 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2264 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2263, %2262 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_451 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2265 = tosa.reshape %2264 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2266 = tosa.sigmoid %2265 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2267 = tosa.mul %2265, %2266 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2268 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2269 = tosa.transpose %arg198, %2268 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2270 = tosa.reshape %2260 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_452 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2271 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2270, %2269 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_452 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2272 = tosa.reshape %2271 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2273 = tosa.mul %2267, %2272 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2274 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2275 = tosa.transpose %arg199, %2274 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2276 = tosa.reshape %2273 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_453 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2277 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2276, %2275 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_453 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2278 = tosa.reshape %2277 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2279 = tosa.add %2251, %2278 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2280 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_454 = arith.constant 2 : i32
    %2281 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2279 : tensor<1x40x4096xf32>) outs(%2280 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_454 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_455 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2282 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2281 : tensor<1x40x4096xf32>) outs(%cst_455 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2283 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2284 = tosa.add %2282, %2283 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2285 = tosa.rsqrt %2284 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2286 = tosa.mul %2279, %2285 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2287 = tosa.reshape %arg200 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2288 = tosa.mul %2287, %2286 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2289 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2290 = tosa.transpose %arg201, %2289 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2291 = tosa.reshape %2288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_456 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2292 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2291, %2290 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_456 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2293 = tosa.reshape %2292 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2294 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2295 = tosa.transpose %arg202, %2294 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2296 = tosa.reshape %2288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_457 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2297 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2296, %2295 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_457 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2298 = tosa.reshape %2297 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2299 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2300 = tosa.transpose %arg203, %2299 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2301 = tosa.reshape %2288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_458 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2302 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2301, %2300 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_458 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2303 = tosa.reshape %2302 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2304 = tosa.reshape %2293 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2305 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2306 = tosa.transpose %2304, %2305 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2307 = tosa.reshape %2298 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2308 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2309 = tosa.transpose %2307, %2308 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2310 = tosa.reshape %2303 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2311 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2312 = tosa.transpose %2310, %2311 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_459 = tensor.extract_slice %arg204[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_460 = tensor.extract_slice %extracted_slice_459[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_461 = tensor.extract_slice %extracted_slice_460[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_462 = tensor.extract_slice %arg205[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_463 = tensor.extract_slice %extracted_slice_462[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_464 = tensor.extract_slice %extracted_slice_463[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2313 = tensor.empty() : tensor<1x40x128xf32>
    %2314 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_461 : tensor<1x1x40x128xf32>) outs(%2313 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2315 = tensor.empty() : tensor<40x128xf32>
    %2316 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2314 : tensor<1x40x128xf32>) outs(%2315 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2317 = tensor.empty() : tensor<1x40x128xf32>
    %2318 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_464 : tensor<1x1x40x128xf32>) outs(%2317 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2319 = tensor.empty() : tensor<40x128xf32>
    %2320 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2318 : tensor<1x40x128xf32>) outs(%2319 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2321 = tensor.empty() : tensor<1x40x128xf32>
    %2322 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2321 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2316[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2323 = tosa.reshape %2322 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2324 = tensor.empty() : tensor<1x40x128xf32>
    %2325 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2324 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2320[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2326 = tosa.reshape %2325 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2327 = tosa.mul %2306, %2323 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_465 = tensor.extract_slice %2306[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_466 = tensor.extract_slice %2306[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2328 = tensor.empty() : tensor<1x32x40x64xf32>
    %2329 = linalg.negf ins(%extracted_slice_466 : tensor<1x32x40x64xf32>) outs(%2328 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2330 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_467 = tensor.insert_slice %2329 into %2330[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_468 = tensor.insert_slice %extracted_slice_465 into %inserted_slice_467[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2331 = tosa.mul %inserted_slice_468, %2326 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2332 = tosa.add %2327, %2331 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2333 = tosa.mul %2309, %2323 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_469 = tensor.extract_slice %2309[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_470 = tensor.extract_slice %2309[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2334 = tensor.empty() : tensor<1x32x40x64xf32>
    %2335 = linalg.negf ins(%extracted_slice_470 : tensor<1x32x40x64xf32>) outs(%2334 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2336 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_471 = tensor.insert_slice %2335 into %2336[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_472 = tensor.insert_slice %extracted_slice_469 into %inserted_slice_471[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2337 = tosa.mul %inserted_slice_472, %2326 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2338 = tosa.add %2333, %2337 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2339 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2340 = tosa.transpose %2338, %2339 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2341 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2342 = tosa.add %2332, %2341 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2343 = tosa.reshape %2342 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2344 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2345 = tosa.add %2340, %2344 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2346 = tosa.reshape %2345 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2347 = tosa.matmul %2343, %2346 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2348 = tosa.reshape %2347 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2349 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2350 = tosa.reciprocal %2349 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2351 = tosa.mul %2348, %2350 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2352 = tosa.add %2351, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2353 = tosa.reduce_max %2352 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2354 = tosa.sub %2352, %2353 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2355 = tosa.exp %2354 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2356 = tosa.reduce_sum %2355 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2357 = tosa.reciprocal %2356 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2358 = tosa.mul %2355, %2357 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2359 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2360 = tosa.add %2358, %2359 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2361 = tosa.reshape %2360 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2362 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2363 = tosa.add %2312, %2362 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2364 = tosa.reshape %2363 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2365 = tosa.matmul %2361, %2364 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2366 = tosa.reshape %2365 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2367 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2368 = tosa.transpose %2366, %2367 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2369 = tosa.identity %2368 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2370 = tosa.reshape %2369 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2371 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2372 = tosa.transpose %arg206, %2371 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2373 = tosa.reshape %2370 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_473 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2374 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2373, %2372 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_473 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2375 = tosa.reshape %2374 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2376 = tosa.add %2279, %2375 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2377 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_474 = arith.constant 2 : i32
    %2378 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2376 : tensor<1x40x4096xf32>) outs(%2377 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_474 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_475 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2379 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2378 : tensor<1x40x4096xf32>) outs(%cst_475 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2380 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2381 = tosa.add %2379, %2380 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2382 = tosa.rsqrt %2381 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2383 = tosa.mul %2376, %2382 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2384 = tosa.reshape %arg207 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2385 = tosa.mul %2384, %2383 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2386 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2387 = tosa.transpose %arg208, %2386 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2388 = tosa.reshape %2385 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_476 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2389 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2388, %2387 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_476 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2390 = tosa.reshape %2389 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2391 = tosa.sigmoid %2390 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2392 = tosa.mul %2390, %2391 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2393 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2394 = tosa.transpose %arg209, %2393 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2395 = tosa.reshape %2385 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_477 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2396 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2395, %2394 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_477 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2397 = tosa.reshape %2396 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2398 = tosa.mul %2392, %2397 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2399 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2400 = tosa.transpose %arg210, %2399 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2401 = tosa.reshape %2398 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_478 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2402 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2401, %2400 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_478 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2403 = tosa.reshape %2402 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2404 = tosa.add %2376, %2403 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2405 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_479 = arith.constant 2 : i32
    %2406 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2404 : tensor<1x40x4096xf32>) outs(%2405 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_479 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_480 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2407 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2406 : tensor<1x40x4096xf32>) outs(%cst_480 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2408 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2409 = tosa.add %2407, %2408 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2410 = tosa.rsqrt %2409 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2411 = tosa.mul %2404, %2410 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2412 = tosa.reshape %arg211 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2413 = tosa.mul %2412, %2411 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2414 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2415 = tosa.transpose %arg212, %2414 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2416 = tosa.reshape %2413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_481 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2417 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2416, %2415 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_481 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2418 = tosa.reshape %2417 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2419 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2420 = tosa.transpose %arg213, %2419 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2421 = tosa.reshape %2413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_482 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2422 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2421, %2420 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_482 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2423 = tosa.reshape %2422 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2424 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2425 = tosa.transpose %arg214, %2424 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2426 = tosa.reshape %2413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_483 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2427 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2426, %2425 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_483 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2428 = tosa.reshape %2427 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2429 = tosa.reshape %2418 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2430 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2431 = tosa.transpose %2429, %2430 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2432 = tosa.reshape %2423 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2433 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2434 = tosa.transpose %2432, %2433 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2435 = tosa.reshape %2428 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2436 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2437 = tosa.transpose %2435, %2436 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_484 = tensor.extract_slice %arg215[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_485 = tensor.extract_slice %extracted_slice_484[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_486 = tensor.extract_slice %extracted_slice_485[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_487 = tensor.extract_slice %arg216[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_488 = tensor.extract_slice %extracted_slice_487[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_489 = tensor.extract_slice %extracted_slice_488[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2438 = tensor.empty() : tensor<1x40x128xf32>
    %2439 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_486 : tensor<1x1x40x128xf32>) outs(%2438 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2440 = tensor.empty() : tensor<40x128xf32>
    %2441 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2439 : tensor<1x40x128xf32>) outs(%2440 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2442 = tensor.empty() : tensor<1x40x128xf32>
    %2443 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_489 : tensor<1x1x40x128xf32>) outs(%2442 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2444 = tensor.empty() : tensor<40x128xf32>
    %2445 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2443 : tensor<1x40x128xf32>) outs(%2444 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2446 = tensor.empty() : tensor<1x40x128xf32>
    %2447 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2446 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2441[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2448 = tosa.reshape %2447 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2449 = tensor.empty() : tensor<1x40x128xf32>
    %2450 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2449 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2445[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2451 = tosa.reshape %2450 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2452 = tosa.mul %2431, %2448 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_490 = tensor.extract_slice %2431[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_491 = tensor.extract_slice %2431[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2453 = tensor.empty() : tensor<1x32x40x64xf32>
    %2454 = linalg.negf ins(%extracted_slice_491 : tensor<1x32x40x64xf32>) outs(%2453 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2455 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_492 = tensor.insert_slice %2454 into %2455[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_493 = tensor.insert_slice %extracted_slice_490 into %inserted_slice_492[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2456 = tosa.mul %inserted_slice_493, %2451 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2457 = tosa.add %2452, %2456 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2458 = tosa.mul %2434, %2448 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_494 = tensor.extract_slice %2434[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_495 = tensor.extract_slice %2434[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2459 = tensor.empty() : tensor<1x32x40x64xf32>
    %2460 = linalg.negf ins(%extracted_slice_495 : tensor<1x32x40x64xf32>) outs(%2459 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2461 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_496 = tensor.insert_slice %2460 into %2461[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_497 = tensor.insert_slice %extracted_slice_494 into %inserted_slice_496[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2462 = tosa.mul %inserted_slice_497, %2451 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2463 = tosa.add %2458, %2462 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2464 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2465 = tosa.transpose %2463, %2464 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2466 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2467 = tosa.add %2457, %2466 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2468 = tosa.reshape %2467 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2469 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2470 = tosa.add %2465, %2469 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2471 = tosa.reshape %2470 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2472 = tosa.matmul %2468, %2471 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2473 = tosa.reshape %2472 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2474 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2475 = tosa.reciprocal %2474 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2476 = tosa.mul %2473, %2475 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2477 = tosa.add %2476, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2478 = tosa.reduce_max %2477 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2479 = tosa.sub %2477, %2478 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2480 = tosa.exp %2479 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2481 = tosa.reduce_sum %2480 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2482 = tosa.reciprocal %2481 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2483 = tosa.mul %2480, %2482 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2484 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2485 = tosa.add %2483, %2484 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2486 = tosa.reshape %2485 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2487 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2488 = tosa.add %2437, %2487 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2489 = tosa.reshape %2488 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2490 = tosa.matmul %2486, %2489 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2491 = tosa.reshape %2490 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2492 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2493 = tosa.transpose %2491, %2492 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2494 = tosa.identity %2493 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2495 = tosa.reshape %2494 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2496 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2497 = tosa.transpose %arg217, %2496 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2498 = tosa.reshape %2495 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_498 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2499 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2498, %2497 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_498 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2500 = tosa.reshape %2499 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2501 = tosa.add %2404, %2500 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2502 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_499 = arith.constant 2 : i32
    %2503 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2501 : tensor<1x40x4096xf32>) outs(%2502 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_499 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_500 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2504 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2503 : tensor<1x40x4096xf32>) outs(%cst_500 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2505 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2506 = tosa.add %2504, %2505 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2507 = tosa.rsqrt %2506 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2508 = tosa.mul %2501, %2507 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2509 = tosa.reshape %arg218 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2510 = tosa.mul %2509, %2508 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2511 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2512 = tosa.transpose %arg219, %2511 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2513 = tosa.reshape %2510 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_501 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2514 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2513, %2512 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_501 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2515 = tosa.reshape %2514 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2516 = tosa.sigmoid %2515 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2517 = tosa.mul %2515, %2516 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2518 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2519 = tosa.transpose %arg220, %2518 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2520 = tosa.reshape %2510 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_502 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2521 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2520, %2519 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_502 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2522 = tosa.reshape %2521 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2523 = tosa.mul %2517, %2522 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2524 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2525 = tosa.transpose %arg221, %2524 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2526 = tosa.reshape %2523 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_503 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2527 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2526, %2525 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_503 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2528 = tosa.reshape %2527 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2529 = tosa.add %2501, %2528 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2530 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_504 = arith.constant 2 : i32
    %2531 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2529 : tensor<1x40x4096xf32>) outs(%2530 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_504 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_505 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2532 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2531 : tensor<1x40x4096xf32>) outs(%cst_505 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2533 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2534 = tosa.add %2532, %2533 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2535 = tosa.rsqrt %2534 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2536 = tosa.mul %2529, %2535 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2537 = tosa.reshape %arg222 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2538 = tosa.mul %2537, %2536 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2539 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2540 = tosa.transpose %arg223, %2539 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2541 = tosa.reshape %2538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_506 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2542 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2541, %2540 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_506 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2543 = tosa.reshape %2542 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2544 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2545 = tosa.transpose %arg224, %2544 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2546 = tosa.reshape %2538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_507 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2547 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2546, %2545 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_507 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2548 = tosa.reshape %2547 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2549 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2550 = tosa.transpose %arg225, %2549 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2551 = tosa.reshape %2538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_508 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2552 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2551, %2550 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_508 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2553 = tosa.reshape %2552 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2554 = tosa.reshape %2543 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2555 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2556 = tosa.transpose %2554, %2555 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2557 = tosa.reshape %2548 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2558 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2559 = tosa.transpose %2557, %2558 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2560 = tosa.reshape %2553 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2561 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2562 = tosa.transpose %2560, %2561 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_509 = tensor.extract_slice %arg226[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_510 = tensor.extract_slice %extracted_slice_509[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_511 = tensor.extract_slice %extracted_slice_510[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_512 = tensor.extract_slice %arg227[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_513 = tensor.extract_slice %extracted_slice_512[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_514 = tensor.extract_slice %extracted_slice_513[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2563 = tensor.empty() : tensor<1x40x128xf32>
    %2564 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_511 : tensor<1x1x40x128xf32>) outs(%2563 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2565 = tensor.empty() : tensor<40x128xf32>
    %2566 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2564 : tensor<1x40x128xf32>) outs(%2565 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2567 = tensor.empty() : tensor<1x40x128xf32>
    %2568 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_514 : tensor<1x1x40x128xf32>) outs(%2567 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2569 = tensor.empty() : tensor<40x128xf32>
    %2570 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2568 : tensor<1x40x128xf32>) outs(%2569 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2571 = tensor.empty() : tensor<1x40x128xf32>
    %2572 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2571 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2566[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2573 = tosa.reshape %2572 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2574 = tensor.empty() : tensor<1x40x128xf32>
    %2575 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2574 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2570[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2576 = tosa.reshape %2575 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2577 = tosa.mul %2556, %2573 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_515 = tensor.extract_slice %2556[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_516 = tensor.extract_slice %2556[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2578 = tensor.empty() : tensor<1x32x40x64xf32>
    %2579 = linalg.negf ins(%extracted_slice_516 : tensor<1x32x40x64xf32>) outs(%2578 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2580 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_517 = tensor.insert_slice %2579 into %2580[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_518 = tensor.insert_slice %extracted_slice_515 into %inserted_slice_517[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2581 = tosa.mul %inserted_slice_518, %2576 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2582 = tosa.add %2577, %2581 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2583 = tosa.mul %2559, %2573 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_519 = tensor.extract_slice %2559[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_520 = tensor.extract_slice %2559[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2584 = tensor.empty() : tensor<1x32x40x64xf32>
    %2585 = linalg.negf ins(%extracted_slice_520 : tensor<1x32x40x64xf32>) outs(%2584 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2586 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_521 = tensor.insert_slice %2585 into %2586[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_522 = tensor.insert_slice %extracted_slice_519 into %inserted_slice_521[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2587 = tosa.mul %inserted_slice_522, %2576 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2588 = tosa.add %2583, %2587 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2589 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2590 = tosa.transpose %2588, %2589 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2591 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2592 = tosa.add %2582, %2591 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2593 = tosa.reshape %2592 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2594 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2595 = tosa.add %2590, %2594 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2596 = tosa.reshape %2595 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2597 = tosa.matmul %2593, %2596 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2598 = tosa.reshape %2597 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2599 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2600 = tosa.reciprocal %2599 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2601 = tosa.mul %2598, %2600 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2602 = tosa.add %2601, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2603 = tosa.reduce_max %2602 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2604 = tosa.sub %2602, %2603 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2605 = tosa.exp %2604 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2606 = tosa.reduce_sum %2605 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2607 = tosa.reciprocal %2606 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2608 = tosa.mul %2605, %2607 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2609 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2610 = tosa.add %2608, %2609 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2611 = tosa.reshape %2610 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2612 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2613 = tosa.add %2562, %2612 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2614 = tosa.reshape %2613 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2615 = tosa.matmul %2611, %2614 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2616 = tosa.reshape %2615 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2617 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2618 = tosa.transpose %2616, %2617 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2619 = tosa.identity %2618 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2620 = tosa.reshape %2619 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2621 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2622 = tosa.transpose %arg228, %2621 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2623 = tosa.reshape %2620 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_523 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2624 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2623, %2622 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_523 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2625 = tosa.reshape %2624 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2626 = tosa.add %2529, %2625 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2627 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_524 = arith.constant 2 : i32
    %2628 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2626 : tensor<1x40x4096xf32>) outs(%2627 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_524 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_525 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2629 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2628 : tensor<1x40x4096xf32>) outs(%cst_525 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2630 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2631 = tosa.add %2629, %2630 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2632 = tosa.rsqrt %2631 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2633 = tosa.mul %2626, %2632 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2634 = tosa.reshape %arg229 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2635 = tosa.mul %2634, %2633 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2636 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2637 = tosa.transpose %arg230, %2636 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2638 = tosa.reshape %2635 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_526 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2639 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2638, %2637 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_526 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2640 = tosa.reshape %2639 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2641 = tosa.sigmoid %2640 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2642 = tosa.mul %2640, %2641 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2643 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2644 = tosa.transpose %arg231, %2643 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2645 = tosa.reshape %2635 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_527 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2646 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2645, %2644 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_527 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2647 = tosa.reshape %2646 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2648 = tosa.mul %2642, %2647 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2649 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2650 = tosa.transpose %arg232, %2649 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2651 = tosa.reshape %2648 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_528 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2652 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2651, %2650 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_528 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2653 = tosa.reshape %2652 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2654 = tosa.add %2626, %2653 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2655 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_529 = arith.constant 2 : i32
    %2656 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2654 : tensor<1x40x4096xf32>) outs(%2655 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_529 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_530 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2657 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2656 : tensor<1x40x4096xf32>) outs(%cst_530 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2658 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2659 = tosa.add %2657, %2658 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2660 = tosa.rsqrt %2659 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2661 = tosa.mul %2654, %2660 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2662 = tosa.reshape %arg233 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2663 = tosa.mul %2662, %2661 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2664 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2665 = tosa.transpose %arg234, %2664 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2666 = tosa.reshape %2663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_531 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2667 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2666, %2665 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_531 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2668 = tosa.reshape %2667 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2669 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2670 = tosa.transpose %arg235, %2669 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2671 = tosa.reshape %2663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_532 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2672 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2671, %2670 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_532 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2673 = tosa.reshape %2672 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2674 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2675 = tosa.transpose %arg236, %2674 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2676 = tosa.reshape %2663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_533 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2677 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2676, %2675 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_533 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2678 = tosa.reshape %2677 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2679 = tosa.reshape %2668 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2680 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2681 = tosa.transpose %2679, %2680 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2682 = tosa.reshape %2673 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2683 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2684 = tosa.transpose %2682, %2683 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2685 = tosa.reshape %2678 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2686 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2687 = tosa.transpose %2685, %2686 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_534 = tensor.extract_slice %arg237[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_535 = tensor.extract_slice %extracted_slice_534[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_536 = tensor.extract_slice %extracted_slice_535[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_537 = tensor.extract_slice %arg238[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_538 = tensor.extract_slice %extracted_slice_537[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_539 = tensor.extract_slice %extracted_slice_538[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2688 = tensor.empty() : tensor<1x40x128xf32>
    %2689 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_536 : tensor<1x1x40x128xf32>) outs(%2688 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2690 = tensor.empty() : tensor<40x128xf32>
    %2691 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2689 : tensor<1x40x128xf32>) outs(%2690 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2692 = tensor.empty() : tensor<1x40x128xf32>
    %2693 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_539 : tensor<1x1x40x128xf32>) outs(%2692 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2694 = tensor.empty() : tensor<40x128xf32>
    %2695 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2693 : tensor<1x40x128xf32>) outs(%2694 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2696 = tensor.empty() : tensor<1x40x128xf32>
    %2697 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2696 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2691[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2698 = tosa.reshape %2697 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2699 = tensor.empty() : tensor<1x40x128xf32>
    %2700 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2699 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2695[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2701 = tosa.reshape %2700 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2702 = tosa.mul %2681, %2698 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_540 = tensor.extract_slice %2681[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_541 = tensor.extract_slice %2681[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2703 = tensor.empty() : tensor<1x32x40x64xf32>
    %2704 = linalg.negf ins(%extracted_slice_541 : tensor<1x32x40x64xf32>) outs(%2703 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2705 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_542 = tensor.insert_slice %2704 into %2705[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_543 = tensor.insert_slice %extracted_slice_540 into %inserted_slice_542[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2706 = tosa.mul %inserted_slice_543, %2701 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2707 = tosa.add %2702, %2706 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2708 = tosa.mul %2684, %2698 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_544 = tensor.extract_slice %2684[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_545 = tensor.extract_slice %2684[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2709 = tensor.empty() : tensor<1x32x40x64xf32>
    %2710 = linalg.negf ins(%extracted_slice_545 : tensor<1x32x40x64xf32>) outs(%2709 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2711 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_546 = tensor.insert_slice %2710 into %2711[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_547 = tensor.insert_slice %extracted_slice_544 into %inserted_slice_546[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2712 = tosa.mul %inserted_slice_547, %2701 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2713 = tosa.add %2708, %2712 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2714 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2715 = tosa.transpose %2713, %2714 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2716 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2717 = tosa.add %2707, %2716 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2718 = tosa.reshape %2717 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2719 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2720 = tosa.add %2715, %2719 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2721 = tosa.reshape %2720 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2722 = tosa.matmul %2718, %2721 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2723 = tosa.reshape %2722 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2724 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2725 = tosa.reciprocal %2724 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2726 = tosa.mul %2723, %2725 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2727 = tosa.add %2726, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2728 = tosa.reduce_max %2727 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2729 = tosa.sub %2727, %2728 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2730 = tosa.exp %2729 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2731 = tosa.reduce_sum %2730 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2732 = tosa.reciprocal %2731 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2733 = tosa.mul %2730, %2732 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2734 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2735 = tosa.add %2733, %2734 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2736 = tosa.reshape %2735 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2737 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2738 = tosa.add %2687, %2737 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2739 = tosa.reshape %2738 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2740 = tosa.matmul %2736, %2739 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2741 = tosa.reshape %2740 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2742 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2743 = tosa.transpose %2741, %2742 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2744 = tosa.identity %2743 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2745 = tosa.reshape %2744 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2746 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2747 = tosa.transpose %arg239, %2746 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2748 = tosa.reshape %2745 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_548 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2749 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2748, %2747 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_548 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2750 = tosa.reshape %2749 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2751 = tosa.add %2654, %2750 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2752 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_549 = arith.constant 2 : i32
    %2753 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2751 : tensor<1x40x4096xf32>) outs(%2752 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_549 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_550 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2754 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2753 : tensor<1x40x4096xf32>) outs(%cst_550 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2755 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2756 = tosa.add %2754, %2755 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2757 = tosa.rsqrt %2756 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2758 = tosa.mul %2751, %2757 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2759 = tosa.reshape %arg240 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2760 = tosa.mul %2759, %2758 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2761 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2762 = tosa.transpose %arg241, %2761 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2763 = tosa.reshape %2760 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_551 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2764 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2763, %2762 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_551 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2765 = tosa.reshape %2764 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2766 = tosa.sigmoid %2765 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2767 = tosa.mul %2765, %2766 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2768 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2769 = tosa.transpose %arg242, %2768 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2770 = tosa.reshape %2760 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_552 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2771 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2770, %2769 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_552 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2772 = tosa.reshape %2771 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2773 = tosa.mul %2767, %2772 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2774 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2775 = tosa.transpose %arg243, %2774 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2776 = tosa.reshape %2773 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_553 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2777 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2776, %2775 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_553 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2778 = tosa.reshape %2777 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2779 = tosa.add %2751, %2778 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2780 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_554 = arith.constant 2 : i32
    %2781 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2779 : tensor<1x40x4096xf32>) outs(%2780 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_554 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_555 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2782 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2781 : tensor<1x40x4096xf32>) outs(%cst_555 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2783 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2784 = tosa.add %2782, %2783 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2785 = tosa.rsqrt %2784 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2786 = tosa.mul %2779, %2785 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2787 = tosa.reshape %arg244 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2788 = tosa.mul %2787, %2786 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2789 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2790 = tosa.transpose %arg245, %2789 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2791 = tosa.reshape %2788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_556 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2792 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2791, %2790 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_556 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2793 = tosa.reshape %2792 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2794 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2795 = tosa.transpose %arg246, %2794 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2796 = tosa.reshape %2788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_557 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2797 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2796, %2795 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_557 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2798 = tosa.reshape %2797 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2799 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2800 = tosa.transpose %arg247, %2799 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2801 = tosa.reshape %2788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_558 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2802 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2801, %2800 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_558 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2803 = tosa.reshape %2802 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2804 = tosa.reshape %2793 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2805 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2806 = tosa.transpose %2804, %2805 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2807 = tosa.reshape %2798 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2808 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2809 = tosa.transpose %2807, %2808 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2810 = tosa.reshape %2803 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2811 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2812 = tosa.transpose %2810, %2811 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_559 = tensor.extract_slice %arg248[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_560 = tensor.extract_slice %extracted_slice_559[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_561 = tensor.extract_slice %extracted_slice_560[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_562 = tensor.extract_slice %arg249[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_563 = tensor.extract_slice %extracted_slice_562[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_564 = tensor.extract_slice %extracted_slice_563[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2813 = tensor.empty() : tensor<1x40x128xf32>
    %2814 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_561 : tensor<1x1x40x128xf32>) outs(%2813 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2815 = tensor.empty() : tensor<40x128xf32>
    %2816 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2814 : tensor<1x40x128xf32>) outs(%2815 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2817 = tensor.empty() : tensor<1x40x128xf32>
    %2818 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_564 : tensor<1x1x40x128xf32>) outs(%2817 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2819 = tensor.empty() : tensor<40x128xf32>
    %2820 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2818 : tensor<1x40x128xf32>) outs(%2819 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2821 = tensor.empty() : tensor<1x40x128xf32>
    %2822 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2821 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2816[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2823 = tosa.reshape %2822 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2824 = tensor.empty() : tensor<1x40x128xf32>
    %2825 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2824 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2820[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2826 = tosa.reshape %2825 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2827 = tosa.mul %2806, %2823 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_565 = tensor.extract_slice %2806[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_566 = tensor.extract_slice %2806[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2828 = tensor.empty() : tensor<1x32x40x64xf32>
    %2829 = linalg.negf ins(%extracted_slice_566 : tensor<1x32x40x64xf32>) outs(%2828 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2830 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_567 = tensor.insert_slice %2829 into %2830[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_568 = tensor.insert_slice %extracted_slice_565 into %inserted_slice_567[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2831 = tosa.mul %inserted_slice_568, %2826 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2832 = tosa.add %2827, %2831 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2833 = tosa.mul %2809, %2823 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_569 = tensor.extract_slice %2809[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_570 = tensor.extract_slice %2809[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2834 = tensor.empty() : tensor<1x32x40x64xf32>
    %2835 = linalg.negf ins(%extracted_slice_570 : tensor<1x32x40x64xf32>) outs(%2834 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2836 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_571 = tensor.insert_slice %2835 into %2836[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_572 = tensor.insert_slice %extracted_slice_569 into %inserted_slice_571[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2837 = tosa.mul %inserted_slice_572, %2826 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2838 = tosa.add %2833, %2837 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2839 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2840 = tosa.transpose %2838, %2839 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2841 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2842 = tosa.add %2832, %2841 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2843 = tosa.reshape %2842 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2844 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2845 = tosa.add %2840, %2844 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2846 = tosa.reshape %2845 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2847 = tosa.matmul %2843, %2846 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2848 = tosa.reshape %2847 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2849 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2850 = tosa.reciprocal %2849 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2851 = tosa.mul %2848, %2850 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2852 = tosa.add %2851, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2853 = tosa.reduce_max %2852 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2854 = tosa.sub %2852, %2853 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2855 = tosa.exp %2854 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2856 = tosa.reduce_sum %2855 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2857 = tosa.reciprocal %2856 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2858 = tosa.mul %2855, %2857 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2859 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2860 = tosa.add %2858, %2859 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2861 = tosa.reshape %2860 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2862 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2863 = tosa.add %2812, %2862 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2864 = tosa.reshape %2863 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2865 = tosa.matmul %2861, %2864 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2866 = tosa.reshape %2865 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2867 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2868 = tosa.transpose %2866, %2867 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2869 = tosa.identity %2868 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2870 = tosa.reshape %2869 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2871 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2872 = tosa.transpose %arg250, %2871 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2873 = tosa.reshape %2870 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_573 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2874 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2873, %2872 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_573 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2875 = tosa.reshape %2874 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2876 = tosa.add %2779, %2875 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2877 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_574 = arith.constant 2 : i32
    %2878 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2876 : tensor<1x40x4096xf32>) outs(%2877 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_574 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_575 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2879 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2878 : tensor<1x40x4096xf32>) outs(%cst_575 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2880 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2881 = tosa.add %2879, %2880 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2882 = tosa.rsqrt %2881 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2883 = tosa.mul %2876, %2882 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2884 = tosa.reshape %arg251 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2885 = tosa.mul %2884, %2883 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2886 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2887 = tosa.transpose %arg252, %2886 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2888 = tosa.reshape %2885 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_576 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2889 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2888, %2887 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_576 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2890 = tosa.reshape %2889 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2891 = tosa.sigmoid %2890 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2892 = tosa.mul %2890, %2891 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2893 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2894 = tosa.transpose %arg253, %2893 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %2895 = tosa.reshape %2885 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_577 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2896 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2895, %2894 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_577 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %2897 = tosa.reshape %2896 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %2898 = tosa.mul %2892, %2897 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %2899 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2900 = tosa.transpose %arg254, %2899 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %2901 = tosa.reshape %2898 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_578 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2902 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2901, %2900 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_578 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2903 = tosa.reshape %2902 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2904 = tosa.add %2876, %2903 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2905 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_579 = arith.constant 2 : i32
    %2906 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2904 : tensor<1x40x4096xf32>) outs(%2905 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_579 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_580 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2907 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2906 : tensor<1x40x4096xf32>) outs(%cst_580 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %2908 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2909 = tosa.add %2907, %2908 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2910 = tosa.rsqrt %2909 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2911 = tosa.mul %2904, %2910 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2912 = tosa.reshape %arg255 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2913 = tosa.mul %2912, %2911 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %2914 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2915 = tosa.transpose %arg256, %2914 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2916 = tosa.reshape %2913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_581 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2917 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2916, %2915 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_581 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2918 = tosa.reshape %2917 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2919 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2920 = tosa.transpose %arg257, %2919 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2921 = tosa.reshape %2913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_582 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2922 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2921, %2920 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_582 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2923 = tosa.reshape %2922 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2924 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2925 = tosa.transpose %arg258, %2924 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2926 = tosa.reshape %2913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_583 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2927 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2926, %2925 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_583 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %2928 = tosa.reshape %2927 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %2929 = tosa.reshape %2918 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2930 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2931 = tosa.transpose %2929, %2930 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2932 = tosa.reshape %2923 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2933 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2934 = tosa.transpose %2932, %2933 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2935 = tosa.reshape %2928 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %2936 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2937 = tosa.transpose %2935, %2936 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_584 = tensor.extract_slice %arg259[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_585 = tensor.extract_slice %extracted_slice_584[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_586 = tensor.extract_slice %extracted_slice_585[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_587 = tensor.extract_slice %arg260[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_588 = tensor.extract_slice %extracted_slice_587[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_589 = tensor.extract_slice %extracted_slice_588[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2938 = tensor.empty() : tensor<1x40x128xf32>
    %2939 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_586 : tensor<1x1x40x128xf32>) outs(%2938 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2940 = tensor.empty() : tensor<40x128xf32>
    %2941 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2939 : tensor<1x40x128xf32>) outs(%2940 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2942 = tensor.empty() : tensor<1x40x128xf32>
    %2943 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_589 : tensor<1x1x40x128xf32>) outs(%2942 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2944 = tensor.empty() : tensor<40x128xf32>
    %2945 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%2943 : tensor<1x40x128xf32>) outs(%2944 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2946 = tensor.empty() : tensor<1x40x128xf32>
    %2947 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2946 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2941[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2948 = tosa.reshape %2947 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2949 = tensor.empty() : tensor<1x40x128xf32>
    %2950 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2949 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %2945[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %2951 = tosa.reshape %2950 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2952 = tosa.mul %2931, %2948 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_590 = tensor.extract_slice %2931[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_591 = tensor.extract_slice %2931[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2953 = tensor.empty() : tensor<1x32x40x64xf32>
    %2954 = linalg.negf ins(%extracted_slice_591 : tensor<1x32x40x64xf32>) outs(%2953 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2955 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_592 = tensor.insert_slice %2954 into %2955[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_593 = tensor.insert_slice %extracted_slice_590 into %inserted_slice_592[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2956 = tosa.mul %inserted_slice_593, %2951 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2957 = tosa.add %2952, %2956 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2958 = tosa.mul %2934, %2948 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_594 = tensor.extract_slice %2934[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_595 = tensor.extract_slice %2934[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2959 = tensor.empty() : tensor<1x32x40x64xf32>
    %2960 = linalg.negf ins(%extracted_slice_595 : tensor<1x32x40x64xf32>) outs(%2959 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2961 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_596 = tensor.insert_slice %2960 into %2961[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_597 = tensor.insert_slice %extracted_slice_594 into %inserted_slice_596[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %2962 = tosa.mul %inserted_slice_597, %2951 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2963 = tosa.add %2958, %2962 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2964 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2965 = tosa.transpose %2963, %2964 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %2966 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2967 = tosa.add %2957, %2966 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2968 = tosa.reshape %2967 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2969 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %2970 = tosa.add %2965, %2969 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %2971 = tosa.reshape %2970 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %2972 = tosa.matmul %2968, %2971 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %2973 = tosa.reshape %2972 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2974 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2975 = tosa.reciprocal %2974 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2976 = tosa.mul %2973, %2975 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2977 = tosa.add %2976, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2978 = tosa.reduce_max %2977 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2979 = tosa.sub %2977, %2978 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2980 = tosa.exp %2979 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2981 = tosa.reduce_sum %2980 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %2982 = tosa.reciprocal %2981 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %2983 = tosa.mul %2980, %2982 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %2984 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %2985 = tosa.add %2983, %2984 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %2986 = tosa.reshape %2985 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %2987 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %2988 = tosa.add %2937, %2987 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2989 = tosa.reshape %2988 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %2990 = tosa.matmul %2986, %2989 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %2991 = tosa.reshape %2990 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %2992 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2993 = tosa.transpose %2991, %2992 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %2994 = tosa.identity %2993 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %2995 = tosa.reshape %2994 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %2996 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2997 = tosa.transpose %arg261, %2996 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %2998 = tosa.reshape %2995 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_598 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2999 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2998, %2997 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_598 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3000 = tosa.reshape %2999 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3001 = tosa.add %2904, %3000 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3002 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_599 = arith.constant 2 : i32
    %3003 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3001 : tensor<1x40x4096xf32>) outs(%3002 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_599 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_600 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3004 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3003 : tensor<1x40x4096xf32>) outs(%cst_600 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3005 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3006 = tosa.add %3004, %3005 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3007 = tosa.rsqrt %3006 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3008 = tosa.mul %3001, %3007 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3009 = tosa.reshape %arg262 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3010 = tosa.mul %3009, %3008 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3011 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3012 = tosa.transpose %arg263, %3011 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3013 = tosa.reshape %3010 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_601 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3014 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3013, %3012 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_601 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3015 = tosa.reshape %3014 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3016 = tosa.sigmoid %3015 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3017 = tosa.mul %3015, %3016 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3018 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3019 = tosa.transpose %arg264, %3018 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3020 = tosa.reshape %3010 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_602 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3021 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3020, %3019 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_602 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3022 = tosa.reshape %3021 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3023 = tosa.mul %3017, %3022 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3024 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3025 = tosa.transpose %arg265, %3024 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3026 = tosa.reshape %3023 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_603 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3027 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3026, %3025 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_603 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3028 = tosa.reshape %3027 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3029 = tosa.add %3001, %3028 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3030 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_604 = arith.constant 2 : i32
    %3031 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3029 : tensor<1x40x4096xf32>) outs(%3030 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_604 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_605 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3032 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3031 : tensor<1x40x4096xf32>) outs(%cst_605 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3033 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3034 = tosa.add %3032, %3033 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3035 = tosa.rsqrt %3034 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3036 = tosa.mul %3029, %3035 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3037 = tosa.reshape %arg266 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3038 = tosa.mul %3037, %3036 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3039 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3040 = tosa.transpose %arg267, %3039 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3041 = tosa.reshape %3038 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_606 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3042 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3041, %3040 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_606 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3043 = tosa.reshape %3042 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3044 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3045 = tosa.transpose %arg268, %3044 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3046 = tosa.reshape %3038 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_607 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3047 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3046, %3045 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_607 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3048 = tosa.reshape %3047 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3049 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3050 = tosa.transpose %arg269, %3049 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3051 = tosa.reshape %3038 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_608 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3052 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3051, %3050 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_608 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3053 = tosa.reshape %3052 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3054 = tosa.reshape %3043 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3055 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3056 = tosa.transpose %3054, %3055 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3057 = tosa.reshape %3048 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3058 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3059 = tosa.transpose %3057, %3058 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3060 = tosa.reshape %3053 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3061 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3062 = tosa.transpose %3060, %3061 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_609 = tensor.extract_slice %arg270[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_610 = tensor.extract_slice %extracted_slice_609[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_611 = tensor.extract_slice %extracted_slice_610[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_612 = tensor.extract_slice %arg271[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_613 = tensor.extract_slice %extracted_slice_612[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_614 = tensor.extract_slice %extracted_slice_613[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3063 = tensor.empty() : tensor<1x40x128xf32>
    %3064 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_611 : tensor<1x1x40x128xf32>) outs(%3063 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3065 = tensor.empty() : tensor<40x128xf32>
    %3066 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3064 : tensor<1x40x128xf32>) outs(%3065 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3067 = tensor.empty() : tensor<1x40x128xf32>
    %3068 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_614 : tensor<1x1x40x128xf32>) outs(%3067 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3069 = tensor.empty() : tensor<40x128xf32>
    %3070 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3068 : tensor<1x40x128xf32>) outs(%3069 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3071 = tensor.empty() : tensor<1x40x128xf32>
    %3072 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3071 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3066[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3073 = tosa.reshape %3072 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3074 = tensor.empty() : tensor<1x40x128xf32>
    %3075 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3074 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3070[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3076 = tosa.reshape %3075 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3077 = tosa.mul %3056, %3073 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_615 = tensor.extract_slice %3056[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_616 = tensor.extract_slice %3056[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3078 = tensor.empty() : tensor<1x32x40x64xf32>
    %3079 = linalg.negf ins(%extracted_slice_616 : tensor<1x32x40x64xf32>) outs(%3078 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3080 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_617 = tensor.insert_slice %3079 into %3080[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_618 = tensor.insert_slice %extracted_slice_615 into %inserted_slice_617[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3081 = tosa.mul %inserted_slice_618, %3076 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3082 = tosa.add %3077, %3081 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3083 = tosa.mul %3059, %3073 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_619 = tensor.extract_slice %3059[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_620 = tensor.extract_slice %3059[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3084 = tensor.empty() : tensor<1x32x40x64xf32>
    %3085 = linalg.negf ins(%extracted_slice_620 : tensor<1x32x40x64xf32>) outs(%3084 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3086 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_621 = tensor.insert_slice %3085 into %3086[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_622 = tensor.insert_slice %extracted_slice_619 into %inserted_slice_621[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3087 = tosa.mul %inserted_slice_622, %3076 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3088 = tosa.add %3083, %3087 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3089 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3090 = tosa.transpose %3088, %3089 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3091 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3092 = tosa.add %3082, %3091 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3093 = tosa.reshape %3092 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3094 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3095 = tosa.add %3090, %3094 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3096 = tosa.reshape %3095 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3097 = tosa.matmul %3093, %3096 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3098 = tosa.reshape %3097 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3099 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3100 = tosa.reciprocal %3099 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3101 = tosa.mul %3098, %3100 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3102 = tosa.add %3101, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3103 = tosa.reduce_max %3102 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3104 = tosa.sub %3102, %3103 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3105 = tosa.exp %3104 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3106 = tosa.reduce_sum %3105 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3107 = tosa.reciprocal %3106 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3108 = tosa.mul %3105, %3107 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3109 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3110 = tosa.add %3108, %3109 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3111 = tosa.reshape %3110 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3112 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3113 = tosa.add %3062, %3112 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3114 = tosa.reshape %3113 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3115 = tosa.matmul %3111, %3114 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3116 = tosa.reshape %3115 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3117 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3118 = tosa.transpose %3116, %3117 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3119 = tosa.identity %3118 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3120 = tosa.reshape %3119 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3121 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3122 = tosa.transpose %arg272, %3121 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3123 = tosa.reshape %3120 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_623 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3124 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3123, %3122 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_623 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3125 = tosa.reshape %3124 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3126 = tosa.add %3029, %3125 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3127 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_624 = arith.constant 2 : i32
    %3128 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3126 : tensor<1x40x4096xf32>) outs(%3127 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_624 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_625 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3129 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3128 : tensor<1x40x4096xf32>) outs(%cst_625 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3130 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3131 = tosa.add %3129, %3130 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3132 = tosa.rsqrt %3131 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3133 = tosa.mul %3126, %3132 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3134 = tosa.reshape %arg273 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3135 = tosa.mul %3134, %3133 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3136 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3137 = tosa.transpose %arg274, %3136 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3138 = tosa.reshape %3135 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_626 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3139 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3138, %3137 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_626 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3140 = tosa.reshape %3139 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3141 = tosa.sigmoid %3140 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3142 = tosa.mul %3140, %3141 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3143 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3144 = tosa.transpose %arg275, %3143 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3145 = tosa.reshape %3135 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_627 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3146 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3145, %3144 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_627 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3147 = tosa.reshape %3146 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3148 = tosa.mul %3142, %3147 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3149 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3150 = tosa.transpose %arg276, %3149 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3151 = tosa.reshape %3148 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_628 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3152 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3151, %3150 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_628 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3153 = tosa.reshape %3152 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3154 = tosa.add %3126, %3153 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3155 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_629 = arith.constant 2 : i32
    %3156 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3154 : tensor<1x40x4096xf32>) outs(%3155 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_629 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_630 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3157 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3156 : tensor<1x40x4096xf32>) outs(%cst_630 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3158 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3159 = tosa.add %3157, %3158 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3160 = tosa.rsqrt %3159 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3161 = tosa.mul %3154, %3160 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3162 = tosa.reshape %arg277 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3163 = tosa.mul %3162, %3161 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3164 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3165 = tosa.transpose %arg278, %3164 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3166 = tosa.reshape %3163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_631 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3167 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3166, %3165 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_631 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3168 = tosa.reshape %3167 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3169 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3170 = tosa.transpose %arg279, %3169 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3171 = tosa.reshape %3163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_632 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3172 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3171, %3170 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_632 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3173 = tosa.reshape %3172 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3174 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3175 = tosa.transpose %arg280, %3174 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3176 = tosa.reshape %3163 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_633 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3177 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3176, %3175 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_633 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3178 = tosa.reshape %3177 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3179 = tosa.reshape %3168 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3180 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3181 = tosa.transpose %3179, %3180 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3182 = tosa.reshape %3173 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3183 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3184 = tosa.transpose %3182, %3183 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3185 = tosa.reshape %3178 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3186 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3187 = tosa.transpose %3185, %3186 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_634 = tensor.extract_slice %arg281[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_635 = tensor.extract_slice %extracted_slice_634[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_636 = tensor.extract_slice %extracted_slice_635[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_637 = tensor.extract_slice %arg282[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_638 = tensor.extract_slice %extracted_slice_637[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_639 = tensor.extract_slice %extracted_slice_638[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3188 = tensor.empty() : tensor<1x40x128xf32>
    %3189 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_636 : tensor<1x1x40x128xf32>) outs(%3188 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3190 = tensor.empty() : tensor<40x128xf32>
    %3191 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3189 : tensor<1x40x128xf32>) outs(%3190 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3192 = tensor.empty() : tensor<1x40x128xf32>
    %3193 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_639 : tensor<1x1x40x128xf32>) outs(%3192 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3194 = tensor.empty() : tensor<40x128xf32>
    %3195 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3193 : tensor<1x40x128xf32>) outs(%3194 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3196 = tensor.empty() : tensor<1x40x128xf32>
    %3197 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3196 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3191[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3198 = tosa.reshape %3197 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3199 = tensor.empty() : tensor<1x40x128xf32>
    %3200 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3199 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3195[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3201 = tosa.reshape %3200 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3202 = tosa.mul %3181, %3198 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_640 = tensor.extract_slice %3181[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_641 = tensor.extract_slice %3181[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3203 = tensor.empty() : tensor<1x32x40x64xf32>
    %3204 = linalg.negf ins(%extracted_slice_641 : tensor<1x32x40x64xf32>) outs(%3203 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3205 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_642 = tensor.insert_slice %3204 into %3205[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_643 = tensor.insert_slice %extracted_slice_640 into %inserted_slice_642[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3206 = tosa.mul %inserted_slice_643, %3201 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3207 = tosa.add %3202, %3206 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3208 = tosa.mul %3184, %3198 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_644 = tensor.extract_slice %3184[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_645 = tensor.extract_slice %3184[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3209 = tensor.empty() : tensor<1x32x40x64xf32>
    %3210 = linalg.negf ins(%extracted_slice_645 : tensor<1x32x40x64xf32>) outs(%3209 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3211 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_646 = tensor.insert_slice %3210 into %3211[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_647 = tensor.insert_slice %extracted_slice_644 into %inserted_slice_646[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3212 = tosa.mul %inserted_slice_647, %3201 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3213 = tosa.add %3208, %3212 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3214 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3215 = tosa.transpose %3213, %3214 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3216 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3217 = tosa.add %3207, %3216 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3218 = tosa.reshape %3217 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3219 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3220 = tosa.add %3215, %3219 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3221 = tosa.reshape %3220 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3222 = tosa.matmul %3218, %3221 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3223 = tosa.reshape %3222 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3224 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3225 = tosa.reciprocal %3224 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3226 = tosa.mul %3223, %3225 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3227 = tosa.add %3226, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3228 = tosa.reduce_max %3227 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3229 = tosa.sub %3227, %3228 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3230 = tosa.exp %3229 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3231 = tosa.reduce_sum %3230 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3232 = tosa.reciprocal %3231 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3233 = tosa.mul %3230, %3232 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3234 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3235 = tosa.add %3233, %3234 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3236 = tosa.reshape %3235 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3237 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3238 = tosa.add %3187, %3237 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3239 = tosa.reshape %3238 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3240 = tosa.matmul %3236, %3239 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3241 = tosa.reshape %3240 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3242 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3243 = tosa.transpose %3241, %3242 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3244 = tosa.identity %3243 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3245 = tosa.reshape %3244 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3246 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3247 = tosa.transpose %arg283, %3246 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3248 = tosa.reshape %3245 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_648 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3249 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3248, %3247 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_648 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3250 = tosa.reshape %3249 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3251 = tosa.add %3154, %3250 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3252 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_649 = arith.constant 2 : i32
    %3253 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3251 : tensor<1x40x4096xf32>) outs(%3252 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_649 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_650 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3254 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3253 : tensor<1x40x4096xf32>) outs(%cst_650 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3255 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3256 = tosa.add %3254, %3255 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3257 = tosa.rsqrt %3256 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3258 = tosa.mul %3251, %3257 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3259 = tosa.reshape %arg284 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3260 = tosa.mul %3259, %3258 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3261 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3262 = tosa.transpose %arg285, %3261 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3263 = tosa.reshape %3260 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_651 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3264 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3263, %3262 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_651 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3265 = tosa.reshape %3264 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3266 = tosa.sigmoid %3265 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3267 = tosa.mul %3265, %3266 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3268 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3269 = tosa.transpose %arg286, %3268 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3270 = tosa.reshape %3260 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_652 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3271 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3270, %3269 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_652 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3272 = tosa.reshape %3271 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3273 = tosa.mul %3267, %3272 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3274 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3275 = tosa.transpose %arg287, %3274 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3276 = tosa.reshape %3273 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_653 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3277 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3276, %3275 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_653 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3278 = tosa.reshape %3277 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3279 = tosa.add %3251, %3278 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3280 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_654 = arith.constant 2 : i32
    %3281 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3279 : tensor<1x40x4096xf32>) outs(%3280 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_654 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_655 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3282 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3281 : tensor<1x40x4096xf32>) outs(%cst_655 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3283 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3284 = tosa.add %3282, %3283 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3285 = tosa.rsqrt %3284 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3286 = tosa.mul %3279, %3285 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3287 = tosa.reshape %arg288 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3288 = tosa.mul %3287, %3286 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3289 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3290 = tosa.transpose %arg289, %3289 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3291 = tosa.reshape %3288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_656 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3292 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3291, %3290 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_656 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3293 = tosa.reshape %3292 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3294 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3295 = tosa.transpose %arg290, %3294 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3296 = tosa.reshape %3288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_657 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3297 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3296, %3295 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_657 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3298 = tosa.reshape %3297 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3299 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3300 = tosa.transpose %arg291, %3299 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3301 = tosa.reshape %3288 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_658 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3302 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3301, %3300 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_658 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3303 = tosa.reshape %3302 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3304 = tosa.reshape %3293 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3305 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3306 = tosa.transpose %3304, %3305 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3307 = tosa.reshape %3298 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3308 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3309 = tosa.transpose %3307, %3308 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3310 = tosa.reshape %3303 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3311 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3312 = tosa.transpose %3310, %3311 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_659 = tensor.extract_slice %arg292[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_660 = tensor.extract_slice %extracted_slice_659[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_661 = tensor.extract_slice %extracted_slice_660[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_662 = tensor.extract_slice %arg293[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_663 = tensor.extract_slice %extracted_slice_662[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_664 = tensor.extract_slice %extracted_slice_663[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3313 = tensor.empty() : tensor<1x40x128xf32>
    %3314 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_661 : tensor<1x1x40x128xf32>) outs(%3313 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3315 = tensor.empty() : tensor<40x128xf32>
    %3316 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3314 : tensor<1x40x128xf32>) outs(%3315 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3317 = tensor.empty() : tensor<1x40x128xf32>
    %3318 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_664 : tensor<1x1x40x128xf32>) outs(%3317 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3319 = tensor.empty() : tensor<40x128xf32>
    %3320 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3318 : tensor<1x40x128xf32>) outs(%3319 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3321 = tensor.empty() : tensor<1x40x128xf32>
    %3322 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3321 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3316[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3323 = tosa.reshape %3322 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3324 = tensor.empty() : tensor<1x40x128xf32>
    %3325 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3324 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3320[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3326 = tosa.reshape %3325 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3327 = tosa.mul %3306, %3323 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_665 = tensor.extract_slice %3306[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_666 = tensor.extract_slice %3306[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3328 = tensor.empty() : tensor<1x32x40x64xf32>
    %3329 = linalg.negf ins(%extracted_slice_666 : tensor<1x32x40x64xf32>) outs(%3328 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3330 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_667 = tensor.insert_slice %3329 into %3330[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_668 = tensor.insert_slice %extracted_slice_665 into %inserted_slice_667[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3331 = tosa.mul %inserted_slice_668, %3326 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3332 = tosa.add %3327, %3331 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3333 = tosa.mul %3309, %3323 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_669 = tensor.extract_slice %3309[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_670 = tensor.extract_slice %3309[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3334 = tensor.empty() : tensor<1x32x40x64xf32>
    %3335 = linalg.negf ins(%extracted_slice_670 : tensor<1x32x40x64xf32>) outs(%3334 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3336 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_671 = tensor.insert_slice %3335 into %3336[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_672 = tensor.insert_slice %extracted_slice_669 into %inserted_slice_671[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3337 = tosa.mul %inserted_slice_672, %3326 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3338 = tosa.add %3333, %3337 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3339 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3340 = tosa.transpose %3338, %3339 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3341 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3342 = tosa.add %3332, %3341 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3343 = tosa.reshape %3342 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3344 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3345 = tosa.add %3340, %3344 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3346 = tosa.reshape %3345 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3347 = tosa.matmul %3343, %3346 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3348 = tosa.reshape %3347 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3349 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3350 = tosa.reciprocal %3349 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3351 = tosa.mul %3348, %3350 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3352 = tosa.add %3351, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3353 = tosa.reduce_max %3352 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3354 = tosa.sub %3352, %3353 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3355 = tosa.exp %3354 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3356 = tosa.reduce_sum %3355 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3357 = tosa.reciprocal %3356 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3358 = tosa.mul %3355, %3357 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3359 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3360 = tosa.add %3358, %3359 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3361 = tosa.reshape %3360 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3362 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3363 = tosa.add %3312, %3362 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3364 = tosa.reshape %3363 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3365 = tosa.matmul %3361, %3364 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3366 = tosa.reshape %3365 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3367 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3368 = tosa.transpose %3366, %3367 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3369 = tosa.identity %3368 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3370 = tosa.reshape %3369 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3371 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3372 = tosa.transpose %arg294, %3371 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3373 = tosa.reshape %3370 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_673 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3374 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3373, %3372 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_673 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3375 = tosa.reshape %3374 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3376 = tosa.add %3279, %3375 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3377 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_674 = arith.constant 2 : i32
    %3378 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3376 : tensor<1x40x4096xf32>) outs(%3377 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_674 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_675 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3379 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3378 : tensor<1x40x4096xf32>) outs(%cst_675 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3380 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3381 = tosa.add %3379, %3380 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3382 = tosa.rsqrt %3381 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3383 = tosa.mul %3376, %3382 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3384 = tosa.reshape %arg295 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3385 = tosa.mul %3384, %3383 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3386 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3387 = tosa.transpose %arg296, %3386 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3388 = tosa.reshape %3385 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_676 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3389 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3388, %3387 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_676 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3390 = tosa.reshape %3389 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3391 = tosa.sigmoid %3390 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3392 = tosa.mul %3390, %3391 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3393 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3394 = tosa.transpose %arg297, %3393 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3395 = tosa.reshape %3385 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_677 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3396 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3395, %3394 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_677 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3397 = tosa.reshape %3396 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3398 = tosa.mul %3392, %3397 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3399 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3400 = tosa.transpose %arg298, %3399 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3401 = tosa.reshape %3398 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_678 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3402 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3401, %3400 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_678 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3403 = tosa.reshape %3402 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3404 = tosa.add %3376, %3403 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3405 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_679 = arith.constant 2 : i32
    %3406 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3404 : tensor<1x40x4096xf32>) outs(%3405 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_679 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_680 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3407 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3406 : tensor<1x40x4096xf32>) outs(%cst_680 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3408 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3409 = tosa.add %3407, %3408 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3410 = tosa.rsqrt %3409 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3411 = tosa.mul %3404, %3410 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3412 = tosa.reshape %arg299 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3413 = tosa.mul %3412, %3411 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3414 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3415 = tosa.transpose %arg300, %3414 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3416 = tosa.reshape %3413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_681 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3417 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3416, %3415 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_681 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3418 = tosa.reshape %3417 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3419 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3420 = tosa.transpose %arg301, %3419 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3421 = tosa.reshape %3413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_682 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3422 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3421, %3420 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_682 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3423 = tosa.reshape %3422 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3424 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3425 = tosa.transpose %arg302, %3424 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3426 = tosa.reshape %3413 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_683 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3427 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3426, %3425 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_683 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3428 = tosa.reshape %3427 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3429 = tosa.reshape %3418 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3430 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3431 = tosa.transpose %3429, %3430 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3432 = tosa.reshape %3423 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3433 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3434 = tosa.transpose %3432, %3433 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3435 = tosa.reshape %3428 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3436 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3437 = tosa.transpose %3435, %3436 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_684 = tensor.extract_slice %arg303[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_685 = tensor.extract_slice %extracted_slice_684[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_686 = tensor.extract_slice %extracted_slice_685[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_687 = tensor.extract_slice %arg304[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_688 = tensor.extract_slice %extracted_slice_687[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_689 = tensor.extract_slice %extracted_slice_688[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3438 = tensor.empty() : tensor<1x40x128xf32>
    %3439 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_686 : tensor<1x1x40x128xf32>) outs(%3438 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3440 = tensor.empty() : tensor<40x128xf32>
    %3441 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3439 : tensor<1x40x128xf32>) outs(%3440 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3442 = tensor.empty() : tensor<1x40x128xf32>
    %3443 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_689 : tensor<1x1x40x128xf32>) outs(%3442 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3444 = tensor.empty() : tensor<40x128xf32>
    %3445 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3443 : tensor<1x40x128xf32>) outs(%3444 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3446 = tensor.empty() : tensor<1x40x128xf32>
    %3447 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3446 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3441[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3448 = tosa.reshape %3447 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3449 = tensor.empty() : tensor<1x40x128xf32>
    %3450 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3449 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3445[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3451 = tosa.reshape %3450 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3452 = tosa.mul %3431, %3448 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_690 = tensor.extract_slice %3431[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_691 = tensor.extract_slice %3431[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3453 = tensor.empty() : tensor<1x32x40x64xf32>
    %3454 = linalg.negf ins(%extracted_slice_691 : tensor<1x32x40x64xf32>) outs(%3453 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3455 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_692 = tensor.insert_slice %3454 into %3455[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_693 = tensor.insert_slice %extracted_slice_690 into %inserted_slice_692[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3456 = tosa.mul %inserted_slice_693, %3451 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3457 = tosa.add %3452, %3456 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3458 = tosa.mul %3434, %3448 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_694 = tensor.extract_slice %3434[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_695 = tensor.extract_slice %3434[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3459 = tensor.empty() : tensor<1x32x40x64xf32>
    %3460 = linalg.negf ins(%extracted_slice_695 : tensor<1x32x40x64xf32>) outs(%3459 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3461 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_696 = tensor.insert_slice %3460 into %3461[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_697 = tensor.insert_slice %extracted_slice_694 into %inserted_slice_696[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3462 = tosa.mul %inserted_slice_697, %3451 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3463 = tosa.add %3458, %3462 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3464 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3465 = tosa.transpose %3463, %3464 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3466 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3467 = tosa.add %3457, %3466 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3468 = tosa.reshape %3467 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3469 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3470 = tosa.add %3465, %3469 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3471 = tosa.reshape %3470 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3472 = tosa.matmul %3468, %3471 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3473 = tosa.reshape %3472 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3474 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3475 = tosa.reciprocal %3474 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3476 = tosa.mul %3473, %3475 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3477 = tosa.add %3476, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3478 = tosa.reduce_max %3477 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3479 = tosa.sub %3477, %3478 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3480 = tosa.exp %3479 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3481 = tosa.reduce_sum %3480 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3482 = tosa.reciprocal %3481 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3483 = tosa.mul %3480, %3482 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3484 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3485 = tosa.add %3483, %3484 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3486 = tosa.reshape %3485 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3487 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3488 = tosa.add %3437, %3487 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3489 = tosa.reshape %3488 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3490 = tosa.matmul %3486, %3489 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3491 = tosa.reshape %3490 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3492 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3493 = tosa.transpose %3491, %3492 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3494 = tosa.identity %3493 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3495 = tosa.reshape %3494 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3496 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3497 = tosa.transpose %arg305, %3496 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3498 = tosa.reshape %3495 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_698 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3499 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3498, %3497 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_698 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3500 = tosa.reshape %3499 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3501 = tosa.add %3404, %3500 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3502 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_699 = arith.constant 2 : i32
    %3503 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3501 : tensor<1x40x4096xf32>) outs(%3502 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_699 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_700 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3504 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3503 : tensor<1x40x4096xf32>) outs(%cst_700 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3505 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3506 = tosa.add %3504, %3505 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3507 = tosa.rsqrt %3506 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3508 = tosa.mul %3501, %3507 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3509 = tosa.reshape %arg306 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3510 = tosa.mul %3509, %3508 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3511 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3512 = tosa.transpose %arg307, %3511 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3513 = tosa.reshape %3510 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_701 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3514 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3513, %3512 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_701 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3515 = tosa.reshape %3514 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3516 = tosa.sigmoid %3515 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3517 = tosa.mul %3515, %3516 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3518 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3519 = tosa.transpose %arg308, %3518 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3520 = tosa.reshape %3510 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_702 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3521 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3520, %3519 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_702 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3522 = tosa.reshape %3521 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3523 = tosa.mul %3517, %3522 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3524 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3525 = tosa.transpose %arg309, %3524 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3526 = tosa.reshape %3523 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_703 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3527 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3526, %3525 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_703 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3528 = tosa.reshape %3527 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3529 = tosa.add %3501, %3528 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3530 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_704 = arith.constant 2 : i32
    %3531 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3529 : tensor<1x40x4096xf32>) outs(%3530 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_704 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_705 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3532 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3531 : tensor<1x40x4096xf32>) outs(%cst_705 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3533 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3534 = tosa.add %3532, %3533 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3535 = tosa.rsqrt %3534 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3536 = tosa.mul %3529, %3535 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3537 = tosa.reshape %arg310 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3538 = tosa.mul %3537, %3536 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3539 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3540 = tosa.transpose %arg311, %3539 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3541 = tosa.reshape %3538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_706 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3542 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3541, %3540 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_706 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3543 = tosa.reshape %3542 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3544 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3545 = tosa.transpose %arg312, %3544 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3546 = tosa.reshape %3538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_707 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3547 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3546, %3545 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_707 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3548 = tosa.reshape %3547 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3549 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3550 = tosa.transpose %arg313, %3549 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3551 = tosa.reshape %3538 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_708 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3552 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3551, %3550 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_708 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3553 = tosa.reshape %3552 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3554 = tosa.reshape %3543 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3555 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3556 = tosa.transpose %3554, %3555 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3557 = tosa.reshape %3548 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3558 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3559 = tosa.transpose %3557, %3558 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3560 = tosa.reshape %3553 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3561 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3562 = tosa.transpose %3560, %3561 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_709 = tensor.extract_slice %arg314[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_710 = tensor.extract_slice %extracted_slice_709[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_711 = tensor.extract_slice %extracted_slice_710[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_712 = tensor.extract_slice %arg315[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_713 = tensor.extract_slice %extracted_slice_712[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_714 = tensor.extract_slice %extracted_slice_713[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3563 = tensor.empty() : tensor<1x40x128xf32>
    %3564 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_711 : tensor<1x1x40x128xf32>) outs(%3563 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3565 = tensor.empty() : tensor<40x128xf32>
    %3566 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3564 : tensor<1x40x128xf32>) outs(%3565 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3567 = tensor.empty() : tensor<1x40x128xf32>
    %3568 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_714 : tensor<1x1x40x128xf32>) outs(%3567 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3569 = tensor.empty() : tensor<40x128xf32>
    %3570 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3568 : tensor<1x40x128xf32>) outs(%3569 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3571 = tensor.empty() : tensor<1x40x128xf32>
    %3572 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3571 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3566[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3573 = tosa.reshape %3572 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3574 = tensor.empty() : tensor<1x40x128xf32>
    %3575 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3574 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3570[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3576 = tosa.reshape %3575 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3577 = tosa.mul %3556, %3573 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_715 = tensor.extract_slice %3556[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_716 = tensor.extract_slice %3556[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3578 = tensor.empty() : tensor<1x32x40x64xf32>
    %3579 = linalg.negf ins(%extracted_slice_716 : tensor<1x32x40x64xf32>) outs(%3578 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3580 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_717 = tensor.insert_slice %3579 into %3580[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_718 = tensor.insert_slice %extracted_slice_715 into %inserted_slice_717[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3581 = tosa.mul %inserted_slice_718, %3576 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3582 = tosa.add %3577, %3581 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3583 = tosa.mul %3559, %3573 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_719 = tensor.extract_slice %3559[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_720 = tensor.extract_slice %3559[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3584 = tensor.empty() : tensor<1x32x40x64xf32>
    %3585 = linalg.negf ins(%extracted_slice_720 : tensor<1x32x40x64xf32>) outs(%3584 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3586 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_721 = tensor.insert_slice %3585 into %3586[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_722 = tensor.insert_slice %extracted_slice_719 into %inserted_slice_721[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3587 = tosa.mul %inserted_slice_722, %3576 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3588 = tosa.add %3583, %3587 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3589 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3590 = tosa.transpose %3588, %3589 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3591 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3592 = tosa.add %3582, %3591 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3593 = tosa.reshape %3592 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3594 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3595 = tosa.add %3590, %3594 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3596 = tosa.reshape %3595 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3597 = tosa.matmul %3593, %3596 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3598 = tosa.reshape %3597 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3599 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3600 = tosa.reciprocal %3599 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3601 = tosa.mul %3598, %3600 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3602 = tosa.add %3601, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3603 = tosa.reduce_max %3602 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3604 = tosa.sub %3602, %3603 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3605 = tosa.exp %3604 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3606 = tosa.reduce_sum %3605 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3607 = tosa.reciprocal %3606 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3608 = tosa.mul %3605, %3607 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3609 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3610 = tosa.add %3608, %3609 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3611 = tosa.reshape %3610 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3612 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3613 = tosa.add %3562, %3612 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3614 = tosa.reshape %3613 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3615 = tosa.matmul %3611, %3614 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3616 = tosa.reshape %3615 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3617 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3618 = tosa.transpose %3616, %3617 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3619 = tosa.identity %3618 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3620 = tosa.reshape %3619 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3621 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3622 = tosa.transpose %arg316, %3621 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3623 = tosa.reshape %3620 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_723 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3624 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3623, %3622 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_723 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3625 = tosa.reshape %3624 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3626 = tosa.add %3529, %3625 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3627 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_724 = arith.constant 2 : i32
    %3628 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3626 : tensor<1x40x4096xf32>) outs(%3627 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_724 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_725 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3629 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3628 : tensor<1x40x4096xf32>) outs(%cst_725 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3630 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3631 = tosa.add %3629, %3630 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3632 = tosa.rsqrt %3631 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3633 = tosa.mul %3626, %3632 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3634 = tosa.reshape %arg317 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3635 = tosa.mul %3634, %3633 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3636 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3637 = tosa.transpose %arg318, %3636 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3638 = tosa.reshape %3635 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_726 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3639 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3638, %3637 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_726 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3640 = tosa.reshape %3639 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3641 = tosa.sigmoid %3640 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3642 = tosa.mul %3640, %3641 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3643 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3644 = tosa.transpose %arg319, %3643 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3645 = tosa.reshape %3635 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_727 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3646 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3645, %3644 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_727 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3647 = tosa.reshape %3646 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3648 = tosa.mul %3642, %3647 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3649 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3650 = tosa.transpose %arg320, %3649 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3651 = tosa.reshape %3648 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_728 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3652 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3651, %3650 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_728 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3653 = tosa.reshape %3652 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3654 = tosa.add %3626, %3653 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3655 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_729 = arith.constant 2 : i32
    %3656 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3654 : tensor<1x40x4096xf32>) outs(%3655 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_729 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_730 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3657 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3656 : tensor<1x40x4096xf32>) outs(%cst_730 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3658 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3659 = tosa.add %3657, %3658 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3660 = tosa.rsqrt %3659 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3661 = tosa.mul %3654, %3660 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3662 = tosa.reshape %arg321 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3663 = tosa.mul %3662, %3661 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3664 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3665 = tosa.transpose %arg322, %3664 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3666 = tosa.reshape %3663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_731 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3667 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3666, %3665 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_731 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3668 = tosa.reshape %3667 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3669 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3670 = tosa.transpose %arg323, %3669 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3671 = tosa.reshape %3663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_732 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3672 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3671, %3670 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_732 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3673 = tosa.reshape %3672 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3674 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3675 = tosa.transpose %arg324, %3674 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3676 = tosa.reshape %3663 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_733 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3677 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3676, %3675 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_733 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3678 = tosa.reshape %3677 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3679 = tosa.reshape %3668 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3680 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3681 = tosa.transpose %3679, %3680 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3682 = tosa.reshape %3673 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3683 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3684 = tosa.transpose %3682, %3683 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3685 = tosa.reshape %3678 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3686 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3687 = tosa.transpose %3685, %3686 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_734 = tensor.extract_slice %arg325[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_735 = tensor.extract_slice %extracted_slice_734[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_736 = tensor.extract_slice %extracted_slice_735[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_737 = tensor.extract_slice %arg326[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_738 = tensor.extract_slice %extracted_slice_737[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_739 = tensor.extract_slice %extracted_slice_738[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3688 = tensor.empty() : tensor<1x40x128xf32>
    %3689 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_736 : tensor<1x1x40x128xf32>) outs(%3688 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3690 = tensor.empty() : tensor<40x128xf32>
    %3691 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3689 : tensor<1x40x128xf32>) outs(%3690 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3692 = tensor.empty() : tensor<1x40x128xf32>
    %3693 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_739 : tensor<1x1x40x128xf32>) outs(%3692 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3694 = tensor.empty() : tensor<40x128xf32>
    %3695 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3693 : tensor<1x40x128xf32>) outs(%3694 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3696 = tensor.empty() : tensor<1x40x128xf32>
    %3697 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3696 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3691[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3698 = tosa.reshape %3697 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3699 = tensor.empty() : tensor<1x40x128xf32>
    %3700 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3699 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3695[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3701 = tosa.reshape %3700 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3702 = tosa.mul %3681, %3698 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_740 = tensor.extract_slice %3681[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_741 = tensor.extract_slice %3681[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3703 = tensor.empty() : tensor<1x32x40x64xf32>
    %3704 = linalg.negf ins(%extracted_slice_741 : tensor<1x32x40x64xf32>) outs(%3703 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3705 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_742 = tensor.insert_slice %3704 into %3705[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_743 = tensor.insert_slice %extracted_slice_740 into %inserted_slice_742[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3706 = tosa.mul %inserted_slice_743, %3701 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3707 = tosa.add %3702, %3706 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3708 = tosa.mul %3684, %3698 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_744 = tensor.extract_slice %3684[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_745 = tensor.extract_slice %3684[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3709 = tensor.empty() : tensor<1x32x40x64xf32>
    %3710 = linalg.negf ins(%extracted_slice_745 : tensor<1x32x40x64xf32>) outs(%3709 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3711 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_746 = tensor.insert_slice %3710 into %3711[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_747 = tensor.insert_slice %extracted_slice_744 into %inserted_slice_746[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3712 = tosa.mul %inserted_slice_747, %3701 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3713 = tosa.add %3708, %3712 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3714 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3715 = tosa.transpose %3713, %3714 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3716 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3717 = tosa.add %3707, %3716 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3718 = tosa.reshape %3717 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3719 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3720 = tosa.add %3715, %3719 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3721 = tosa.reshape %3720 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3722 = tosa.matmul %3718, %3721 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3723 = tosa.reshape %3722 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3724 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3725 = tosa.reciprocal %3724 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3726 = tosa.mul %3723, %3725 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3727 = tosa.add %3726, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3728 = tosa.reduce_max %3727 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3729 = tosa.sub %3727, %3728 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3730 = tosa.exp %3729 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3731 = tosa.reduce_sum %3730 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3732 = tosa.reciprocal %3731 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3733 = tosa.mul %3730, %3732 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3734 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3735 = tosa.add %3733, %3734 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3736 = tosa.reshape %3735 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3737 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3738 = tosa.add %3687, %3737 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3739 = tosa.reshape %3738 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3740 = tosa.matmul %3736, %3739 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3741 = tosa.reshape %3740 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3742 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3743 = tosa.transpose %3741, %3742 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3744 = tosa.identity %3743 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3745 = tosa.reshape %3744 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3746 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3747 = tosa.transpose %arg327, %3746 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3748 = tosa.reshape %3745 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_748 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3749 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3748, %3747 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_748 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3750 = tosa.reshape %3749 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3751 = tosa.add %3654, %3750 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3752 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_749 = arith.constant 2 : i32
    %3753 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3751 : tensor<1x40x4096xf32>) outs(%3752 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_749 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_750 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3754 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3753 : tensor<1x40x4096xf32>) outs(%cst_750 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3755 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3756 = tosa.add %3754, %3755 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3757 = tosa.rsqrt %3756 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3758 = tosa.mul %3751, %3757 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3759 = tosa.reshape %arg328 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3760 = tosa.mul %3759, %3758 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3761 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3762 = tosa.transpose %arg329, %3761 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3763 = tosa.reshape %3760 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_751 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3764 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3763, %3762 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_751 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3765 = tosa.reshape %3764 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3766 = tosa.sigmoid %3765 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3767 = tosa.mul %3765, %3766 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3768 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3769 = tosa.transpose %arg330, %3768 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3770 = tosa.reshape %3760 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_752 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3771 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3770, %3769 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_752 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3772 = tosa.reshape %3771 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3773 = tosa.mul %3767, %3772 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3774 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3775 = tosa.transpose %arg331, %3774 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3776 = tosa.reshape %3773 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_753 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3777 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3776, %3775 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_753 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3778 = tosa.reshape %3777 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3779 = tosa.add %3751, %3778 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3780 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_754 = arith.constant 2 : i32
    %3781 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3779 : tensor<1x40x4096xf32>) outs(%3780 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_754 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_755 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3782 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3781 : tensor<1x40x4096xf32>) outs(%cst_755 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3783 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3784 = tosa.add %3782, %3783 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3785 = tosa.rsqrt %3784 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3786 = tosa.mul %3779, %3785 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3787 = tosa.reshape %arg332 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3788 = tosa.mul %3787, %3786 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3789 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3790 = tosa.transpose %arg333, %3789 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3791 = tosa.reshape %3788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_756 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3792 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3791, %3790 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_756 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3793 = tosa.reshape %3792 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3794 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3795 = tosa.transpose %arg334, %3794 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3796 = tosa.reshape %3788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_757 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3797 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3796, %3795 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_757 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3798 = tosa.reshape %3797 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3799 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3800 = tosa.transpose %arg335, %3799 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3801 = tosa.reshape %3788 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_758 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3802 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3801, %3800 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_758 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3803 = tosa.reshape %3802 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3804 = tosa.reshape %3793 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3805 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3806 = tosa.transpose %3804, %3805 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3807 = tosa.reshape %3798 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3808 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3809 = tosa.transpose %3807, %3808 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3810 = tosa.reshape %3803 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3811 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3812 = tosa.transpose %3810, %3811 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_759 = tensor.extract_slice %arg336[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_760 = tensor.extract_slice %extracted_slice_759[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_761 = tensor.extract_slice %extracted_slice_760[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_762 = tensor.extract_slice %arg337[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_763 = tensor.extract_slice %extracted_slice_762[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_764 = tensor.extract_slice %extracted_slice_763[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3813 = tensor.empty() : tensor<1x40x128xf32>
    %3814 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_761 : tensor<1x1x40x128xf32>) outs(%3813 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3815 = tensor.empty() : tensor<40x128xf32>
    %3816 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3814 : tensor<1x40x128xf32>) outs(%3815 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3817 = tensor.empty() : tensor<1x40x128xf32>
    %3818 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_764 : tensor<1x1x40x128xf32>) outs(%3817 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3819 = tensor.empty() : tensor<40x128xf32>
    %3820 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3818 : tensor<1x40x128xf32>) outs(%3819 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3821 = tensor.empty() : tensor<1x40x128xf32>
    %3822 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3821 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3816[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3823 = tosa.reshape %3822 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3824 = tensor.empty() : tensor<1x40x128xf32>
    %3825 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3824 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3820[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3826 = tosa.reshape %3825 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3827 = tosa.mul %3806, %3823 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_765 = tensor.extract_slice %3806[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_766 = tensor.extract_slice %3806[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3828 = tensor.empty() : tensor<1x32x40x64xf32>
    %3829 = linalg.negf ins(%extracted_slice_766 : tensor<1x32x40x64xf32>) outs(%3828 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3830 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_767 = tensor.insert_slice %3829 into %3830[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_768 = tensor.insert_slice %extracted_slice_765 into %inserted_slice_767[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3831 = tosa.mul %inserted_slice_768, %3826 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3832 = tosa.add %3827, %3831 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3833 = tosa.mul %3809, %3823 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_769 = tensor.extract_slice %3809[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_770 = tensor.extract_slice %3809[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3834 = tensor.empty() : tensor<1x32x40x64xf32>
    %3835 = linalg.negf ins(%extracted_slice_770 : tensor<1x32x40x64xf32>) outs(%3834 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3836 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_771 = tensor.insert_slice %3835 into %3836[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_772 = tensor.insert_slice %extracted_slice_769 into %inserted_slice_771[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3837 = tosa.mul %inserted_slice_772, %3826 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3838 = tosa.add %3833, %3837 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3839 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3840 = tosa.transpose %3838, %3839 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3841 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3842 = tosa.add %3832, %3841 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3843 = tosa.reshape %3842 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3844 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3845 = tosa.add %3840, %3844 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3846 = tosa.reshape %3845 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3847 = tosa.matmul %3843, %3846 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3848 = tosa.reshape %3847 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3849 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3850 = tosa.reciprocal %3849 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3851 = tosa.mul %3848, %3850 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3852 = tosa.add %3851, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3853 = tosa.reduce_max %3852 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3854 = tosa.sub %3852, %3853 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3855 = tosa.exp %3854 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3856 = tosa.reduce_sum %3855 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3857 = tosa.reciprocal %3856 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3858 = tosa.mul %3855, %3857 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3859 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3860 = tosa.add %3858, %3859 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3861 = tosa.reshape %3860 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3862 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3863 = tosa.add %3812, %3862 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3864 = tosa.reshape %3863 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3865 = tosa.matmul %3861, %3864 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3866 = tosa.reshape %3865 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3867 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3868 = tosa.transpose %3866, %3867 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3869 = tosa.identity %3868 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3870 = tosa.reshape %3869 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3871 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3872 = tosa.transpose %arg338, %3871 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3873 = tosa.reshape %3870 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_773 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3874 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3873, %3872 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_773 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3875 = tosa.reshape %3874 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3876 = tosa.add %3779, %3875 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3877 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_774 = arith.constant 2 : i32
    %3878 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3876 : tensor<1x40x4096xf32>) outs(%3877 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_774 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_775 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3879 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3878 : tensor<1x40x4096xf32>) outs(%cst_775 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3880 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3881 = tosa.add %3879, %3880 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3882 = tosa.rsqrt %3881 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3883 = tosa.mul %3876, %3882 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3884 = tosa.reshape %arg339 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3885 = tosa.mul %3884, %3883 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3886 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3887 = tosa.transpose %arg340, %3886 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3888 = tosa.reshape %3885 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_776 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3889 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3888, %3887 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_776 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3890 = tosa.reshape %3889 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3891 = tosa.sigmoid %3890 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3892 = tosa.mul %3890, %3891 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3893 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3894 = tosa.transpose %arg341, %3893 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %3895 = tosa.reshape %3885 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_777 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3896 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3895, %3894 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_777 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %3897 = tosa.reshape %3896 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %3898 = tosa.mul %3892, %3897 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %3899 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3900 = tosa.transpose %arg342, %3899 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %3901 = tosa.reshape %3898 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_778 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3902 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3901, %3900 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_778 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3903 = tosa.reshape %3902 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3904 = tosa.add %3876, %3903 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3905 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_779 = arith.constant 2 : i32
    %3906 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3904 : tensor<1x40x4096xf32>) outs(%3905 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_779 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_780 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3907 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3906 : tensor<1x40x4096xf32>) outs(%cst_780 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %3908 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3909 = tosa.add %3907, %3908 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3910 = tosa.rsqrt %3909 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3911 = tosa.mul %3904, %3910 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3912 = tosa.reshape %arg343 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3913 = tosa.mul %3912, %3911 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %3914 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3915 = tosa.transpose %arg344, %3914 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3916 = tosa.reshape %3913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_781 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3917 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3916, %3915 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_781 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3918 = tosa.reshape %3917 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3919 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3920 = tosa.transpose %arg345, %3919 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3921 = tosa.reshape %3913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_782 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3922 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3921, %3920 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_782 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3923 = tosa.reshape %3922 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3924 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3925 = tosa.transpose %arg346, %3924 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3926 = tosa.reshape %3913 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_783 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3927 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3926, %3925 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_783 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %3928 = tosa.reshape %3927 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %3929 = tosa.reshape %3918 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3930 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3931 = tosa.transpose %3929, %3930 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3932 = tosa.reshape %3923 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3933 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3934 = tosa.transpose %3932, %3933 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %3935 = tosa.reshape %3928 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    %3936 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3937 = tosa.transpose %3935, %3936 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_784 = tensor.extract_slice %arg347[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_785 = tensor.extract_slice %extracted_slice_784[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_786 = tensor.extract_slice %extracted_slice_785[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_787 = tensor.extract_slice %arg348[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_788 = tensor.extract_slice %extracted_slice_787[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_789 = tensor.extract_slice %extracted_slice_788[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3938 = tensor.empty() : tensor<1x40x128xf32>
    %3939 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_786 : tensor<1x1x40x128xf32>) outs(%3938 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3940 = tensor.empty() : tensor<40x128xf32>
    %3941 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3939 : tensor<1x40x128xf32>) outs(%3940 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3942 = tensor.empty() : tensor<1x40x128xf32>
    %3943 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_789 : tensor<1x1x40x128xf32>) outs(%3942 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3944 = tensor.empty() : tensor<40x128xf32>
    %3945 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel"]} ins(%3943 : tensor<1x40x128xf32>) outs(%3944 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3946 = tensor.empty() : tensor<1x40x128xf32>
    %3947 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3946 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3941[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3948 = tosa.reshape %3947 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3949 = tensor.empty() : tensor<1x40x128xf32>
    %3950 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3949 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4044 = arith.index_cast %in : i64 to index
      %4045 = linalg.index 2 : index
      %extracted = tensor.extract %3945[%4044, %4045] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %3951 = tosa.reshape %3950 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3952 = tosa.mul %3931, %3948 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_790 = tensor.extract_slice %3931[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_791 = tensor.extract_slice %3931[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3953 = tensor.empty() : tensor<1x32x40x64xf32>
    %3954 = linalg.negf ins(%extracted_slice_791 : tensor<1x32x40x64xf32>) outs(%3953 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3955 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_792 = tensor.insert_slice %3954 into %3955[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_793 = tensor.insert_slice %extracted_slice_790 into %inserted_slice_792[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3956 = tosa.mul %inserted_slice_793, %3951 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3957 = tosa.add %3952, %3956 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3958 = tosa.mul %3934, %3948 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %extracted_slice_794 = tensor.extract_slice %3934[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_795 = tensor.extract_slice %3934[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3959 = tensor.empty() : tensor<1x32x40x64xf32>
    %3960 = linalg.negf ins(%extracted_slice_795 : tensor<1x32x40x64xf32>) outs(%3959 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3961 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_796 = tensor.insert_slice %3960 into %3961[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_797 = tensor.insert_slice %extracted_slice_794 into %inserted_slice_796[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %3962 = tosa.mul %inserted_slice_797, %3951 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3963 = tosa.add %3958, %3962 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3964 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3965 = tosa.transpose %3963, %3964 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    %3966 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3967 = tosa.add %3957, %3966 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3968 = tosa.reshape %3967 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3969 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    %3970 = tosa.add %3965, %3969 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    %3971 = tosa.reshape %3970 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    %3972 = tosa.matmul %3968, %3971 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    %3973 = tosa.reshape %3972 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3974 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3975 = tosa.reciprocal %3974 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3976 = tosa.mul %3973, %3975 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3977 = tosa.add %3976, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3978 = tosa.reduce_max %3977 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3979 = tosa.sub %3977, %3978 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3980 = tosa.exp %3979 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3981 = tosa.reduce_sum %3980 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    %3982 = tosa.reciprocal %3981 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    %3983 = tosa.mul %3980, %3982 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    %3984 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    %3985 = tosa.add %3983, %3984 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %3986 = tosa.reshape %3985 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %3987 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    %3988 = tosa.add %3937, %3987 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3989 = tosa.reshape %3988 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3990 = tosa.matmul %3986, %3989 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %3991 = tosa.reshape %3990 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %3992 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3993 = tosa.transpose %3991, %3992 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %3994 = tosa.identity %3993 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    %3995 = tosa.reshape %3994 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    %3996 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3997 = tosa.transpose %arg349, %3996 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %3998 = tosa.reshape %3995 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_798 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3999 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3998, %3997 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_798 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %4000 = tosa.reshape %3999 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %4001 = tosa.add %3904, %4000 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %4002 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_799 = arith.constant 2 : i32
    %4003 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4001 : tensor<1x40x4096xf32>) outs(%4002 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_799 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_800 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %4004 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4003 : tensor<1x40x4096xf32>) outs(%cst_800 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %4005 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4006 = tosa.add %4004, %4005 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4007 = tosa.rsqrt %4006 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4008 = tosa.mul %4001, %4007 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4009 = tosa.reshape %arg350 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4010 = tosa.mul %4009, %4008 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %4011 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4012 = tosa.transpose %arg351, %4011 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %4013 = tosa.reshape %4010 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_801 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %4014 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4013, %4012 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_801 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %4015 = tosa.reshape %4014 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %4016 = tosa.sigmoid %4015 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %4017 = tosa.mul %4015, %4016 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %4018 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4019 = tosa.transpose %arg352, %4018 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    %4020 = tosa.reshape %4010 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_802 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %4021 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4020, %4019 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_802 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    %4022 = tosa.reshape %4021 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    %4023 = tosa.mul %4017, %4022 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    %4024 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4025 = tosa.transpose %arg353, %4024 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    %4026 = tosa.reshape %4023 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    %cst_803 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4027 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4026, %4025 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_803 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    %4028 = tosa.reshape %4027 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    %4029 = tosa.add %4001, %4028 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %4030 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_804 = arith.constant 2 : i32
    %4031 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4029 : tensor<1x40x4096xf32>) outs(%4030 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4044 = math.fpowi %in, %c2_i32_804 : f32, i32
      linalg.yield %4044 : f32
    } -> tensor<1x40x4096xf32>
    %cst_805 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %4032 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4031 : tensor<1x40x4096xf32>) outs(%cst_805 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_807 = arith.constant 4.096000e+03 : f32
      %4044 = arith.divf %in, %cst_807 : f32
      %4045 = arith.addf %4044, %out : f32
      linalg.yield %4045 : f32
    } -> tensor<1x40x1xf32>
    %4033 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4034 = tosa.add %4032, %4033 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4035 = tosa.rsqrt %4034 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4036 = tosa.mul %4029, %4035 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4037 = tosa.reshape %arg354 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4038 = tosa.mul %4037, %4036 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    %4039 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4040 = tosa.transpose %arg355, %4039 : (tensor<32000x4096xf32>, tensor<2xi32>) -> tensor<4096x32000xf32>
    %4041 = tosa.reshape %4038 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    %cst_806 = arith.constant dense<0.000000e+00> : tensor<40x32000xf32>
    %4042 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4041, %4040 : tensor<40x4096xf32>, tensor<4096x32000xf32>) outs(%cst_806 : tensor<40x32000xf32>) -> tensor<40x32000xf32>
    %4043 = tosa.reshape %4042 {new_shape = array<i64: 1, 40, 32000>} : (tensor<40x32000xf32>) -> tensor<1x40x32000xf32>
    return %4038, %4043 : tensor<1x40x4096xf32>, tensor<1x40x32000xf32>
  }
}

