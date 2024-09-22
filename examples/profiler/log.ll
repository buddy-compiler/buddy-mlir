#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
#map7 = affine_map<(d0, d1) -> (0, d0, d1)>
module {
  func.func private @timingStart()
  func.func private @timingEnd()
  func.func @subgraph0(%arg0: tensor<32000x4096xf32>, %arg1: tensor<1x40xi64>, %arg2: tensor<4096xf32>, %arg3: tensor<4096x4096xf32>, %arg4: tensor<4096x4096xf32>, %arg5: tensor<4096x4096xf32>, %arg6: tensor<1x1x2048x128xf32>, %arg7: tensor<1x1x2048x128xf32>, %arg8: tensor<4096x4096xf32>, %arg9: tensor<4096xf32>, %arg10: tensor<11008x4096xf32>, %arg11: tensor<11008x4096xf32>, %arg12: tensor<4096x11008xf32>, %arg13: tensor<4096xf32>, %arg14: tensor<4096x4096xf32>, %arg15: tensor<4096x4096xf32>, %arg16: tensor<4096x4096xf32>, %arg17: tensor<1x1x2048x128xf32>, %arg18: tensor<1x1x2048x128xf32>, %arg19: tensor<4096x4096xf32>, %arg20: tensor<4096xf32>, %arg21: tensor<11008x4096xf32>, %arg22: tensor<11008x4096xf32>, %arg23: tensor<4096x11008xf32>, %arg24: tensor<4096xf32>, %arg25: tensor<4096x4096xf32>, %arg26: tensor<4096x4096xf32>, %arg27: tensor<4096x4096xf32>, %arg28: tensor<1x1x2048x128xf32>, %arg29: tensor<1x1x2048x128xf32>, %arg30: tensor<4096x4096xf32>, %arg31: tensor<4096xf32>, %arg32: tensor<11008x4096xf32>, %arg33: tensor<11008x4096xf32>, %arg34: tensor<4096x11008xf32>, %arg35: tensor<4096xf32>, %arg36: tensor<4096x4096xf32>, %arg37: tensor<4096x4096xf32>, %arg38: tensor<4096x4096xf32>, %arg39: tensor<1x1x2048x128xf32>, %arg40: tensor<1x1x2048x128xf32>, %arg41: tensor<4096x4096xf32>, %arg42: tensor<4096xf32>, %arg43: tensor<11008x4096xf32>, %arg44: tensor<11008x4096xf32>, %arg45: tensor<4096x11008xf32>, %arg46: tensor<4096xf32>, %arg47: tensor<4096x4096xf32>, %arg48: tensor<4096x4096xf32>, %arg49: tensor<4096x4096xf32>, %arg50: tensor<1x1x2048x128xf32>, %arg51: tensor<1x1x2048x128xf32>, %arg52: tensor<4096x4096xf32>, %arg53: tensor<4096xf32>, %arg54: tensor<11008x4096xf32>, %arg55: tensor<11008x4096xf32>, %arg56: tensor<4096x11008xf32>, %arg57: tensor<4096xf32>, %arg58: tensor<4096x4096xf32>, %arg59: tensor<4096x4096xf32>, %arg60: tensor<4096x4096xf32>, %arg61: tensor<1x1x2048x128xf32>, %arg62: tensor<1x1x2048x128xf32>, %arg63: tensor<4096x4096xf32>, %arg64: tensor<4096xf32>, %arg65: tensor<11008x4096xf32>, %arg66: tensor<11008x4096xf32>, %arg67: tensor<4096x11008xf32>, %arg68: tensor<4096xf32>, %arg69: tensor<4096x4096xf32>, %arg70: tensor<4096x4096xf32>, %arg71: tensor<4096x4096xf32>, %arg72: tensor<1x1x2048x128xf32>, %arg73: tensor<1x1x2048x128xf32>, %arg74: tensor<4096x4096xf32>, %arg75: tensor<4096xf32>, %arg76: tensor<11008x4096xf32>, %arg77: tensor<11008x4096xf32>, %arg78: tensor<4096x11008xf32>, %arg79: tensor<4096xf32>, %arg80: tensor<4096x4096xf32>, %arg81: tensor<4096x4096xf32>, %arg82: tensor<4096x4096xf32>, %arg83: tensor<1x1x2048x128xf32>, %arg84: tensor<1x1x2048x128xf32>, %arg85: tensor<4096x4096xf32>, %arg86: tensor<4096xf32>, %arg87: tensor<11008x4096xf32>, %arg88: tensor<11008x4096xf32>, %arg89: tensor<4096x11008xf32>, %arg90: tensor<4096xf32>, %arg91: tensor<4096x4096xf32>, %arg92: tensor<4096x4096xf32>, %arg93: tensor<4096x4096xf32>, %arg94: tensor<1x1x2048x128xf32>, %arg95: tensor<1x1x2048x128xf32>, %arg96: tensor<4096x4096xf32>, %arg97: tensor<4096xf32>, %arg98: tensor<11008x4096xf32>, %arg99: tensor<11008x4096xf32>, %arg100: tensor<4096x11008xf32>, %arg101: tensor<4096xf32>, %arg102: tensor<4096x4096xf32>, %arg103: tensor<4096x4096xf32>, %arg104: tensor<4096x4096xf32>, %arg105: tensor<1x1x2048x128xf32>, %arg106: tensor<1x1x2048x128xf32>, %arg107: tensor<4096x4096xf32>, %arg108: tensor<4096xf32>, %arg109: tensor<11008x4096xf32>, %arg110: tensor<11008x4096xf32>, %arg111: tensor<4096x11008xf32>, %arg112: tensor<4096xf32>, %arg113: tensor<4096x4096xf32>, %arg114: tensor<4096x4096xf32>, %arg115: tensor<4096x4096xf32>, %arg116: tensor<1x1x2048x128xf32>, %arg117: tensor<1x1x2048x128xf32>, %arg118: tensor<4096x4096xf32>, %arg119: tensor<4096xf32>, %arg120: tensor<11008x4096xf32>, %arg121: tensor<11008x4096xf32>, %arg122: tensor<4096x11008xf32>, %arg123: tensor<4096xf32>, %arg124: tensor<4096x4096xf32>, %arg125: tensor<4096x4096xf32>, %arg126: tensor<4096x4096xf32>, %arg127: tensor<1x1x2048x128xf32>, %arg128: tensor<1x1x2048x128xf32>, %arg129: tensor<4096x4096xf32>, %arg130: tensor<4096xf32>, %arg131: tensor<11008x4096xf32>, %arg132: tensor<11008x4096xf32>, %arg133: tensor<4096x11008xf32>, %arg134: tensor<4096xf32>, %arg135: tensor<4096x4096xf32>, %arg136: tensor<4096x4096xf32>, %arg137: tensor<4096x4096xf32>, %arg138: tensor<1x1x2048x128xf32>, %arg139: tensor<1x1x2048x128xf32>, %arg140: tensor<4096x4096xf32>, %arg141: tensor<4096xf32>, %arg142: tensor<11008x4096xf32>, %arg143: tensor<11008x4096xf32>, %arg144: tensor<4096x11008xf32>, %arg145: tensor<4096xf32>, %arg146: tensor<4096x4096xf32>, %arg147: tensor<4096x4096xf32>, %arg148: tensor<4096x4096xf32>, %arg149: tensor<1x1x2048x128xf32>, %arg150: tensor<1x1x2048x128xf32>, %arg151: tensor<4096x4096xf32>, %arg152: tensor<4096xf32>, %arg153: tensor<11008x4096xf32>, %arg154: tensor<11008x4096xf32>, %arg155: tensor<4096x11008xf32>, %arg156: tensor<4096xf32>, %arg157: tensor<4096x4096xf32>, %arg158: tensor<4096x4096xf32>, %arg159: tensor<4096x4096xf32>, %arg160: tensor<1x1x2048x128xf32>, %arg161: tensor<1x1x2048x128xf32>, %arg162: tensor<4096x4096xf32>, %arg163: tensor<4096xf32>, %arg164: tensor<11008x4096xf32>, %arg165: tensor<11008x4096xf32>, %arg166: tensor<4096x11008xf32>, %arg167: tensor<4096xf32>, %arg168: tensor<4096x4096xf32>, %arg169: tensor<4096x4096xf32>, %arg170: tensor<4096x4096xf32>, %arg171: tensor<1x1x2048x128xf32>, %arg172: tensor<1x1x2048x128xf32>, %arg173: tensor<4096x4096xf32>, %arg174: tensor<4096xf32>, %arg175: tensor<11008x4096xf32>, %arg176: tensor<11008x4096xf32>, %arg177: tensor<4096x11008xf32>, %arg178: tensor<4096xf32>, %arg179: tensor<4096x4096xf32>, %arg180: tensor<4096x4096xf32>, %arg181: tensor<4096x4096xf32>, %arg182: tensor<1x1x2048x128xf32>, %arg183: tensor<1x1x2048x128xf32>, %arg184: tensor<4096x4096xf32>, %arg185: tensor<4096xf32>, %arg186: tensor<11008x4096xf32>, %arg187: tensor<11008x4096xf32>, %arg188: tensor<4096x11008xf32>, %arg189: tensor<4096xf32>, %arg190: tensor<4096x4096xf32>, %arg191: tensor<4096x4096xf32>, %arg192: tensor<4096x4096xf32>, %arg193: tensor<1x1x2048x128xf32>, %arg194: tensor<1x1x2048x128xf32>, %arg195: tensor<4096x4096xf32>, %arg196: tensor<4096xf32>, %arg197: tensor<11008x4096xf32>, %arg198: tensor<11008x4096xf32>, %arg199: tensor<4096x11008xf32>, %arg200: tensor<4096xf32>, %arg201: tensor<4096x4096xf32>, %arg202: tensor<4096x4096xf32>, %arg203: tensor<4096x4096xf32>, %arg204: tensor<1x1x2048x128xf32>, %arg205: tensor<1x1x2048x128xf32>, %arg206: tensor<4096x4096xf32>, %arg207: tensor<4096xf32>, %arg208: tensor<11008x4096xf32>, %arg209: tensor<11008x4096xf32>, %arg210: tensor<4096x11008xf32>, %arg211: tensor<4096xf32>, %arg212: tensor<4096x4096xf32>, %arg213: tensor<4096x4096xf32>, %arg214: tensor<4096x4096xf32>, %arg215: tensor<1x1x2048x128xf32>, %arg216: tensor<1x1x2048x128xf32>, %arg217: tensor<4096x4096xf32>, %arg218: tensor<4096xf32>, %arg219: tensor<11008x4096xf32>, %arg220: tensor<11008x4096xf32>, %arg221: tensor<4096x11008xf32>, %arg222: tensor<4096xf32>, %arg223: tensor<4096x4096xf32>, %arg224: tensor<4096x4096xf32>, %arg225: tensor<4096x4096xf32>, %arg226: tensor<1x1x2048x128xf32>, %arg227: tensor<1x1x2048x128xf32>, %arg228: tensor<4096x4096xf32>, %arg229: tensor<4096xf32>, %arg230: tensor<11008x4096xf32>, %arg231: tensor<11008x4096xf32>, %arg232: tensor<4096x11008xf32>, %arg233: tensor<4096xf32>, %arg234: tensor<4096x4096xf32>, %arg235: tensor<4096x4096xf32>, %arg236: tensor<4096x4096xf32>, %arg237: tensor<1x1x2048x128xf32>, %arg238: tensor<1x1x2048x128xf32>, %arg239: tensor<4096x4096xf32>, %arg240: tensor<4096xf32>, %arg241: tensor<11008x4096xf32>, %arg242: tensor<11008x4096xf32>, %arg243: tensor<4096x11008xf32>, %arg244: tensor<4096xf32>, %arg245: tensor<4096x4096xf32>, %arg246: tensor<4096x4096xf32>, %arg247: tensor<4096x4096xf32>, %arg248: tensor<1x1x2048x128xf32>, %arg249: tensor<1x1x2048x128xf32>, %arg250: tensor<4096x4096xf32>, %arg251: tensor<4096xf32>, %arg252: tensor<11008x4096xf32>, %arg253: tensor<11008x4096xf32>, %arg254: tensor<4096x11008xf32>, %arg255: tensor<4096xf32>, %arg256: tensor<4096x4096xf32>, %arg257: tensor<4096x4096xf32>, %arg258: tensor<4096x4096xf32>, %arg259: tensor<1x1x2048x128xf32>, %arg260: tensor<1x1x2048x128xf32>, %arg261: tensor<4096x4096xf32>, %arg262: tensor<4096xf32>, %arg263: tensor<11008x4096xf32>, %arg264: tensor<11008x4096xf32>, %arg265: tensor<4096x11008xf32>, %arg266: tensor<4096xf32>, %arg267: tensor<4096x4096xf32>, %arg268: tensor<4096x4096xf32>, %arg269: tensor<4096x4096xf32>, %arg270: tensor<1x1x2048x128xf32>, %arg271: tensor<1x1x2048x128xf32>, %arg272: tensor<4096x4096xf32>, %arg273: tensor<4096xf32>, %arg274: tensor<11008x4096xf32>, %arg275: tensor<11008x4096xf32>, %arg276: tensor<4096x11008xf32>, %arg277: tensor<4096xf32>, %arg278: tensor<4096x4096xf32>, %arg279: tensor<4096x4096xf32>, %arg280: tensor<4096x4096xf32>, %arg281: tensor<1x1x2048x128xf32>, %arg282: tensor<1x1x2048x128xf32>, %arg283: tensor<4096x4096xf32>, %arg284: tensor<4096xf32>, %arg285: tensor<11008x4096xf32>, %arg286: tensor<11008x4096xf32>, %arg287: tensor<4096x11008xf32>, %arg288: tensor<4096xf32>, %arg289: tensor<4096x4096xf32>, %arg290: tensor<4096x4096xf32>, %arg291: tensor<4096x4096xf32>, %arg292: tensor<1x1x2048x128xf32>, %arg293: tensor<1x1x2048x128xf32>, %arg294: tensor<4096x4096xf32>, %arg295: tensor<4096xf32>, %arg296: tensor<11008x4096xf32>, %arg297: tensor<11008x4096xf32>, %arg298: tensor<4096x11008xf32>, %arg299: tensor<4096xf32>, %arg300: tensor<4096x4096xf32>, %arg301: tensor<4096x4096xf32>, %arg302: tensor<4096x4096xf32>, %arg303: tensor<1x1x2048x128xf32>, %arg304: tensor<1x1x2048x128xf32>, %arg305: tensor<4096x4096xf32>, %arg306: tensor<4096xf32>, %arg307: tensor<11008x4096xf32>, %arg308: tensor<11008x4096xf32>, %arg309: tensor<4096x11008xf32>, %arg310: tensor<4096xf32>, %arg311: tensor<4096x4096xf32>, %arg312: tensor<4096x4096xf32>, %arg313: tensor<4096x4096xf32>, %arg314: tensor<1x1x2048x128xf32>, %arg315: tensor<1x1x2048x128xf32>, %arg316: tensor<4096x4096xf32>, %arg317: tensor<4096xf32>, %arg318: tensor<11008x4096xf32>, %arg319: tensor<11008x4096xf32>, %arg320: tensor<4096x11008xf32>, %arg321: tensor<4096xf32>, %arg322: tensor<4096x4096xf32>, %arg323: tensor<4096x4096xf32>, %arg324: tensor<4096x4096xf32>, %arg325: tensor<1x1x2048x128xf32>, %arg326: tensor<1x1x2048x128xf32>, %arg327: tensor<4096x4096xf32>, %arg328: tensor<4096xf32>, %arg329: tensor<11008x4096xf32>, %arg330: tensor<11008x4096xf32>, %arg331: tensor<4096x11008xf32>, %arg332: tensor<4096xf32>, %arg333: tensor<4096x4096xf32>, %arg334: tensor<4096x4096xf32>, %arg335: tensor<4096x4096xf32>, %arg336: tensor<1x1x2048x128xf32>, %arg337: tensor<1x1x2048x128xf32>, %arg338: tensor<4096x4096xf32>, %arg339: tensor<4096xf32>, %arg340: tensor<11008x4096xf32>, %arg341: tensor<11008x4096xf32>, %arg342: tensor<4096x11008xf32>, %arg343: tensor<4096xf32>, %arg344: tensor<4096x4096xf32>, %arg345: tensor<4096x4096xf32>, %arg346: tensor<4096x4096xf32>, %arg347: tensor<1x1x2048x128xf32>, %arg348: tensor<1x1x2048x128xf32>, %arg349: tensor<4096x4096xf32>, %arg350: tensor<4096xf32>, %arg351: tensor<11008x4096xf32>, %arg352: tensor<11008x4096xf32>, %arg353: tensor<4096x11008xf32>, %arg354: tensor<4096xf32>, %arg355: tensor<32000x4096xf32>) -> (tensor<1x40x4096xf32>, tensor<1x40x32000xf32>) {
    call @timingStart() : () -> ()
    %0 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1 = tosa.reshape %0 {new_shape = array<i64: 1, 40>} : (tensor<40xi64>) -> tensor<1x40xi64>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2 = tosa.reshape %1 {new_shape = array<i64: 1, 40>} : (tensor<1x40xi64>) -> tensor<1x40xi64>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3 = tosa.cast %arg1 : (tensor<1x40xi64>) -> tensor<1x40xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4 = tosa.reshape %arg0 {new_shape = array<i64: 1, 32000, 4096>} : (tensor<32000x4096xf32>) -> tensor<1x32000x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %5 = tosa.gather %4, %3 : (tensor<1x32000x4096xf32>, tensor<1x40xi32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %6 = tosa.reshape %5 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %cst = arith.constant dense<true> : tensor<1x40xi1>
    %cst_0 = arith.constant dense<-3.40282347E+38> : tensor<40x40xf32>
    call @timingStart() : () -> ()
    %7 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %8 = "tosa.const"() <{value = dense<1> : tensor<40xi64>}> : () -> tensor<40xi64>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %9 = tosa.add %7, %8 : (tensor<40xi64>, tensor<40xi64>) -> tensor<40xi64>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %10 = tosa.reshape %9 {new_shape = array<i64: 40, 1>} : (tensor<40xi64>) -> tensor<40x1xi64>
    call @timingEnd() : () -> ()
    %11 = tensor.empty() : tensor<40x40xi1>
    %12 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7, %10 : tensor<40xi64>, tensor<40x1xi64>) outs(%11 : tensor<40x40xi1>) {
    ^bb0(%in: i64, %in_742: i64, %out: i1):
      %4239 = arith.cmpi slt, %in, %in_742 : i64
      linalg.yield %4239 : i1
    } -> tensor<40x40xi1>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %13 = tensor.empty() : tensor<40x40xf32>
    %14 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%12, %cst_0 : tensor<40x40xi1>, tensor<40x40xf32>) outs(%13 : tensor<40x40xf32>) {
    ^bb0(%in: i1, %in_742: f32, %out: f32):
      %4239 = arith.select %in, %cst_1, %in_742 : f32
      linalg.yield %4239 : f32
    } -> tensor<40x40xf32>
    %extracted_slice = tensor.extract_slice %cst[0, 0] [1, 40] [1, 1] : tensor<1x40xi1> to tensor<1x40xi1>
    call @timingStart() : () -> ()
    %15 = tosa.reshape %extracted_slice {new_shape = array<i64: 1, 1, 40>} : (tensor<1x40xi1>) -> tensor<1x1x40xi1>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %16 = tosa.reshape %15 {new_shape = array<i64: 1, 1, 1, 40>} : (tensor<1x1x40xi1>) -> tensor<1x1x1x40xi1>
    call @timingEnd() : () -> ()
    %extracted_slice_2 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 1, 40] [1, 1, 1, 1] : tensor<1x1x1x40xi1> to tensor<1x1x1x40xi1>
    call @timingStart() : () -> ()
    %17 = "tosa.const"() <{value = dense<false> : tensor<1x1x40x40xi1>}> : () -> tensor<1x1x40x40xi1>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %18 = tosa.add %extracted_slice_2, %17 : (tensor<1x1x1x40xi1>, tensor<1x1x40x40xi1>) -> tensor<1x1x40x40xi1>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %19 = tosa.cast %18 : (tensor<1x1x40x40xi1>) -> tensor<1x1x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %20 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x40x40xf32>}> : () -> tensor<1x1x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %21 = tosa.sub %20, %19 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %22 = tosa.cast %21 : (tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xi1>
    call @timingEnd() : () -> ()
    %cst_3 = arith.constant -3.40282347E+38 : f32
    %23 = tensor.empty() : tensor<1x1x40x40xf32>
    %24 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %21 : tensor<1x1x40x40xi1>, tensor<1x1x40x40xf32>) outs(%23 : tensor<1x1x40x40xf32>) {
    ^bb0(%in: i1, %in_742: f32, %out: f32):
      %4239 = arith.select %in, %cst_3, %in_742 : f32
      linalg.yield %4239 : f32
    } -> tensor<1x1x40x40xf32>
    call @timingStart() : () -> ()
    %25 = tosa.reshape %14 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %26 = tosa.reshape %25 {new_shape = array<i64: 1, 1, 40, 40>} : (tensor<1x40x40xf32>) -> tensor<1x1x40x40xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_4 = tensor.extract_slice %26[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_5 = tensor.extract_slice %extracted_slice_4[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    call @timingStart() : () -> ()
    %27 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x40xf32>}> : () -> tensor<1x1x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %28 = tosa.add %extracted_slice_5, %27 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %29 = tosa.add %24, %28 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
    call @timingEnd() : () -> ()
    %30 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32 = arith.constant 2 : i32
    %31 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<1x40x4096xf32>) outs(%30 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %32 = tosa.reduce_sum %31 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %33 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %34 = tosa.reciprocal %33 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %35 = tosa.mul %34, %32 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %36 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %37 = tosa.add %35, %36 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %38 = tosa.rsqrt %37 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %39 = tosa.mul %6, %38 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %40 = tosa.reshape %arg2 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %41 = tosa.mul %40, %39 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %42 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %43 = tosa.transpose %arg3, %42 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %44 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %45 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%44, %43 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_6 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %46 = tosa.reshape %45 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %47 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %48 = tosa.transpose %arg4, %47 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %49 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %50 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%49, %48 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_7 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %51 = tosa.reshape %50 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %52 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %53 = tosa.transpose %arg5, %52 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %54 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %55 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%54, %53 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_8 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %56 = tosa.reshape %55 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %57 = tosa.reshape %46 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %58 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %59 = tosa.transpose %57, %58 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %60 = tosa.reshape %51 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %61 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %62 = tosa.transpose %60, %61 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %63 = tosa.reshape %56 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %64 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %65 = tosa.transpose %63, %64 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
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
    %74 = tensor.empty() : tensor<1x40x128xf32>
    %75 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%74 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %69[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %76 = tosa.reshape %75 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %77 = tensor.empty() : tensor<1x40x128xf32>
    %78 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%77 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %73[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %79 = tosa.reshape %78 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %80 = tosa.mul %59, %76 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_15 = tensor.extract_slice %59[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_16 = tensor.extract_slice %59[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %81 = tensor.empty() : tensor<1x32x40x64xf32>
    %82 = linalg.negf ins(%extracted_slice_16 : tensor<1x32x40x64xf32>) outs(%81 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %83 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice = tensor.insert_slice %82 into %83[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_17 = tensor.insert_slice %extracted_slice_15 into %inserted_slice[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %84 = tosa.mul %inserted_slice_17, %79 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %85 = tosa.add %80, %84 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %86 = tosa.mul %62, %76 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_18 = tensor.extract_slice %62[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_19 = tensor.extract_slice %62[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %87 = tensor.empty() : tensor<1x32x40x64xf32>
    %88 = linalg.negf ins(%extracted_slice_19 : tensor<1x32x40x64xf32>) outs(%87 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %89 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_20 = tensor.insert_slice %88 into %89[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_21 = tensor.insert_slice %extracted_slice_18 into %inserted_slice_20[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %90 = tosa.mul %inserted_slice_21, %79 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %91 = tosa.add %86, %90 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %92 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %93 = tosa.transpose %91, %92 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %94 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %95 = tosa.add %85, %94 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %96 = tosa.reshape %95 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %97 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %98 = tosa.add %93, %97 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %99 = tosa.reshape %98 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %100 = tosa.matmul %96, %99 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %101 = tosa.reshape %100 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %102 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %103 = tosa.reciprocal %102 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %104 = tosa.mul %101, %103 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %105 = tosa.add %104, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %106 = tosa.reduce_max %105 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %107 = tosa.sub %105, %106 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %108 = tosa.exp %107 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %109 = tosa.reduce_sum %108 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %110 = tosa.reciprocal %109 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %111 = tosa.mul %108, %110 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %112 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %113 = tosa.add %111, %112 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %114 = tosa.reshape %113 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %115 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %116 = tosa.add %65, %115 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %117 = tosa.reshape %116 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %118 = tosa.matmul %114, %117 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %119 = tosa.reshape %118 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %120 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %121 = tosa.transpose %119, %120 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %122 = tosa.identity %121 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %123 = tosa.reshape %122 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %124 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %125 = tosa.transpose %arg8, %124 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %126 = tosa.reshape %123 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_22 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %127 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%126, %125 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_22 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %128 = tosa.reshape %127 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %129 = tosa.add %6, %128 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %130 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_23 = arith.constant 2 : i32
    %131 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129 : tensor<1x40x4096xf32>) outs(%130 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_23 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %132 = tosa.reduce_sum %131 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %133 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %134 = tosa.reciprocal %133 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %135 = tosa.mul %134, %132 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %136 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %137 = tosa.add %135, %136 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %138 = tosa.rsqrt %137 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %139 = tosa.mul %129, %138 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %140 = tosa.reshape %arg9 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %141 = tosa.mul %140, %139 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %142 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %143 = tosa.transpose %arg10, %142 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %144 = tosa.reshape %141 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_24 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %145 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%144, %143 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_24 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %146 = tosa.reshape %145 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %147 = tosa.sigmoid %146 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %148 = tosa.mul %146, %147 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %149 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %150 = tosa.transpose %arg11, %149 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %151 = tosa.reshape %141 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_25 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %152 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%151, %150 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_25 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %153 = tosa.reshape %152 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %154 = tosa.mul %148, %153 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %155 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %156 = tosa.transpose %arg12, %155 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %157 = tosa.reshape %154 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_26 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %158 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%157, %156 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_26 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %159 = tosa.reshape %158 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %160 = tosa.add %129, %159 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %161 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_27 = arith.constant 2 : i32
    %162 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%160 : tensor<1x40x4096xf32>) outs(%161 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_27 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %163 = tosa.reduce_sum %162 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %164 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %165 = tosa.reciprocal %164 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %166 = tosa.mul %165, %163 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %167 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %168 = tosa.add %166, %167 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %169 = tosa.rsqrt %168 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %170 = tosa.mul %160, %169 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %171 = tosa.reshape %arg13 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %172 = tosa.mul %171, %170 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %173 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %174 = tosa.transpose %arg14, %173 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %175 = tosa.reshape %172 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_28 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %176 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%175, %174 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_28 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %177 = tosa.reshape %176 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %178 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %179 = tosa.transpose %arg15, %178 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %180 = tosa.reshape %172 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_29 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %181 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%180, %179 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_29 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %182 = tosa.reshape %181 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %183 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %184 = tosa.transpose %arg16, %183 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %185 = tosa.reshape %172 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_30 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %186 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%185, %184 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_30 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %187 = tosa.reshape %186 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %188 = tosa.reshape %177 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %189 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %190 = tosa.transpose %188, %189 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %191 = tosa.reshape %182 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %192 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %193 = tosa.transpose %191, %192 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %194 = tosa.reshape %187 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %195 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %196 = tosa.transpose %194, %195 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_31 = tensor.extract_slice %arg17[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_32 = tensor.extract_slice %extracted_slice_31[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_33 = tensor.extract_slice %extracted_slice_32[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_34 = tensor.extract_slice %arg18[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_35 = tensor.extract_slice %extracted_slice_34[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_36 = tensor.extract_slice %extracted_slice_35[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %197 = tensor.empty() : tensor<1x40x128xf32>
    %198 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_33 : tensor<1x1x40x128xf32>) outs(%197 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %199 = tensor.empty() : tensor<40x128xf32>
    %200 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%198 : tensor<1x40x128xf32>) outs(%199 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %201 = tensor.empty() : tensor<1x40x128xf32>
    %202 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_36 : tensor<1x1x40x128xf32>) outs(%201 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %203 = tensor.empty() : tensor<40x128xf32>
    %204 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%202 : tensor<1x40x128xf32>) outs(%203 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %205 = tensor.empty() : tensor<1x40x128xf32>
    %206 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%205 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %200[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %207 = tosa.reshape %206 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %208 = tensor.empty() : tensor<1x40x128xf32>
    %209 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%208 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %204[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %210 = tosa.reshape %209 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %211 = tosa.mul %190, %207 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_37 = tensor.extract_slice %190[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_38 = tensor.extract_slice %190[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %212 = tensor.empty() : tensor<1x32x40x64xf32>
    %213 = linalg.negf ins(%extracted_slice_38 : tensor<1x32x40x64xf32>) outs(%212 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %214 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_39 = tensor.insert_slice %213 into %214[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_40 = tensor.insert_slice %extracted_slice_37 into %inserted_slice_39[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %215 = tosa.mul %inserted_slice_40, %210 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %216 = tosa.add %211, %215 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %217 = tosa.mul %193, %207 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_41 = tensor.extract_slice %193[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_42 = tensor.extract_slice %193[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %218 = tensor.empty() : tensor<1x32x40x64xf32>
    %219 = linalg.negf ins(%extracted_slice_42 : tensor<1x32x40x64xf32>) outs(%218 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %220 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_43 = tensor.insert_slice %219 into %220[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_44 = tensor.insert_slice %extracted_slice_41 into %inserted_slice_43[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %221 = tosa.mul %inserted_slice_44, %210 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %222 = tosa.add %217, %221 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %223 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %224 = tosa.transpose %222, %223 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %225 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %226 = tosa.add %216, %225 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %227 = tosa.reshape %226 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %228 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %229 = tosa.add %224, %228 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %230 = tosa.reshape %229 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %231 = tosa.matmul %227, %230 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %232 = tosa.reshape %231 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %233 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %234 = tosa.reciprocal %233 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %235 = tosa.mul %232, %234 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %236 = tosa.add %235, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %237 = tosa.reduce_max %236 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %238 = tosa.sub %236, %237 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %239 = tosa.exp %238 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %240 = tosa.reduce_sum %239 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %241 = tosa.reciprocal %240 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %242 = tosa.mul %239, %241 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %243 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %244 = tosa.add %242, %243 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %245 = tosa.reshape %244 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %246 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %247 = tosa.add %196, %246 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %248 = tosa.reshape %247 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %249 = tosa.matmul %245, %248 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %250 = tosa.reshape %249 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %251 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %252 = tosa.transpose %250, %251 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %253 = tosa.identity %252 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %254 = tosa.reshape %253 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %255 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %256 = tosa.transpose %arg19, %255 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %257 = tosa.reshape %254 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_45 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %258 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%257, %256 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_45 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %259 = tosa.reshape %258 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %260 = tosa.add %160, %259 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %261 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_46 = arith.constant 2 : i32
    %262 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%260 : tensor<1x40x4096xf32>) outs(%261 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_46 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %263 = tosa.reduce_sum %262 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %264 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %265 = tosa.reciprocal %264 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %266 = tosa.mul %265, %263 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %267 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %268 = tosa.add %266, %267 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %269 = tosa.rsqrt %268 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %270 = tosa.mul %260, %269 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %271 = tosa.reshape %arg20 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %272 = tosa.mul %271, %270 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %273 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %274 = tosa.transpose %arg21, %273 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %275 = tosa.reshape %272 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_47 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %276 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%275, %274 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_47 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %277 = tosa.reshape %276 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %278 = tosa.sigmoid %277 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %279 = tosa.mul %277, %278 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %280 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %281 = tosa.transpose %arg22, %280 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %282 = tosa.reshape %272 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_48 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %283 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%282, %281 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_48 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %284 = tosa.reshape %283 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %285 = tosa.mul %279, %284 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %286 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %287 = tosa.transpose %arg23, %286 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %288 = tosa.reshape %285 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_49 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %289 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%288, %287 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_49 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %290 = tosa.reshape %289 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %291 = tosa.add %260, %290 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %292 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_50 = arith.constant 2 : i32
    %293 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%291 : tensor<1x40x4096xf32>) outs(%292 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_50 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %294 = tosa.reduce_sum %293 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %295 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %296 = tosa.reciprocal %295 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %297 = tosa.mul %296, %294 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %298 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %299 = tosa.add %297, %298 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %300 = tosa.rsqrt %299 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %301 = tosa.mul %291, %300 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %302 = tosa.reshape %arg24 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %303 = tosa.mul %302, %301 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %304 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %305 = tosa.transpose %arg25, %304 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %306 = tosa.reshape %303 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_51 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %307 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%306, %305 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_51 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %308 = tosa.reshape %307 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %309 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %310 = tosa.transpose %arg26, %309 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %311 = tosa.reshape %303 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_52 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %312 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%311, %310 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_52 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %313 = tosa.reshape %312 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %314 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %315 = tosa.transpose %arg27, %314 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %316 = tosa.reshape %303 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_53 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %317 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%316, %315 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_53 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %318 = tosa.reshape %317 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %319 = tosa.reshape %308 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %320 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %321 = tosa.transpose %319, %320 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %322 = tosa.reshape %313 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %323 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %324 = tosa.transpose %322, %323 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %325 = tosa.reshape %318 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %326 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %327 = tosa.transpose %325, %326 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_54 = tensor.extract_slice %arg28[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_55 = tensor.extract_slice %extracted_slice_54[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_56 = tensor.extract_slice %extracted_slice_55[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_57 = tensor.extract_slice %arg29[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_58 = tensor.extract_slice %extracted_slice_57[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_59 = tensor.extract_slice %extracted_slice_58[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %328 = tensor.empty() : tensor<1x40x128xf32>
    %329 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_56 : tensor<1x1x40x128xf32>) outs(%328 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %330 = tensor.empty() : tensor<40x128xf32>
    %331 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%329 : tensor<1x40x128xf32>) outs(%330 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %332 = tensor.empty() : tensor<1x40x128xf32>
    %333 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_59 : tensor<1x1x40x128xf32>) outs(%332 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %334 = tensor.empty() : tensor<40x128xf32>
    %335 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%333 : tensor<1x40x128xf32>) outs(%334 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %336 = tensor.empty() : tensor<1x40x128xf32>
    %337 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%336 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %331[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %338 = tosa.reshape %337 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %339 = tensor.empty() : tensor<1x40x128xf32>
    %340 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%339 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %335[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %341 = tosa.reshape %340 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %342 = tosa.mul %321, %338 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_60 = tensor.extract_slice %321[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_61 = tensor.extract_slice %321[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %343 = tensor.empty() : tensor<1x32x40x64xf32>
    %344 = linalg.negf ins(%extracted_slice_61 : tensor<1x32x40x64xf32>) outs(%343 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %345 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_62 = tensor.insert_slice %344 into %345[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_63 = tensor.insert_slice %extracted_slice_60 into %inserted_slice_62[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %346 = tosa.mul %inserted_slice_63, %341 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %347 = tosa.add %342, %346 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %348 = tosa.mul %324, %338 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_64 = tensor.extract_slice %324[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_65 = tensor.extract_slice %324[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %349 = tensor.empty() : tensor<1x32x40x64xf32>
    %350 = linalg.negf ins(%extracted_slice_65 : tensor<1x32x40x64xf32>) outs(%349 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %351 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_66 = tensor.insert_slice %350 into %351[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_67 = tensor.insert_slice %extracted_slice_64 into %inserted_slice_66[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %352 = tosa.mul %inserted_slice_67, %341 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %353 = tosa.add %348, %352 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %354 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %355 = tosa.transpose %353, %354 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %356 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %357 = tosa.add %347, %356 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %358 = tosa.reshape %357 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %359 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %360 = tosa.add %355, %359 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %361 = tosa.reshape %360 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %362 = tosa.matmul %358, %361 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %363 = tosa.reshape %362 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %364 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %365 = tosa.reciprocal %364 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %366 = tosa.mul %363, %365 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %367 = tosa.add %366, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %368 = tosa.reduce_max %367 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %369 = tosa.sub %367, %368 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %370 = tosa.exp %369 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %371 = tosa.reduce_sum %370 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %372 = tosa.reciprocal %371 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %373 = tosa.mul %370, %372 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %374 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %375 = tosa.add %373, %374 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %376 = tosa.reshape %375 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %377 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %378 = tosa.add %327, %377 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %379 = tosa.reshape %378 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %380 = tosa.matmul %376, %379 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %381 = tosa.reshape %380 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %382 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %383 = tosa.transpose %381, %382 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %384 = tosa.identity %383 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %385 = tosa.reshape %384 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %386 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %387 = tosa.transpose %arg30, %386 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %388 = tosa.reshape %385 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_68 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %389 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%388, %387 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_68 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %390 = tosa.reshape %389 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %391 = tosa.add %291, %390 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %392 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_69 = arith.constant 2 : i32
    %393 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%391 : tensor<1x40x4096xf32>) outs(%392 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_69 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %394 = tosa.reduce_sum %393 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %395 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %396 = tosa.reciprocal %395 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %397 = tosa.mul %396, %394 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %398 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %399 = tosa.add %397, %398 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %400 = tosa.rsqrt %399 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %401 = tosa.mul %391, %400 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %402 = tosa.reshape %arg31 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %403 = tosa.mul %402, %401 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %404 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %405 = tosa.transpose %arg32, %404 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %406 = tosa.reshape %403 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_70 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %407 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%406, %405 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_70 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %408 = tosa.reshape %407 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %409 = tosa.sigmoid %408 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %410 = tosa.mul %408, %409 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %411 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %412 = tosa.transpose %arg33, %411 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %413 = tosa.reshape %403 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_71 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %414 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%413, %412 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_71 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %415 = tosa.reshape %414 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %416 = tosa.mul %410, %415 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %417 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %418 = tosa.transpose %arg34, %417 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %419 = tosa.reshape %416 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_72 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %420 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%419, %418 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_72 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %421 = tosa.reshape %420 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %422 = tosa.add %391, %421 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %423 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_73 = arith.constant 2 : i32
    %424 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%422 : tensor<1x40x4096xf32>) outs(%423 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_73 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %425 = tosa.reduce_sum %424 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %426 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %427 = tosa.reciprocal %426 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %428 = tosa.mul %427, %425 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %429 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %430 = tosa.add %428, %429 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %431 = tosa.rsqrt %430 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %432 = tosa.mul %422, %431 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %433 = tosa.reshape %arg35 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %434 = tosa.mul %433, %432 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %435 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %436 = tosa.transpose %arg36, %435 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %437 = tosa.reshape %434 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_74 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %438 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%437, %436 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_74 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %439 = tosa.reshape %438 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %440 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %441 = tosa.transpose %arg37, %440 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %442 = tosa.reshape %434 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_75 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %443 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%442, %441 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_75 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %444 = tosa.reshape %443 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %445 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %446 = tosa.transpose %arg38, %445 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %447 = tosa.reshape %434 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_76 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %448 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%447, %446 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_76 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %449 = tosa.reshape %448 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %450 = tosa.reshape %439 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %451 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %452 = tosa.transpose %450, %451 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %453 = tosa.reshape %444 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %454 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %455 = tosa.transpose %453, %454 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %456 = tosa.reshape %449 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %457 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %458 = tosa.transpose %456, %457 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_77 = tensor.extract_slice %arg39[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_78 = tensor.extract_slice %extracted_slice_77[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_79 = tensor.extract_slice %extracted_slice_78[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_80 = tensor.extract_slice %arg40[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_81 = tensor.extract_slice %extracted_slice_80[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_82 = tensor.extract_slice %extracted_slice_81[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %459 = tensor.empty() : tensor<1x40x128xf32>
    %460 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_79 : tensor<1x1x40x128xf32>) outs(%459 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %461 = tensor.empty() : tensor<40x128xf32>
    %462 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%460 : tensor<1x40x128xf32>) outs(%461 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %463 = tensor.empty() : tensor<1x40x128xf32>
    %464 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_82 : tensor<1x1x40x128xf32>) outs(%463 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %465 = tensor.empty() : tensor<40x128xf32>
    %466 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%464 : tensor<1x40x128xf32>) outs(%465 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %467 = tensor.empty() : tensor<1x40x128xf32>
    %468 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%467 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %462[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %469 = tosa.reshape %468 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %470 = tensor.empty() : tensor<1x40x128xf32>
    %471 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%470 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %466[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %472 = tosa.reshape %471 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %473 = tosa.mul %452, %469 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_83 = tensor.extract_slice %452[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_84 = tensor.extract_slice %452[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %474 = tensor.empty() : tensor<1x32x40x64xf32>
    %475 = linalg.negf ins(%extracted_slice_84 : tensor<1x32x40x64xf32>) outs(%474 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %476 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_85 = tensor.insert_slice %475 into %476[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_86 = tensor.insert_slice %extracted_slice_83 into %inserted_slice_85[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %477 = tosa.mul %inserted_slice_86, %472 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %478 = tosa.add %473, %477 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %479 = tosa.mul %455, %469 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_87 = tensor.extract_slice %455[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_88 = tensor.extract_slice %455[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %480 = tensor.empty() : tensor<1x32x40x64xf32>
    %481 = linalg.negf ins(%extracted_slice_88 : tensor<1x32x40x64xf32>) outs(%480 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %482 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_89 = tensor.insert_slice %481 into %482[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_90 = tensor.insert_slice %extracted_slice_87 into %inserted_slice_89[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %483 = tosa.mul %inserted_slice_90, %472 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %484 = tosa.add %479, %483 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %485 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %486 = tosa.transpose %484, %485 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %487 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %488 = tosa.add %478, %487 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %489 = tosa.reshape %488 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %490 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %491 = tosa.add %486, %490 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %492 = tosa.reshape %491 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %493 = tosa.matmul %489, %492 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %494 = tosa.reshape %493 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %495 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %496 = tosa.reciprocal %495 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %497 = tosa.mul %494, %496 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %498 = tosa.add %497, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %499 = tosa.reduce_max %498 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %500 = tosa.sub %498, %499 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %501 = tosa.exp %500 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %502 = tosa.reduce_sum %501 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %503 = tosa.reciprocal %502 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %504 = tosa.mul %501, %503 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %505 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %506 = tosa.add %504, %505 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %507 = tosa.reshape %506 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %508 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %509 = tosa.add %458, %508 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %510 = tosa.reshape %509 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %511 = tosa.matmul %507, %510 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %512 = tosa.reshape %511 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %513 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %514 = tosa.transpose %512, %513 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %515 = tosa.identity %514 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %516 = tosa.reshape %515 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %517 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %518 = tosa.transpose %arg41, %517 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %519 = tosa.reshape %516 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_91 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %520 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%519, %518 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_91 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %521 = tosa.reshape %520 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %522 = tosa.add %422, %521 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %523 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_92 = arith.constant 2 : i32
    %524 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%522 : tensor<1x40x4096xf32>) outs(%523 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_92 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %525 = tosa.reduce_sum %524 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %526 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %527 = tosa.reciprocal %526 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %528 = tosa.mul %527, %525 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %529 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %530 = tosa.add %528, %529 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %531 = tosa.rsqrt %530 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %532 = tosa.mul %522, %531 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %533 = tosa.reshape %arg42 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %534 = tosa.mul %533, %532 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %535 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %536 = tosa.transpose %arg43, %535 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %537 = tosa.reshape %534 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_93 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %538 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%537, %536 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_93 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %539 = tosa.reshape %538 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %540 = tosa.sigmoid %539 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %541 = tosa.mul %539, %540 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %542 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %543 = tosa.transpose %arg44, %542 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %544 = tosa.reshape %534 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_94 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %545 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%544, %543 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_94 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %546 = tosa.reshape %545 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %547 = tosa.mul %541, %546 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %548 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %549 = tosa.transpose %arg45, %548 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %550 = tosa.reshape %547 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_95 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %551 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%550, %549 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_95 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %552 = tosa.reshape %551 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %553 = tosa.add %522, %552 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %554 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_96 = arith.constant 2 : i32
    %555 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%553 : tensor<1x40x4096xf32>) outs(%554 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_96 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %556 = tosa.reduce_sum %555 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %557 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %558 = tosa.reciprocal %557 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %559 = tosa.mul %558, %556 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %560 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %561 = tosa.add %559, %560 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %562 = tosa.rsqrt %561 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %563 = tosa.mul %553, %562 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %564 = tosa.reshape %arg46 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %565 = tosa.mul %564, %563 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %566 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %567 = tosa.transpose %arg47, %566 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %568 = tosa.reshape %565 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_97 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %569 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%568, %567 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_97 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %570 = tosa.reshape %569 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %571 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %572 = tosa.transpose %arg48, %571 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %573 = tosa.reshape %565 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_98 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %574 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%573, %572 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_98 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %575 = tosa.reshape %574 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %576 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %577 = tosa.transpose %arg49, %576 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %578 = tosa.reshape %565 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_99 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %579 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%578, %577 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_99 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %580 = tosa.reshape %579 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %581 = tosa.reshape %570 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %582 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %583 = tosa.transpose %581, %582 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %584 = tosa.reshape %575 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %585 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %586 = tosa.transpose %584, %585 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %587 = tosa.reshape %580 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %588 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %589 = tosa.transpose %587, %588 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_100 = tensor.extract_slice %arg50[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_101 = tensor.extract_slice %extracted_slice_100[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_102 = tensor.extract_slice %extracted_slice_101[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_103 = tensor.extract_slice %arg51[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_104 = tensor.extract_slice %extracted_slice_103[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_105 = tensor.extract_slice %extracted_slice_104[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %590 = tensor.empty() : tensor<1x40x128xf32>
    %591 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_102 : tensor<1x1x40x128xf32>) outs(%590 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %592 = tensor.empty() : tensor<40x128xf32>
    %593 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%591 : tensor<1x40x128xf32>) outs(%592 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %594 = tensor.empty() : tensor<1x40x128xf32>
    %595 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_105 : tensor<1x1x40x128xf32>) outs(%594 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %596 = tensor.empty() : tensor<40x128xf32>
    %597 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%595 : tensor<1x40x128xf32>) outs(%596 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %598 = tensor.empty() : tensor<1x40x128xf32>
    %599 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%598 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %593[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %600 = tosa.reshape %599 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %601 = tensor.empty() : tensor<1x40x128xf32>
    %602 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%601 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %597[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %603 = tosa.reshape %602 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %604 = tosa.mul %583, %600 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_106 = tensor.extract_slice %583[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_107 = tensor.extract_slice %583[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %605 = tensor.empty() : tensor<1x32x40x64xf32>
    %606 = linalg.negf ins(%extracted_slice_107 : tensor<1x32x40x64xf32>) outs(%605 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %607 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_108 = tensor.insert_slice %606 into %607[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_109 = tensor.insert_slice %extracted_slice_106 into %inserted_slice_108[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %608 = tosa.mul %inserted_slice_109, %603 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %609 = tosa.add %604, %608 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %610 = tosa.mul %586, %600 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_110 = tensor.extract_slice %586[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_111 = tensor.extract_slice %586[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %611 = tensor.empty() : tensor<1x32x40x64xf32>
    %612 = linalg.negf ins(%extracted_slice_111 : tensor<1x32x40x64xf32>) outs(%611 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %613 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_112 = tensor.insert_slice %612 into %613[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_113 = tensor.insert_slice %extracted_slice_110 into %inserted_slice_112[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %614 = tosa.mul %inserted_slice_113, %603 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %615 = tosa.add %610, %614 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %616 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %617 = tosa.transpose %615, %616 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %618 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %619 = tosa.add %609, %618 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %620 = tosa.reshape %619 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %621 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %622 = tosa.add %617, %621 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %623 = tosa.reshape %622 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %624 = tosa.matmul %620, %623 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %625 = tosa.reshape %624 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %626 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %627 = tosa.reciprocal %626 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %628 = tosa.mul %625, %627 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %629 = tosa.add %628, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %630 = tosa.reduce_max %629 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %631 = tosa.sub %629, %630 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %632 = tosa.exp %631 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %633 = tosa.reduce_sum %632 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %634 = tosa.reciprocal %633 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %635 = tosa.mul %632, %634 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %636 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %637 = tosa.add %635, %636 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %638 = tosa.reshape %637 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %639 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %640 = tosa.add %589, %639 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %641 = tosa.reshape %640 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %642 = tosa.matmul %638, %641 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %643 = tosa.reshape %642 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %644 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %645 = tosa.transpose %643, %644 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %646 = tosa.identity %645 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %647 = tosa.reshape %646 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %648 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %649 = tosa.transpose %arg52, %648 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %650 = tosa.reshape %647 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_114 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %651 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%650, %649 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_114 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %652 = tosa.reshape %651 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %653 = tosa.add %553, %652 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %654 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_115 = arith.constant 2 : i32
    %655 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%653 : tensor<1x40x4096xf32>) outs(%654 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_115 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %656 = tosa.reduce_sum %655 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %657 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %658 = tosa.reciprocal %657 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %659 = tosa.mul %658, %656 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %660 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %661 = tosa.add %659, %660 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %662 = tosa.rsqrt %661 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %663 = tosa.mul %653, %662 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %664 = tosa.reshape %arg53 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %665 = tosa.mul %664, %663 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %666 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %667 = tosa.transpose %arg54, %666 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %668 = tosa.reshape %665 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_116 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %669 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%668, %667 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_116 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %670 = tosa.reshape %669 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %671 = tosa.sigmoid %670 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %672 = tosa.mul %670, %671 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %673 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %674 = tosa.transpose %arg55, %673 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %675 = tosa.reshape %665 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_117 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %676 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%675, %674 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_117 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %677 = tosa.reshape %676 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %678 = tosa.mul %672, %677 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %679 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %680 = tosa.transpose %arg56, %679 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %681 = tosa.reshape %678 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_118 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %682 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%681, %680 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_118 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %683 = tosa.reshape %682 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %684 = tosa.add %653, %683 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %685 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_119 = arith.constant 2 : i32
    %686 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%684 : tensor<1x40x4096xf32>) outs(%685 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_119 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %687 = tosa.reduce_sum %686 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %688 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %689 = tosa.reciprocal %688 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %690 = tosa.mul %689, %687 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %691 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %692 = tosa.add %690, %691 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %693 = tosa.rsqrt %692 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %694 = tosa.mul %684, %693 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %695 = tosa.reshape %arg57 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %696 = tosa.mul %695, %694 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %697 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %698 = tosa.transpose %arg58, %697 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %699 = tosa.reshape %696 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_120 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %700 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%699, %698 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_120 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %701 = tosa.reshape %700 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %702 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %703 = tosa.transpose %arg59, %702 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %704 = tosa.reshape %696 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_121 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %705 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%704, %703 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_121 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %706 = tosa.reshape %705 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %707 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %708 = tosa.transpose %arg60, %707 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %709 = tosa.reshape %696 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_122 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %710 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%709, %708 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_122 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %711 = tosa.reshape %710 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %712 = tosa.reshape %701 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %713 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %714 = tosa.transpose %712, %713 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %715 = tosa.reshape %706 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %716 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %717 = tosa.transpose %715, %716 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %718 = tosa.reshape %711 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %719 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %720 = tosa.transpose %718, %719 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_123 = tensor.extract_slice %arg61[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_124 = tensor.extract_slice %extracted_slice_123[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_125 = tensor.extract_slice %extracted_slice_124[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_126 = tensor.extract_slice %arg62[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_127 = tensor.extract_slice %extracted_slice_126[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_128 = tensor.extract_slice %extracted_slice_127[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %721 = tensor.empty() : tensor<1x40x128xf32>
    %722 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_125 : tensor<1x1x40x128xf32>) outs(%721 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %723 = tensor.empty() : tensor<40x128xf32>
    %724 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%722 : tensor<1x40x128xf32>) outs(%723 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %725 = tensor.empty() : tensor<1x40x128xf32>
    %726 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_128 : tensor<1x1x40x128xf32>) outs(%725 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %727 = tensor.empty() : tensor<40x128xf32>
    %728 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%726 : tensor<1x40x128xf32>) outs(%727 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %729 = tensor.empty() : tensor<1x40x128xf32>
    %730 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%729 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %724[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %731 = tosa.reshape %730 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %732 = tensor.empty() : tensor<1x40x128xf32>
    %733 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%732 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %728[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %734 = tosa.reshape %733 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %735 = tosa.mul %714, %731 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_129 = tensor.extract_slice %714[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_130 = tensor.extract_slice %714[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %736 = tensor.empty() : tensor<1x32x40x64xf32>
    %737 = linalg.negf ins(%extracted_slice_130 : tensor<1x32x40x64xf32>) outs(%736 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %738 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_131 = tensor.insert_slice %737 into %738[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_132 = tensor.insert_slice %extracted_slice_129 into %inserted_slice_131[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %739 = tosa.mul %inserted_slice_132, %734 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %740 = tosa.add %735, %739 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %741 = tosa.mul %717, %731 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_133 = tensor.extract_slice %717[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_134 = tensor.extract_slice %717[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %742 = tensor.empty() : tensor<1x32x40x64xf32>
    %743 = linalg.negf ins(%extracted_slice_134 : tensor<1x32x40x64xf32>) outs(%742 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %744 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_135 = tensor.insert_slice %743 into %744[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_136 = tensor.insert_slice %extracted_slice_133 into %inserted_slice_135[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %745 = tosa.mul %inserted_slice_136, %734 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %746 = tosa.add %741, %745 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %747 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %748 = tosa.transpose %746, %747 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %749 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %750 = tosa.add %740, %749 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %751 = tosa.reshape %750 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %752 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %753 = tosa.add %748, %752 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %754 = tosa.reshape %753 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %755 = tosa.matmul %751, %754 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %756 = tosa.reshape %755 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %757 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %758 = tosa.reciprocal %757 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %759 = tosa.mul %756, %758 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %760 = tosa.add %759, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %761 = tosa.reduce_max %760 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %762 = tosa.sub %760, %761 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %763 = tosa.exp %762 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %764 = tosa.reduce_sum %763 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %765 = tosa.reciprocal %764 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %766 = tosa.mul %763, %765 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %767 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %768 = tosa.add %766, %767 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %769 = tosa.reshape %768 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %770 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %771 = tosa.add %720, %770 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %772 = tosa.reshape %771 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %773 = tosa.matmul %769, %772 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %774 = tosa.reshape %773 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %775 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %776 = tosa.transpose %774, %775 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %777 = tosa.identity %776 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %778 = tosa.reshape %777 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %779 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %780 = tosa.transpose %arg63, %779 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %781 = tosa.reshape %778 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_137 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %782 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%781, %780 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_137 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %783 = tosa.reshape %782 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %784 = tosa.add %684, %783 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %785 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_138 = arith.constant 2 : i32
    %786 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%784 : tensor<1x40x4096xf32>) outs(%785 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_138 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %787 = tosa.reduce_sum %786 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %788 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %789 = tosa.reciprocal %788 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %790 = tosa.mul %789, %787 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %791 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %792 = tosa.add %790, %791 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %793 = tosa.rsqrt %792 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %794 = tosa.mul %784, %793 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %795 = tosa.reshape %arg64 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %796 = tosa.mul %795, %794 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %797 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %798 = tosa.transpose %arg65, %797 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %799 = tosa.reshape %796 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_139 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %800 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%799, %798 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_139 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %801 = tosa.reshape %800 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %802 = tosa.sigmoid %801 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %803 = tosa.mul %801, %802 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %804 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %805 = tosa.transpose %arg66, %804 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %806 = tosa.reshape %796 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_140 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %807 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%806, %805 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_140 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %808 = tosa.reshape %807 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %809 = tosa.mul %803, %808 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %810 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %811 = tosa.transpose %arg67, %810 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %812 = tosa.reshape %809 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_141 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %813 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%812, %811 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_141 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %814 = tosa.reshape %813 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %815 = tosa.add %784, %814 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %816 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_142 = arith.constant 2 : i32
    %817 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%815 : tensor<1x40x4096xf32>) outs(%816 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_142 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %818 = tosa.reduce_sum %817 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %819 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %820 = tosa.reciprocal %819 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %821 = tosa.mul %820, %818 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %822 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %823 = tosa.add %821, %822 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %824 = tosa.rsqrt %823 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %825 = tosa.mul %815, %824 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %826 = tosa.reshape %arg68 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %827 = tosa.mul %826, %825 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %828 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %829 = tosa.transpose %arg69, %828 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %830 = tosa.reshape %827 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_143 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %831 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%830, %829 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_143 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %832 = tosa.reshape %831 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %833 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %834 = tosa.transpose %arg70, %833 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %835 = tosa.reshape %827 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_144 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %836 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%835, %834 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_144 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %837 = tosa.reshape %836 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %838 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %839 = tosa.transpose %arg71, %838 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %840 = tosa.reshape %827 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_145 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %841 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%840, %839 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_145 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %842 = tosa.reshape %841 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %843 = tosa.reshape %832 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %844 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %845 = tosa.transpose %843, %844 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %846 = tosa.reshape %837 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %847 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %848 = tosa.transpose %846, %847 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %849 = tosa.reshape %842 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %850 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %851 = tosa.transpose %849, %850 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_146 = tensor.extract_slice %arg72[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_147 = tensor.extract_slice %extracted_slice_146[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_148 = tensor.extract_slice %extracted_slice_147[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_149 = tensor.extract_slice %arg73[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_150 = tensor.extract_slice %extracted_slice_149[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_151 = tensor.extract_slice %extracted_slice_150[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %852 = tensor.empty() : tensor<1x40x128xf32>
    %853 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_148 : tensor<1x1x40x128xf32>) outs(%852 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %854 = tensor.empty() : tensor<40x128xf32>
    %855 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%853 : tensor<1x40x128xf32>) outs(%854 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %856 = tensor.empty() : tensor<1x40x128xf32>
    %857 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_151 : tensor<1x1x40x128xf32>) outs(%856 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %858 = tensor.empty() : tensor<40x128xf32>
    %859 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%857 : tensor<1x40x128xf32>) outs(%858 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %860 = tensor.empty() : tensor<1x40x128xf32>
    %861 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%860 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %855[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %862 = tosa.reshape %861 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %863 = tensor.empty() : tensor<1x40x128xf32>
    %864 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%863 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %859[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %865 = tosa.reshape %864 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %866 = tosa.mul %845, %862 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_152 = tensor.extract_slice %845[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_153 = tensor.extract_slice %845[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %867 = tensor.empty() : tensor<1x32x40x64xf32>
    %868 = linalg.negf ins(%extracted_slice_153 : tensor<1x32x40x64xf32>) outs(%867 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %869 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_154 = tensor.insert_slice %868 into %869[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_155 = tensor.insert_slice %extracted_slice_152 into %inserted_slice_154[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %870 = tosa.mul %inserted_slice_155, %865 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %871 = tosa.add %866, %870 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %872 = tosa.mul %848, %862 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_156 = tensor.extract_slice %848[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_157 = tensor.extract_slice %848[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %873 = tensor.empty() : tensor<1x32x40x64xf32>
    %874 = linalg.negf ins(%extracted_slice_157 : tensor<1x32x40x64xf32>) outs(%873 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %875 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_158 = tensor.insert_slice %874 into %875[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_159 = tensor.insert_slice %extracted_slice_156 into %inserted_slice_158[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %876 = tosa.mul %inserted_slice_159, %865 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %877 = tosa.add %872, %876 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %878 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %879 = tosa.transpose %877, %878 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %880 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %881 = tosa.add %871, %880 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %882 = tosa.reshape %881 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %883 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %884 = tosa.add %879, %883 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %885 = tosa.reshape %884 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %886 = tosa.matmul %882, %885 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %887 = tosa.reshape %886 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %888 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %889 = tosa.reciprocal %888 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %890 = tosa.mul %887, %889 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %891 = tosa.add %890, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %892 = tosa.reduce_max %891 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %893 = tosa.sub %891, %892 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %894 = tosa.exp %893 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %895 = tosa.reduce_sum %894 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %896 = tosa.reciprocal %895 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %897 = tosa.mul %894, %896 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %898 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %899 = tosa.add %897, %898 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %900 = tosa.reshape %899 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %901 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %902 = tosa.add %851, %901 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %903 = tosa.reshape %902 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %904 = tosa.matmul %900, %903 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %905 = tosa.reshape %904 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %906 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %907 = tosa.transpose %905, %906 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %908 = tosa.identity %907 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %909 = tosa.reshape %908 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %910 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %911 = tosa.transpose %arg74, %910 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %912 = tosa.reshape %909 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_160 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %913 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%912, %911 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_160 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %914 = tosa.reshape %913 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %915 = tosa.add %815, %914 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %916 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_161 = arith.constant 2 : i32
    %917 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%915 : tensor<1x40x4096xf32>) outs(%916 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_161 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %918 = tosa.reduce_sum %917 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %919 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %920 = tosa.reciprocal %919 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %921 = tosa.mul %920, %918 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %922 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %923 = tosa.add %921, %922 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %924 = tosa.rsqrt %923 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %925 = tosa.mul %915, %924 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %926 = tosa.reshape %arg75 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %927 = tosa.mul %926, %925 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %928 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %929 = tosa.transpose %arg76, %928 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %930 = tosa.reshape %927 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_162 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %931 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%930, %929 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_162 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %932 = tosa.reshape %931 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %933 = tosa.sigmoid %932 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %934 = tosa.mul %932, %933 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %935 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %936 = tosa.transpose %arg77, %935 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %937 = tosa.reshape %927 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_163 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %938 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%937, %936 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_163 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %939 = tosa.reshape %938 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %940 = tosa.mul %934, %939 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %941 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %942 = tosa.transpose %arg78, %941 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %943 = tosa.reshape %940 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_164 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %944 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%943, %942 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_164 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %945 = tosa.reshape %944 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %946 = tosa.add %915, %945 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %947 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_165 = arith.constant 2 : i32
    %948 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%946 : tensor<1x40x4096xf32>) outs(%947 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_165 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %949 = tosa.reduce_sum %948 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %950 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %951 = tosa.reciprocal %950 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %952 = tosa.mul %951, %949 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %953 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %954 = tosa.add %952, %953 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %955 = tosa.rsqrt %954 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %956 = tosa.mul %946, %955 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %957 = tosa.reshape %arg79 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %958 = tosa.mul %957, %956 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %959 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %960 = tosa.transpose %arg80, %959 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %961 = tosa.reshape %958 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_166 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %962 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%961, %960 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_166 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %963 = tosa.reshape %962 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %964 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %965 = tosa.transpose %arg81, %964 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %966 = tosa.reshape %958 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_167 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %967 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%966, %965 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_167 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %968 = tosa.reshape %967 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %969 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %970 = tosa.transpose %arg82, %969 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %971 = tosa.reshape %958 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_168 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %972 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%971, %970 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_168 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %973 = tosa.reshape %972 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %974 = tosa.reshape %963 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %975 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %976 = tosa.transpose %974, %975 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %977 = tosa.reshape %968 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %978 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %979 = tosa.transpose %977, %978 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %980 = tosa.reshape %973 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %981 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %982 = tosa.transpose %980, %981 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_169 = tensor.extract_slice %arg83[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_170 = tensor.extract_slice %extracted_slice_169[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_171 = tensor.extract_slice %extracted_slice_170[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_172 = tensor.extract_slice %arg84[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_173 = tensor.extract_slice %extracted_slice_172[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_174 = tensor.extract_slice %extracted_slice_173[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %983 = tensor.empty() : tensor<1x40x128xf32>
    %984 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_171 : tensor<1x1x40x128xf32>) outs(%983 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %985 = tensor.empty() : tensor<40x128xf32>
    %986 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%984 : tensor<1x40x128xf32>) outs(%985 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %987 = tensor.empty() : tensor<1x40x128xf32>
    %988 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_174 : tensor<1x1x40x128xf32>) outs(%987 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %989 = tensor.empty() : tensor<40x128xf32>
    %990 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%988 : tensor<1x40x128xf32>) outs(%989 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %991 = tensor.empty() : tensor<1x40x128xf32>
    %992 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%991 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %986[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %993 = tosa.reshape %992 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %994 = tensor.empty() : tensor<1x40x128xf32>
    %995 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%994 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %990[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %996 = tosa.reshape %995 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %997 = tosa.mul %976, %993 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_175 = tensor.extract_slice %976[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_176 = tensor.extract_slice %976[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %998 = tensor.empty() : tensor<1x32x40x64xf32>
    %999 = linalg.negf ins(%extracted_slice_176 : tensor<1x32x40x64xf32>) outs(%998 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1000 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_177 = tensor.insert_slice %999 into %1000[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_178 = tensor.insert_slice %extracted_slice_175 into %inserted_slice_177[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1001 = tosa.mul %inserted_slice_178, %996 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1002 = tosa.add %997, %1001 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1003 = tosa.mul %979, %993 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_179 = tensor.extract_slice %979[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_180 = tensor.extract_slice %979[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1004 = tensor.empty() : tensor<1x32x40x64xf32>
    %1005 = linalg.negf ins(%extracted_slice_180 : tensor<1x32x40x64xf32>) outs(%1004 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1006 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_181 = tensor.insert_slice %1005 into %1006[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_182 = tensor.insert_slice %extracted_slice_179 into %inserted_slice_181[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1007 = tosa.mul %inserted_slice_182, %996 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1008 = tosa.add %1003, %1007 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1009 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1010 = tosa.transpose %1008, %1009 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1011 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1012 = tosa.add %1002, %1011 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1013 = tosa.reshape %1012 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1014 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1015 = tosa.add %1010, %1014 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1016 = tosa.reshape %1015 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1017 = tosa.matmul %1013, %1016 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1018 = tosa.reshape %1017 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1019 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1020 = tosa.reciprocal %1019 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1021 = tosa.mul %1018, %1020 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1022 = tosa.add %1021, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1023 = tosa.reduce_max %1022 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1024 = tosa.sub %1022, %1023 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1025 = tosa.exp %1024 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1026 = tosa.reduce_sum %1025 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1027 = tosa.reciprocal %1026 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1028 = tosa.mul %1025, %1027 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1029 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1030 = tosa.add %1028, %1029 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1031 = tosa.reshape %1030 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1032 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1033 = tosa.add %982, %1032 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1034 = tosa.reshape %1033 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1035 = tosa.matmul %1031, %1034 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1036 = tosa.reshape %1035 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1037 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1038 = tosa.transpose %1036, %1037 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1039 = tosa.identity %1038 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1040 = tosa.reshape %1039 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1041 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1042 = tosa.transpose %arg85, %1041 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1043 = tosa.reshape %1040 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_183 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1044 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1043, %1042 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_183 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1045 = tosa.reshape %1044 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1046 = tosa.add %946, %1045 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1047 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_184 = arith.constant 2 : i32
    %1048 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1046 : tensor<1x40x4096xf32>) outs(%1047 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_184 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1049 = tosa.reduce_sum %1048 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1050 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1051 = tosa.reciprocal %1050 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1052 = tosa.mul %1051, %1049 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1053 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1054 = tosa.add %1052, %1053 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1055 = tosa.rsqrt %1054 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1056 = tosa.mul %1046, %1055 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1057 = tosa.reshape %arg86 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1058 = tosa.mul %1057, %1056 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1059 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1060 = tosa.transpose %arg87, %1059 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1061 = tosa.reshape %1058 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_185 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1062 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1061, %1060 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_185 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1063 = tosa.reshape %1062 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1064 = tosa.sigmoid %1063 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1065 = tosa.mul %1063, %1064 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1066 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1067 = tosa.transpose %arg88, %1066 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1068 = tosa.reshape %1058 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_186 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1069 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1068, %1067 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_186 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1070 = tosa.reshape %1069 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1071 = tosa.mul %1065, %1070 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1072 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1073 = tosa.transpose %arg89, %1072 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1074 = tosa.reshape %1071 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_187 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1075 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1074, %1073 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_187 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1076 = tosa.reshape %1075 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1077 = tosa.add %1046, %1076 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1078 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_188 = arith.constant 2 : i32
    %1079 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1077 : tensor<1x40x4096xf32>) outs(%1078 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_188 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1080 = tosa.reduce_sum %1079 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1081 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1082 = tosa.reciprocal %1081 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1083 = tosa.mul %1082, %1080 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1084 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1085 = tosa.add %1083, %1084 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1086 = tosa.rsqrt %1085 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1087 = tosa.mul %1077, %1086 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1088 = tosa.reshape %arg90 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1089 = tosa.mul %1088, %1087 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1090 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1091 = tosa.transpose %arg91, %1090 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1092 = tosa.reshape %1089 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_189 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1093 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1092, %1091 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_189 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1094 = tosa.reshape %1093 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1095 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1096 = tosa.transpose %arg92, %1095 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1097 = tosa.reshape %1089 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_190 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1098 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1097, %1096 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_190 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1099 = tosa.reshape %1098 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1100 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1101 = tosa.transpose %arg93, %1100 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1102 = tosa.reshape %1089 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_191 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1103 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1102, %1101 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_191 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1104 = tosa.reshape %1103 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1105 = tosa.reshape %1094 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1106 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1107 = tosa.transpose %1105, %1106 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1108 = tosa.reshape %1099 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1109 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1110 = tosa.transpose %1108, %1109 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1111 = tosa.reshape %1104 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1112 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1113 = tosa.transpose %1111, %1112 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_192 = tensor.extract_slice %arg94[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_193 = tensor.extract_slice %extracted_slice_192[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_194 = tensor.extract_slice %extracted_slice_193[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_195 = tensor.extract_slice %arg95[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_196 = tensor.extract_slice %extracted_slice_195[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_197 = tensor.extract_slice %extracted_slice_196[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1114 = tensor.empty() : tensor<1x40x128xf32>
    %1115 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_194 : tensor<1x1x40x128xf32>) outs(%1114 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1116 = tensor.empty() : tensor<40x128xf32>
    %1117 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1115 : tensor<1x40x128xf32>) outs(%1116 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1118 = tensor.empty() : tensor<1x40x128xf32>
    %1119 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_197 : tensor<1x1x40x128xf32>) outs(%1118 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1120 = tensor.empty() : tensor<40x128xf32>
    %1121 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1119 : tensor<1x40x128xf32>) outs(%1120 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1122 = tensor.empty() : tensor<1x40x128xf32>
    %1123 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1122 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1117[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1124 = tosa.reshape %1123 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %1125 = tensor.empty() : tensor<1x40x128xf32>
    %1126 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1125 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1121[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1127 = tosa.reshape %1126 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1128 = tosa.mul %1107, %1124 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_198 = tensor.extract_slice %1107[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_199 = tensor.extract_slice %1107[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1129 = tensor.empty() : tensor<1x32x40x64xf32>
    %1130 = linalg.negf ins(%extracted_slice_199 : tensor<1x32x40x64xf32>) outs(%1129 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1131 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_200 = tensor.insert_slice %1130 into %1131[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_201 = tensor.insert_slice %extracted_slice_198 into %inserted_slice_200[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1132 = tosa.mul %inserted_slice_201, %1127 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1133 = tosa.add %1128, %1132 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1134 = tosa.mul %1110, %1124 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_202 = tensor.extract_slice %1110[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_203 = tensor.extract_slice %1110[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1135 = tensor.empty() : tensor<1x32x40x64xf32>
    %1136 = linalg.negf ins(%extracted_slice_203 : tensor<1x32x40x64xf32>) outs(%1135 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1137 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_204 = tensor.insert_slice %1136 into %1137[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_205 = tensor.insert_slice %extracted_slice_202 into %inserted_slice_204[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1138 = tosa.mul %inserted_slice_205, %1127 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1139 = tosa.add %1134, %1138 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1140 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1141 = tosa.transpose %1139, %1140 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1142 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1143 = tosa.add %1133, %1142 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1144 = tosa.reshape %1143 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1145 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1146 = tosa.add %1141, %1145 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1147 = tosa.reshape %1146 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1148 = tosa.matmul %1144, %1147 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1149 = tosa.reshape %1148 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1150 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1151 = tosa.reciprocal %1150 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1152 = tosa.mul %1149, %1151 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1153 = tosa.add %1152, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1154 = tosa.reduce_max %1153 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1155 = tosa.sub %1153, %1154 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1156 = tosa.exp %1155 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1157 = tosa.reduce_sum %1156 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1158 = tosa.reciprocal %1157 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1159 = tosa.mul %1156, %1158 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1160 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1161 = tosa.add %1159, %1160 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1162 = tosa.reshape %1161 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1163 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1164 = tosa.add %1113, %1163 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1165 = tosa.reshape %1164 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1166 = tosa.matmul %1162, %1165 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1167 = tosa.reshape %1166 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1168 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1169 = tosa.transpose %1167, %1168 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1170 = tosa.identity %1169 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1171 = tosa.reshape %1170 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1172 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1173 = tosa.transpose %arg96, %1172 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1174 = tosa.reshape %1171 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_206 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1175 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1174, %1173 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_206 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1176 = tosa.reshape %1175 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1177 = tosa.add %1077, %1176 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1178 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_207 = arith.constant 2 : i32
    %1179 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1177 : tensor<1x40x4096xf32>) outs(%1178 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_207 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1180 = tosa.reduce_sum %1179 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1181 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1182 = tosa.reciprocal %1181 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1183 = tosa.mul %1182, %1180 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1184 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1185 = tosa.add %1183, %1184 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1186 = tosa.rsqrt %1185 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1187 = tosa.mul %1177, %1186 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1188 = tosa.reshape %arg97 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1189 = tosa.mul %1188, %1187 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1190 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1191 = tosa.transpose %arg98, %1190 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1192 = tosa.reshape %1189 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_208 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1193 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1192, %1191 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_208 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1194 = tosa.reshape %1193 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1195 = tosa.sigmoid %1194 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1196 = tosa.mul %1194, %1195 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1197 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1198 = tosa.transpose %arg99, %1197 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1199 = tosa.reshape %1189 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_209 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1200 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1199, %1198 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_209 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1201 = tosa.reshape %1200 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1202 = tosa.mul %1196, %1201 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1203 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1204 = tosa.transpose %arg100, %1203 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1205 = tosa.reshape %1202 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_210 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1206 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1205, %1204 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_210 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1207 = tosa.reshape %1206 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1208 = tosa.add %1177, %1207 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1209 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_211 = arith.constant 2 : i32
    %1210 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1208 : tensor<1x40x4096xf32>) outs(%1209 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_211 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1211 = tosa.reduce_sum %1210 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1212 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1213 = tosa.reciprocal %1212 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1214 = tosa.mul %1213, %1211 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1215 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1216 = tosa.add %1214, %1215 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1217 = tosa.rsqrt %1216 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1218 = tosa.mul %1208, %1217 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1219 = tosa.reshape %arg101 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1220 = tosa.mul %1219, %1218 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1221 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1222 = tosa.transpose %arg102, %1221 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1223 = tosa.reshape %1220 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_212 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1224 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1223, %1222 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_212 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1225 = tosa.reshape %1224 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1226 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1227 = tosa.transpose %arg103, %1226 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1228 = tosa.reshape %1220 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_213 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1229 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1228, %1227 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_213 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1230 = tosa.reshape %1229 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1231 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1232 = tosa.transpose %arg104, %1231 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1233 = tosa.reshape %1220 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_214 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1234 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1233, %1232 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_214 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1235 = tosa.reshape %1234 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1236 = tosa.reshape %1225 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1237 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1238 = tosa.transpose %1236, %1237 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1239 = tosa.reshape %1230 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1240 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1241 = tosa.transpose %1239, %1240 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1242 = tosa.reshape %1235 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1243 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1244 = tosa.transpose %1242, %1243 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_215 = tensor.extract_slice %arg105[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_216 = tensor.extract_slice %extracted_slice_215[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_217 = tensor.extract_slice %extracted_slice_216[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_218 = tensor.extract_slice %arg106[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_219 = tensor.extract_slice %extracted_slice_218[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_220 = tensor.extract_slice %extracted_slice_219[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1245 = tensor.empty() : tensor<1x40x128xf32>
    %1246 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_217 : tensor<1x1x40x128xf32>) outs(%1245 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1247 = tensor.empty() : tensor<40x128xf32>
    %1248 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1246 : tensor<1x40x128xf32>) outs(%1247 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1249 = tensor.empty() : tensor<1x40x128xf32>
    %1250 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_220 : tensor<1x1x40x128xf32>) outs(%1249 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1251 = tensor.empty() : tensor<40x128xf32>
    %1252 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1250 : tensor<1x40x128xf32>) outs(%1251 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1253 = tensor.empty() : tensor<1x40x128xf32>
    %1254 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1253 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1248[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1255 = tosa.reshape %1254 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %1256 = tensor.empty() : tensor<1x40x128xf32>
    %1257 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1256 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1252[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1258 = tosa.reshape %1257 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1259 = tosa.mul %1238, %1255 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_221 = tensor.extract_slice %1238[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_222 = tensor.extract_slice %1238[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1260 = tensor.empty() : tensor<1x32x40x64xf32>
    %1261 = linalg.negf ins(%extracted_slice_222 : tensor<1x32x40x64xf32>) outs(%1260 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1262 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_223 = tensor.insert_slice %1261 into %1262[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_224 = tensor.insert_slice %extracted_slice_221 into %inserted_slice_223[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1263 = tosa.mul %inserted_slice_224, %1258 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1264 = tosa.add %1259, %1263 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1265 = tosa.mul %1241, %1255 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_225 = tensor.extract_slice %1241[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_226 = tensor.extract_slice %1241[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1266 = tensor.empty() : tensor<1x32x40x64xf32>
    %1267 = linalg.negf ins(%extracted_slice_226 : tensor<1x32x40x64xf32>) outs(%1266 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1268 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_227 = tensor.insert_slice %1267 into %1268[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_228 = tensor.insert_slice %extracted_slice_225 into %inserted_slice_227[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1269 = tosa.mul %inserted_slice_228, %1258 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1270 = tosa.add %1265, %1269 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1271 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1272 = tosa.transpose %1270, %1271 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1273 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1274 = tosa.add %1264, %1273 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1275 = tosa.reshape %1274 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1276 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1277 = tosa.add %1272, %1276 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1278 = tosa.reshape %1277 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1279 = tosa.matmul %1275, %1278 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1280 = tosa.reshape %1279 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1281 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1282 = tosa.reciprocal %1281 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1283 = tosa.mul %1280, %1282 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1284 = tosa.add %1283, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1285 = tosa.reduce_max %1284 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1286 = tosa.sub %1284, %1285 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1287 = tosa.exp %1286 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1288 = tosa.reduce_sum %1287 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1289 = tosa.reciprocal %1288 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1290 = tosa.mul %1287, %1289 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1291 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1292 = tosa.add %1290, %1291 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1293 = tosa.reshape %1292 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1294 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1295 = tosa.add %1244, %1294 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1296 = tosa.reshape %1295 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1297 = tosa.matmul %1293, %1296 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1298 = tosa.reshape %1297 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1299 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1300 = tosa.transpose %1298, %1299 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1301 = tosa.identity %1300 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1302 = tosa.reshape %1301 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1303 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1304 = tosa.transpose %arg107, %1303 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1305 = tosa.reshape %1302 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_229 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1306 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1305, %1304 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_229 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1307 = tosa.reshape %1306 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1308 = tosa.add %1208, %1307 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1309 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_230 = arith.constant 2 : i32
    %1310 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1308 : tensor<1x40x4096xf32>) outs(%1309 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_230 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1311 = tosa.reduce_sum %1310 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1312 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1313 = tosa.reciprocal %1312 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1314 = tosa.mul %1313, %1311 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1315 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1316 = tosa.add %1314, %1315 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1317 = tosa.rsqrt %1316 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1318 = tosa.mul %1308, %1317 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1319 = tosa.reshape %arg108 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1320 = tosa.mul %1319, %1318 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1321 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1322 = tosa.transpose %arg109, %1321 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1323 = tosa.reshape %1320 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_231 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1324 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1323, %1322 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_231 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1325 = tosa.reshape %1324 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1326 = tosa.sigmoid %1325 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1327 = tosa.mul %1325, %1326 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1328 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1329 = tosa.transpose %arg110, %1328 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1330 = tosa.reshape %1320 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_232 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1331 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1330, %1329 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_232 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1332 = tosa.reshape %1331 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1333 = tosa.mul %1327, %1332 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1334 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1335 = tosa.transpose %arg111, %1334 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1336 = tosa.reshape %1333 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_233 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1337 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1336, %1335 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_233 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1338 = tosa.reshape %1337 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1339 = tosa.add %1308, %1338 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1340 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_234 = arith.constant 2 : i32
    %1341 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1339 : tensor<1x40x4096xf32>) outs(%1340 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_234 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1342 = tosa.reduce_sum %1341 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1343 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1344 = tosa.reciprocal %1343 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1345 = tosa.mul %1344, %1342 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1346 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1347 = tosa.add %1345, %1346 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1348 = tosa.rsqrt %1347 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1349 = tosa.mul %1339, %1348 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1350 = tosa.reshape %arg112 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1351 = tosa.mul %1350, %1349 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1352 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1353 = tosa.transpose %arg113, %1352 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1354 = tosa.reshape %1351 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_235 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1355 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1354, %1353 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_235 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1356 = tosa.reshape %1355 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1357 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1358 = tosa.transpose %arg114, %1357 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1359 = tosa.reshape %1351 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_236 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1360 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1359, %1358 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_236 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1361 = tosa.reshape %1360 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1362 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1363 = tosa.transpose %arg115, %1362 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1364 = tosa.reshape %1351 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_237 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1365 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1364, %1363 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_237 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1366 = tosa.reshape %1365 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1367 = tosa.reshape %1356 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1368 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1369 = tosa.transpose %1367, %1368 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1370 = tosa.reshape %1361 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1371 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1372 = tosa.transpose %1370, %1371 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1373 = tosa.reshape %1366 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1374 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1375 = tosa.transpose %1373, %1374 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_238 = tensor.extract_slice %arg116[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_239 = tensor.extract_slice %extracted_slice_238[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_240 = tensor.extract_slice %extracted_slice_239[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_241 = tensor.extract_slice %arg117[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_242 = tensor.extract_slice %extracted_slice_241[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_243 = tensor.extract_slice %extracted_slice_242[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1376 = tensor.empty() : tensor<1x40x128xf32>
    %1377 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_240 : tensor<1x1x40x128xf32>) outs(%1376 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1378 = tensor.empty() : tensor<40x128xf32>
    %1379 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1377 : tensor<1x40x128xf32>) outs(%1378 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1380 = tensor.empty() : tensor<1x40x128xf32>
    %1381 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_243 : tensor<1x1x40x128xf32>) outs(%1380 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1382 = tensor.empty() : tensor<40x128xf32>
    %1383 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1381 : tensor<1x40x128xf32>) outs(%1382 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1384 = tensor.empty() : tensor<1x40x128xf32>
    %1385 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1384 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1379[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1386 = tosa.reshape %1385 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %1387 = tensor.empty() : tensor<1x40x128xf32>
    %1388 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1387 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1383[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1389 = tosa.reshape %1388 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1390 = tosa.mul %1369, %1386 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_244 = tensor.extract_slice %1369[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_245 = tensor.extract_slice %1369[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1391 = tensor.empty() : tensor<1x32x40x64xf32>
    %1392 = linalg.negf ins(%extracted_slice_245 : tensor<1x32x40x64xf32>) outs(%1391 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1393 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_246 = tensor.insert_slice %1392 into %1393[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_247 = tensor.insert_slice %extracted_slice_244 into %inserted_slice_246[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1394 = tosa.mul %inserted_slice_247, %1389 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1395 = tosa.add %1390, %1394 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1396 = tosa.mul %1372, %1386 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_248 = tensor.extract_slice %1372[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_249 = tensor.extract_slice %1372[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1397 = tensor.empty() : tensor<1x32x40x64xf32>
    %1398 = linalg.negf ins(%extracted_slice_249 : tensor<1x32x40x64xf32>) outs(%1397 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1399 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_250 = tensor.insert_slice %1398 into %1399[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_251 = tensor.insert_slice %extracted_slice_248 into %inserted_slice_250[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1400 = tosa.mul %inserted_slice_251, %1389 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1401 = tosa.add %1396, %1400 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1402 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1403 = tosa.transpose %1401, %1402 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1404 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1405 = tosa.add %1395, %1404 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1406 = tosa.reshape %1405 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1407 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1408 = tosa.add %1403, %1407 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1409 = tosa.reshape %1408 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1410 = tosa.matmul %1406, %1409 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1411 = tosa.reshape %1410 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1412 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1413 = tosa.reciprocal %1412 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1414 = tosa.mul %1411, %1413 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1415 = tosa.add %1414, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1416 = tosa.reduce_max %1415 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1417 = tosa.sub %1415, %1416 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1418 = tosa.exp %1417 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1419 = tosa.reduce_sum %1418 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1420 = tosa.reciprocal %1419 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1421 = tosa.mul %1418, %1420 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1422 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1423 = tosa.add %1421, %1422 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1424 = tosa.reshape %1423 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1425 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1426 = tosa.add %1375, %1425 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1427 = tosa.reshape %1426 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1428 = tosa.matmul %1424, %1427 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1429 = tosa.reshape %1428 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1430 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1431 = tosa.transpose %1429, %1430 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1432 = tosa.identity %1431 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1433 = tosa.reshape %1432 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1434 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1435 = tosa.transpose %arg118, %1434 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1436 = tosa.reshape %1433 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_252 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1437 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1436, %1435 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_252 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1438 = tosa.reshape %1437 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1439 = tosa.add %1339, %1438 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1440 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_253 = arith.constant 2 : i32
    %1441 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1439 : tensor<1x40x4096xf32>) outs(%1440 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_253 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1442 = tosa.reduce_sum %1441 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1443 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1444 = tosa.reciprocal %1443 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1445 = tosa.mul %1444, %1442 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1446 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1447 = tosa.add %1445, %1446 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1448 = tosa.rsqrt %1447 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1449 = tosa.mul %1439, %1448 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1450 = tosa.reshape %arg119 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1451 = tosa.mul %1450, %1449 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1452 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1453 = tosa.transpose %arg120, %1452 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1454 = tosa.reshape %1451 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_254 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1455 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1454, %1453 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_254 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1456 = tosa.reshape %1455 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1457 = tosa.sigmoid %1456 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1458 = tosa.mul %1456, %1457 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1459 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1460 = tosa.transpose %arg121, %1459 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1461 = tosa.reshape %1451 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_255 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1462 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1461, %1460 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_255 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1463 = tosa.reshape %1462 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1464 = tosa.mul %1458, %1463 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1465 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1466 = tosa.transpose %arg122, %1465 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1467 = tosa.reshape %1464 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_256 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1468 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1467, %1466 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_256 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1469 = tosa.reshape %1468 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1470 = tosa.add %1439, %1469 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1471 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_257 = arith.constant 2 : i32
    %1472 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1470 : tensor<1x40x4096xf32>) outs(%1471 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_257 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1473 = tosa.reduce_sum %1472 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1474 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1475 = tosa.reciprocal %1474 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1476 = tosa.mul %1475, %1473 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1477 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1478 = tosa.add %1476, %1477 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1479 = tosa.rsqrt %1478 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1480 = tosa.mul %1470, %1479 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1481 = tosa.reshape %arg123 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1482 = tosa.mul %1481, %1480 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1483 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1484 = tosa.transpose %arg124, %1483 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1485 = tosa.reshape %1482 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_258 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1486 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1485, %1484 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_258 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1487 = tosa.reshape %1486 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1488 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1489 = tosa.transpose %arg125, %1488 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1490 = tosa.reshape %1482 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_259 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1491 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1490, %1489 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_259 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1492 = tosa.reshape %1491 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1493 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1494 = tosa.transpose %arg126, %1493 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1495 = tosa.reshape %1482 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_260 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1496 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1495, %1494 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_260 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1497 = tosa.reshape %1496 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1498 = tosa.reshape %1487 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1499 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1500 = tosa.transpose %1498, %1499 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1501 = tosa.reshape %1492 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1502 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1503 = tosa.transpose %1501, %1502 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1504 = tosa.reshape %1497 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1505 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1506 = tosa.transpose %1504, %1505 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_261 = tensor.extract_slice %arg127[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_262 = tensor.extract_slice %extracted_slice_261[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_263 = tensor.extract_slice %extracted_slice_262[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_264 = tensor.extract_slice %arg128[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_265 = tensor.extract_slice %extracted_slice_264[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_266 = tensor.extract_slice %extracted_slice_265[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1507 = tensor.empty() : tensor<1x40x128xf32>
    %1508 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_263 : tensor<1x1x40x128xf32>) outs(%1507 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1509 = tensor.empty() : tensor<40x128xf32>
    %1510 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1508 : tensor<1x40x128xf32>) outs(%1509 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1511 = tensor.empty() : tensor<1x40x128xf32>
    %1512 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_266 : tensor<1x1x40x128xf32>) outs(%1511 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1513 = tensor.empty() : tensor<40x128xf32>
    %1514 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1512 : tensor<1x40x128xf32>) outs(%1513 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1515 = tensor.empty() : tensor<1x40x128xf32>
    %1516 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1515 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1510[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1517 = tosa.reshape %1516 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %1518 = tensor.empty() : tensor<1x40x128xf32>
    %1519 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1518 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1514[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1520 = tosa.reshape %1519 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1521 = tosa.mul %1500, %1517 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_267 = tensor.extract_slice %1500[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_268 = tensor.extract_slice %1500[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1522 = tensor.empty() : tensor<1x32x40x64xf32>
    %1523 = linalg.negf ins(%extracted_slice_268 : tensor<1x32x40x64xf32>) outs(%1522 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1524 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_269 = tensor.insert_slice %1523 into %1524[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_270 = tensor.insert_slice %extracted_slice_267 into %inserted_slice_269[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1525 = tosa.mul %inserted_slice_270, %1520 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1526 = tosa.add %1521, %1525 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1527 = tosa.mul %1503, %1517 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_271 = tensor.extract_slice %1503[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_272 = tensor.extract_slice %1503[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1528 = tensor.empty() : tensor<1x32x40x64xf32>
    %1529 = linalg.negf ins(%extracted_slice_272 : tensor<1x32x40x64xf32>) outs(%1528 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1530 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_273 = tensor.insert_slice %1529 into %1530[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_274 = tensor.insert_slice %extracted_slice_271 into %inserted_slice_273[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1531 = tosa.mul %inserted_slice_274, %1520 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1532 = tosa.add %1527, %1531 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1533 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1534 = tosa.transpose %1532, %1533 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1535 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1536 = tosa.add %1526, %1535 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1537 = tosa.reshape %1536 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1538 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1539 = tosa.add %1534, %1538 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1540 = tosa.reshape %1539 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1541 = tosa.matmul %1537, %1540 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1542 = tosa.reshape %1541 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1543 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1544 = tosa.reciprocal %1543 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1545 = tosa.mul %1542, %1544 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1546 = tosa.add %1545, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1547 = tosa.reduce_max %1546 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1548 = tosa.sub %1546, %1547 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1549 = tosa.exp %1548 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1550 = tosa.reduce_sum %1549 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1551 = tosa.reciprocal %1550 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1552 = tosa.mul %1549, %1551 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1553 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1554 = tosa.add %1552, %1553 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1555 = tosa.reshape %1554 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1556 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1557 = tosa.add %1506, %1556 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1558 = tosa.reshape %1557 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1559 = tosa.matmul %1555, %1558 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1560 = tosa.reshape %1559 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1561 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1562 = tosa.transpose %1560, %1561 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1563 = tosa.identity %1562 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1564 = tosa.reshape %1563 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1565 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1566 = tosa.transpose %arg129, %1565 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1567 = tosa.reshape %1564 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_275 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1568 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1567, %1566 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_275 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1569 = tosa.reshape %1568 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1570 = tosa.add %1470, %1569 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1571 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_276 = arith.constant 2 : i32
    %1572 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1570 : tensor<1x40x4096xf32>) outs(%1571 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_276 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1573 = tosa.reduce_sum %1572 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1574 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1575 = tosa.reciprocal %1574 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1576 = tosa.mul %1575, %1573 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1577 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1578 = tosa.add %1576, %1577 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1579 = tosa.rsqrt %1578 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1580 = tosa.mul %1570, %1579 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1581 = tosa.reshape %arg130 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1582 = tosa.mul %1581, %1580 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1583 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1584 = tosa.transpose %arg131, %1583 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1585 = tosa.reshape %1582 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_277 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1586 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1585, %1584 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_277 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1587 = tosa.reshape %1586 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1588 = tosa.sigmoid %1587 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1589 = tosa.mul %1587, %1588 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1590 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1591 = tosa.transpose %arg132, %1590 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1592 = tosa.reshape %1582 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_278 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1593 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1592, %1591 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_278 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1594 = tosa.reshape %1593 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1595 = tosa.mul %1589, %1594 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1596 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1597 = tosa.transpose %arg133, %1596 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1598 = tosa.reshape %1595 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_279 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1599 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1598, %1597 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_279 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1600 = tosa.reshape %1599 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1601 = tosa.add %1570, %1600 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1602 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_280 = arith.constant 2 : i32
    %1603 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1601 : tensor<1x40x4096xf32>) outs(%1602 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_280 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1604 = tosa.reduce_sum %1603 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1605 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1606 = tosa.reciprocal %1605 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1607 = tosa.mul %1606, %1604 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1608 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1609 = tosa.add %1607, %1608 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1610 = tosa.rsqrt %1609 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1611 = tosa.mul %1601, %1610 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1612 = tosa.reshape %arg134 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1613 = tosa.mul %1612, %1611 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1614 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1615 = tosa.transpose %arg135, %1614 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1616 = tosa.reshape %1613 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_281 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1617 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1616, %1615 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_281 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1618 = tosa.reshape %1617 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1619 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1620 = tosa.transpose %arg136, %1619 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1621 = tosa.reshape %1613 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_282 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1622 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1621, %1620 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_282 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1623 = tosa.reshape %1622 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1624 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1625 = tosa.transpose %arg137, %1624 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1626 = tosa.reshape %1613 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_283 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1627 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1626, %1625 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_283 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1628 = tosa.reshape %1627 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1629 = tosa.reshape %1618 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1630 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1631 = tosa.transpose %1629, %1630 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1632 = tosa.reshape %1623 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1633 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1634 = tosa.transpose %1632, %1633 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1635 = tosa.reshape %1628 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1636 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1637 = tosa.transpose %1635, %1636 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_284 = tensor.extract_slice %arg138[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_285 = tensor.extract_slice %extracted_slice_284[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_286 = tensor.extract_slice %extracted_slice_285[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_287 = tensor.extract_slice %arg139[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_288 = tensor.extract_slice %extracted_slice_287[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_289 = tensor.extract_slice %extracted_slice_288[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1638 = tensor.empty() : tensor<1x40x128xf32>
    %1639 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_286 : tensor<1x1x40x128xf32>) outs(%1638 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1640 = tensor.empty() : tensor<40x128xf32>
    %1641 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1639 : tensor<1x40x128xf32>) outs(%1640 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1642 = tensor.empty() : tensor<1x40x128xf32>
    %1643 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_289 : tensor<1x1x40x128xf32>) outs(%1642 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1644 = tensor.empty() : tensor<40x128xf32>
    %1645 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1643 : tensor<1x40x128xf32>) outs(%1644 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1646 = tensor.empty() : tensor<1x40x128xf32>
    %1647 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1646 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1641[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1648 = tosa.reshape %1647 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %1649 = tensor.empty() : tensor<1x40x128xf32>
    %1650 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1649 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1645[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1651 = tosa.reshape %1650 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1652 = tosa.mul %1631, %1648 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_290 = tensor.extract_slice %1631[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_291 = tensor.extract_slice %1631[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1653 = tensor.empty() : tensor<1x32x40x64xf32>
    %1654 = linalg.negf ins(%extracted_slice_291 : tensor<1x32x40x64xf32>) outs(%1653 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1655 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_292 = tensor.insert_slice %1654 into %1655[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_293 = tensor.insert_slice %extracted_slice_290 into %inserted_slice_292[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1656 = tosa.mul %inserted_slice_293, %1651 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1657 = tosa.add %1652, %1656 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1658 = tosa.mul %1634, %1648 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_294 = tensor.extract_slice %1634[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_295 = tensor.extract_slice %1634[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1659 = tensor.empty() : tensor<1x32x40x64xf32>
    %1660 = linalg.negf ins(%extracted_slice_295 : tensor<1x32x40x64xf32>) outs(%1659 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1661 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_296 = tensor.insert_slice %1660 into %1661[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_297 = tensor.insert_slice %extracted_slice_294 into %inserted_slice_296[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1662 = tosa.mul %inserted_slice_297, %1651 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1663 = tosa.add %1658, %1662 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1664 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1665 = tosa.transpose %1663, %1664 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1666 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1667 = tosa.add %1657, %1666 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1668 = tosa.reshape %1667 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1669 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1670 = tosa.add %1665, %1669 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1671 = tosa.reshape %1670 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1672 = tosa.matmul %1668, %1671 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1673 = tosa.reshape %1672 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1674 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1675 = tosa.reciprocal %1674 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1676 = tosa.mul %1673, %1675 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1677 = tosa.add %1676, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1678 = tosa.reduce_max %1677 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1679 = tosa.sub %1677, %1678 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1680 = tosa.exp %1679 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1681 = tosa.reduce_sum %1680 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1682 = tosa.reciprocal %1681 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1683 = tosa.mul %1680, %1682 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1684 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1685 = tosa.add %1683, %1684 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1686 = tosa.reshape %1685 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1687 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1688 = tosa.add %1637, %1687 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1689 = tosa.reshape %1688 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1690 = tosa.matmul %1686, %1689 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1691 = tosa.reshape %1690 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1692 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1693 = tosa.transpose %1691, %1692 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1694 = tosa.identity %1693 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1695 = tosa.reshape %1694 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1696 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1697 = tosa.transpose %arg140, %1696 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1698 = tosa.reshape %1695 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_298 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1699 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1698, %1697 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_298 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1700 = tosa.reshape %1699 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1701 = tosa.add %1601, %1700 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1702 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_299 = arith.constant 2 : i32
    %1703 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1701 : tensor<1x40x4096xf32>) outs(%1702 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_299 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1704 = tosa.reduce_sum %1703 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1705 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1706 = tosa.reciprocal %1705 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1707 = tosa.mul %1706, %1704 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1708 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1709 = tosa.add %1707, %1708 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1710 = tosa.rsqrt %1709 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1711 = tosa.mul %1701, %1710 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1712 = tosa.reshape %arg141 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1713 = tosa.mul %1712, %1711 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1714 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1715 = tosa.transpose %arg142, %1714 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1716 = tosa.reshape %1713 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_300 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1717 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1716, %1715 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_300 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1718 = tosa.reshape %1717 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1719 = tosa.sigmoid %1718 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1720 = tosa.mul %1718, %1719 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1721 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1722 = tosa.transpose %arg143, %1721 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1723 = tosa.reshape %1713 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_301 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1724 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1723, %1722 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_301 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1725 = tosa.reshape %1724 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1726 = tosa.mul %1720, %1725 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1727 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1728 = tosa.transpose %arg144, %1727 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1729 = tosa.reshape %1726 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_302 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1730 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1729, %1728 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_302 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1731 = tosa.reshape %1730 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1732 = tosa.add %1701, %1731 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1733 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_303 = arith.constant 2 : i32
    %1734 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1732 : tensor<1x40x4096xf32>) outs(%1733 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_303 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1735 = tosa.reduce_sum %1734 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1736 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1737 = tosa.reciprocal %1736 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1738 = tosa.mul %1737, %1735 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1739 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1740 = tosa.add %1738, %1739 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1741 = tosa.rsqrt %1740 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1742 = tosa.mul %1732, %1741 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1743 = tosa.reshape %arg145 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1744 = tosa.mul %1743, %1742 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1745 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1746 = tosa.transpose %arg146, %1745 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1747 = tosa.reshape %1744 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_304 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1748 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1747, %1746 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_304 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1749 = tosa.reshape %1748 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1750 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1751 = tosa.transpose %arg147, %1750 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1752 = tosa.reshape %1744 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_305 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1753 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1752, %1751 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_305 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1754 = tosa.reshape %1753 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1755 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1756 = tosa.transpose %arg148, %1755 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1757 = tosa.reshape %1744 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_306 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1758 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1757, %1756 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_306 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1759 = tosa.reshape %1758 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1760 = tosa.reshape %1749 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1761 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1762 = tosa.transpose %1760, %1761 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1763 = tosa.reshape %1754 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1764 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1765 = tosa.transpose %1763, %1764 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1766 = tosa.reshape %1759 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1767 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1768 = tosa.transpose %1766, %1767 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_307 = tensor.extract_slice %arg149[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_308 = tensor.extract_slice %extracted_slice_307[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_309 = tensor.extract_slice %extracted_slice_308[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_310 = tensor.extract_slice %arg150[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_311 = tensor.extract_slice %extracted_slice_310[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_312 = tensor.extract_slice %extracted_slice_311[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1769 = tensor.empty() : tensor<1x40x128xf32>
    %1770 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_309 : tensor<1x1x40x128xf32>) outs(%1769 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1771 = tensor.empty() : tensor<40x128xf32>
    %1772 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1770 : tensor<1x40x128xf32>) outs(%1771 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1773 = tensor.empty() : tensor<1x40x128xf32>
    %1774 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_312 : tensor<1x1x40x128xf32>) outs(%1773 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1775 = tensor.empty() : tensor<40x128xf32>
    %1776 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1774 : tensor<1x40x128xf32>) outs(%1775 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1777 = tensor.empty() : tensor<1x40x128xf32>
    %1778 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1777 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1772[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1779 = tosa.reshape %1778 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %1780 = tensor.empty() : tensor<1x40x128xf32>
    %1781 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1780 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1776[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1782 = tosa.reshape %1781 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1783 = tosa.mul %1762, %1779 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_313 = tensor.extract_slice %1762[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_314 = tensor.extract_slice %1762[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1784 = tensor.empty() : tensor<1x32x40x64xf32>
    %1785 = linalg.negf ins(%extracted_slice_314 : tensor<1x32x40x64xf32>) outs(%1784 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1786 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_315 = tensor.insert_slice %1785 into %1786[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_316 = tensor.insert_slice %extracted_slice_313 into %inserted_slice_315[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1787 = tosa.mul %inserted_slice_316, %1782 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1788 = tosa.add %1783, %1787 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1789 = tosa.mul %1765, %1779 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_317 = tensor.extract_slice %1765[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_318 = tensor.extract_slice %1765[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1790 = tensor.empty() : tensor<1x32x40x64xf32>
    %1791 = linalg.negf ins(%extracted_slice_318 : tensor<1x32x40x64xf32>) outs(%1790 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1792 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_319 = tensor.insert_slice %1791 into %1792[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_320 = tensor.insert_slice %extracted_slice_317 into %inserted_slice_319[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1793 = tosa.mul %inserted_slice_320, %1782 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1794 = tosa.add %1789, %1793 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1795 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1796 = tosa.transpose %1794, %1795 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1797 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1798 = tosa.add %1788, %1797 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1799 = tosa.reshape %1798 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1800 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1801 = tosa.add %1796, %1800 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1802 = tosa.reshape %1801 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1803 = tosa.matmul %1799, %1802 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1804 = tosa.reshape %1803 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1805 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1806 = tosa.reciprocal %1805 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1807 = tosa.mul %1804, %1806 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1808 = tosa.add %1807, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1809 = tosa.reduce_max %1808 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1810 = tosa.sub %1808, %1809 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1811 = tosa.exp %1810 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1812 = tosa.reduce_sum %1811 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1813 = tosa.reciprocal %1812 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1814 = tosa.mul %1811, %1813 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1815 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1816 = tosa.add %1814, %1815 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1817 = tosa.reshape %1816 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1818 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1819 = tosa.add %1768, %1818 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1820 = tosa.reshape %1819 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1821 = tosa.matmul %1817, %1820 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1822 = tosa.reshape %1821 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1823 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1824 = tosa.transpose %1822, %1823 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1825 = tosa.identity %1824 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1826 = tosa.reshape %1825 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1827 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1828 = tosa.transpose %arg151, %1827 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1829 = tosa.reshape %1826 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_321 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1830 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1829, %1828 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_321 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1831 = tosa.reshape %1830 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1832 = tosa.add %1732, %1831 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1833 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_322 = arith.constant 2 : i32
    %1834 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1832 : tensor<1x40x4096xf32>) outs(%1833 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_322 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1835 = tosa.reduce_sum %1834 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1836 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1837 = tosa.reciprocal %1836 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1838 = tosa.mul %1837, %1835 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1839 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1840 = tosa.add %1838, %1839 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1841 = tosa.rsqrt %1840 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1842 = tosa.mul %1832, %1841 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1843 = tosa.reshape %arg152 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1844 = tosa.mul %1843, %1842 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1845 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1846 = tosa.transpose %arg153, %1845 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1847 = tosa.reshape %1844 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_323 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1848 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1847, %1846 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_323 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1849 = tosa.reshape %1848 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1850 = tosa.sigmoid %1849 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1851 = tosa.mul %1849, %1850 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1852 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1853 = tosa.transpose %arg154, %1852 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1854 = tosa.reshape %1844 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_324 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1855 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1854, %1853 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_324 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1856 = tosa.reshape %1855 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1857 = tosa.mul %1851, %1856 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1858 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1859 = tosa.transpose %arg155, %1858 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1860 = tosa.reshape %1857 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_325 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1861 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1860, %1859 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_325 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1862 = tosa.reshape %1861 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1863 = tosa.add %1832, %1862 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1864 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_326 = arith.constant 2 : i32
    %1865 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1863 : tensor<1x40x4096xf32>) outs(%1864 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_326 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1866 = tosa.reduce_sum %1865 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1867 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1868 = tosa.reciprocal %1867 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1869 = tosa.mul %1868, %1866 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1870 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1871 = tosa.add %1869, %1870 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1872 = tosa.rsqrt %1871 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1873 = tosa.mul %1863, %1872 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1874 = tosa.reshape %arg156 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1875 = tosa.mul %1874, %1873 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1876 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1877 = tosa.transpose %arg157, %1876 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1878 = tosa.reshape %1875 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_327 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1879 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1878, %1877 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_327 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1880 = tosa.reshape %1879 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1881 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1882 = tosa.transpose %arg158, %1881 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1883 = tosa.reshape %1875 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_328 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1884 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1883, %1882 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_328 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1885 = tosa.reshape %1884 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1886 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1887 = tosa.transpose %arg159, %1886 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1888 = tosa.reshape %1875 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_329 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1889 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1888, %1887 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_329 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1890 = tosa.reshape %1889 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1891 = tosa.reshape %1880 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1892 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1893 = tosa.transpose %1891, %1892 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1894 = tosa.reshape %1885 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1895 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1896 = tosa.transpose %1894, %1895 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1897 = tosa.reshape %1890 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1898 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1899 = tosa.transpose %1897, %1898 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_330 = tensor.extract_slice %arg160[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_331 = tensor.extract_slice %extracted_slice_330[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_332 = tensor.extract_slice %extracted_slice_331[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_333 = tensor.extract_slice %arg161[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_334 = tensor.extract_slice %extracted_slice_333[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_335 = tensor.extract_slice %extracted_slice_334[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %1900 = tensor.empty() : tensor<1x40x128xf32>
    %1901 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_332 : tensor<1x1x40x128xf32>) outs(%1900 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1902 = tensor.empty() : tensor<40x128xf32>
    %1903 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1901 : tensor<1x40x128xf32>) outs(%1902 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1904 = tensor.empty() : tensor<1x40x128xf32>
    %1905 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_335 : tensor<1x1x40x128xf32>) outs(%1904 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %1906 = tensor.empty() : tensor<40x128xf32>
    %1907 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%1905 : tensor<1x40x128xf32>) outs(%1906 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %1908 = tensor.empty() : tensor<1x40x128xf32>
    %1909 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1908 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1903[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1910 = tosa.reshape %1909 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %1911 = tensor.empty() : tensor<1x40x128xf32>
    %1912 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%1911 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %1907[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %1913 = tosa.reshape %1912 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1914 = tosa.mul %1893, %1910 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_336 = tensor.extract_slice %1893[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_337 = tensor.extract_slice %1893[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1915 = tensor.empty() : tensor<1x32x40x64xf32>
    %1916 = linalg.negf ins(%extracted_slice_337 : tensor<1x32x40x64xf32>) outs(%1915 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1917 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_338 = tensor.insert_slice %1916 into %1917[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_339 = tensor.insert_slice %extracted_slice_336 into %inserted_slice_338[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1918 = tosa.mul %inserted_slice_339, %1913 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1919 = tosa.add %1914, %1918 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1920 = tosa.mul %1896, %1910 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_340 = tensor.extract_slice %1896[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_341 = tensor.extract_slice %1896[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %1921 = tensor.empty() : tensor<1x32x40x64xf32>
    %1922 = linalg.negf ins(%extracted_slice_341 : tensor<1x32x40x64xf32>) outs(%1921 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %1923 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_342 = tensor.insert_slice %1922 into %1923[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_343 = tensor.insert_slice %extracted_slice_340 into %inserted_slice_342[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %1924 = tosa.mul %inserted_slice_343, %1913 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1925 = tosa.add %1920, %1924 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1926 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1927 = tosa.transpose %1925, %1926 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1928 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1929 = tosa.add %1919, %1928 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1930 = tosa.reshape %1929 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1931 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1932 = tosa.add %1927, %1931 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1933 = tosa.reshape %1932 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1934 = tosa.matmul %1930, %1933 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1935 = tosa.reshape %1934 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1936 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1937 = tosa.reciprocal %1936 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1938 = tosa.mul %1935, %1937 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1939 = tosa.add %1938, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1940 = tosa.reduce_max %1939 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1941 = tosa.sub %1939, %1940 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1942 = tosa.exp %1941 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1943 = tosa.reduce_sum %1942 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1944 = tosa.reciprocal %1943 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1945 = tosa.mul %1942, %1944 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1946 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1947 = tosa.add %1945, %1946 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1948 = tosa.reshape %1947 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1949 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1950 = tosa.add %1899, %1949 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1951 = tosa.reshape %1950 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1952 = tosa.matmul %1948, %1951 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1953 = tosa.reshape %1952 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1954 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1955 = tosa.transpose %1953, %1954 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1956 = tosa.identity %1955 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1957 = tosa.reshape %1956 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1958 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1959 = tosa.transpose %arg162, %1958 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1960 = tosa.reshape %1957 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_344 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1961 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1960, %1959 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_344 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1962 = tosa.reshape %1961 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1963 = tosa.add %1863, %1962 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1964 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_345 = arith.constant 2 : i32
    %1965 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1963 : tensor<1x40x4096xf32>) outs(%1964 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_345 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1966 = tosa.reduce_sum %1965 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1967 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1968 = tosa.reciprocal %1967 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1969 = tosa.mul %1968, %1966 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1970 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1971 = tosa.add %1969, %1970 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1972 = tosa.rsqrt %1971 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1973 = tosa.mul %1963, %1972 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1974 = tosa.reshape %arg163 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1975 = tosa.mul %1974, %1973 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1976 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1977 = tosa.transpose %arg164, %1976 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1978 = tosa.reshape %1975 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_346 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1979 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1978, %1977 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_346 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1980 = tosa.reshape %1979 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1981 = tosa.sigmoid %1980 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1982 = tosa.mul %1980, %1981 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1983 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1984 = tosa.transpose %arg165, %1983 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1985 = tosa.reshape %1975 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_347 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %1986 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1985, %1984 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_347 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %1987 = tosa.reshape %1986 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1988 = tosa.mul %1982, %1987 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1989 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1990 = tosa.transpose %arg166, %1989 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1991 = tosa.reshape %1988 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_348 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %1992 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1991, %1990 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_348 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %1993 = tosa.reshape %1992 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1994 = tosa.add %1963, %1993 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %1995 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_349 = arith.constant 2 : i32
    %1996 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1994 : tensor<1x40x4096xf32>) outs(%1995 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_349 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %1997 = tosa.reduce_sum %1996 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1998 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %1999 = tosa.reciprocal %1998 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2000 = tosa.mul %1999, %1997 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2001 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2002 = tosa.add %2000, %2001 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2003 = tosa.rsqrt %2002 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2004 = tosa.mul %1994, %2003 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2005 = tosa.reshape %arg167 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2006 = tosa.mul %2005, %2004 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2007 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2008 = tosa.transpose %arg168, %2007 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2009 = tosa.reshape %2006 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_350 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2010 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2009, %2008 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_350 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2011 = tosa.reshape %2010 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2012 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2013 = tosa.transpose %arg169, %2012 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2014 = tosa.reshape %2006 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_351 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2015 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2014, %2013 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_351 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2016 = tosa.reshape %2015 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2017 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2018 = tosa.transpose %arg170, %2017 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2019 = tosa.reshape %2006 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_352 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2020 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2019, %2018 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_352 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2021 = tosa.reshape %2020 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2022 = tosa.reshape %2011 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2023 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2024 = tosa.transpose %2022, %2023 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2025 = tosa.reshape %2016 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2026 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2027 = tosa.transpose %2025, %2026 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2028 = tosa.reshape %2021 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2029 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2030 = tosa.transpose %2028, %2029 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_353 = tensor.extract_slice %arg171[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_354 = tensor.extract_slice %extracted_slice_353[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_355 = tensor.extract_slice %extracted_slice_354[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_356 = tensor.extract_slice %arg172[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_357 = tensor.extract_slice %extracted_slice_356[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_358 = tensor.extract_slice %extracted_slice_357[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2031 = tensor.empty() : tensor<1x40x128xf32>
    %2032 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_355 : tensor<1x1x40x128xf32>) outs(%2031 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2033 = tensor.empty() : tensor<40x128xf32>
    %2034 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2032 : tensor<1x40x128xf32>) outs(%2033 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2035 = tensor.empty() : tensor<1x40x128xf32>
    %2036 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_358 : tensor<1x1x40x128xf32>) outs(%2035 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2037 = tensor.empty() : tensor<40x128xf32>
    %2038 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2036 : tensor<1x40x128xf32>) outs(%2037 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2039 = tensor.empty() : tensor<1x40x128xf32>
    %2040 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2039 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2034[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2041 = tosa.reshape %2040 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %2042 = tensor.empty() : tensor<1x40x128xf32>
    %2043 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2042 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2038[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2044 = tosa.reshape %2043 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2045 = tosa.mul %2024, %2041 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_359 = tensor.extract_slice %2024[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_360 = tensor.extract_slice %2024[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2046 = tensor.empty() : tensor<1x32x40x64xf32>
    %2047 = linalg.negf ins(%extracted_slice_360 : tensor<1x32x40x64xf32>) outs(%2046 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2048 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_361 = tensor.insert_slice %2047 into %2048[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_362 = tensor.insert_slice %extracted_slice_359 into %inserted_slice_361[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2049 = tosa.mul %inserted_slice_362, %2044 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2050 = tosa.add %2045, %2049 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2051 = tosa.mul %2027, %2041 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_363 = tensor.extract_slice %2027[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_364 = tensor.extract_slice %2027[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2052 = tensor.empty() : tensor<1x32x40x64xf32>
    %2053 = linalg.negf ins(%extracted_slice_364 : tensor<1x32x40x64xf32>) outs(%2052 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2054 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_365 = tensor.insert_slice %2053 into %2054[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_366 = tensor.insert_slice %extracted_slice_363 into %inserted_slice_365[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2055 = tosa.mul %inserted_slice_366, %2044 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2056 = tosa.add %2051, %2055 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2057 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2058 = tosa.transpose %2056, %2057 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2059 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2060 = tosa.add %2050, %2059 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2061 = tosa.reshape %2060 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2062 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2063 = tosa.add %2058, %2062 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2064 = tosa.reshape %2063 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2065 = tosa.matmul %2061, %2064 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2066 = tosa.reshape %2065 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2067 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2068 = tosa.reciprocal %2067 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2069 = tosa.mul %2066, %2068 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2070 = tosa.add %2069, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2071 = tosa.reduce_max %2070 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2072 = tosa.sub %2070, %2071 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2073 = tosa.exp %2072 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2074 = tosa.reduce_sum %2073 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2075 = tosa.reciprocal %2074 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2076 = tosa.mul %2073, %2075 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2077 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2078 = tosa.add %2076, %2077 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2079 = tosa.reshape %2078 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2080 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2081 = tosa.add %2030, %2080 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2082 = tosa.reshape %2081 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2083 = tosa.matmul %2079, %2082 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2084 = tosa.reshape %2083 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2085 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2086 = tosa.transpose %2084, %2085 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2087 = tosa.identity %2086 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2088 = tosa.reshape %2087 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2089 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2090 = tosa.transpose %arg173, %2089 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2091 = tosa.reshape %2088 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_367 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2092 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2091, %2090 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_367 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2093 = tosa.reshape %2092 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2094 = tosa.add %1994, %2093 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2095 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_368 = arith.constant 2 : i32
    %2096 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2094 : tensor<1x40x4096xf32>) outs(%2095 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_368 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2097 = tosa.reduce_sum %2096 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2098 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2099 = tosa.reciprocal %2098 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2100 = tosa.mul %2099, %2097 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2101 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2102 = tosa.add %2100, %2101 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2103 = tosa.rsqrt %2102 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2104 = tosa.mul %2094, %2103 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2105 = tosa.reshape %arg174 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2106 = tosa.mul %2105, %2104 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2107 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2108 = tosa.transpose %arg175, %2107 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2109 = tosa.reshape %2106 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_369 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2110 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2109, %2108 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_369 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2111 = tosa.reshape %2110 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2112 = tosa.sigmoid %2111 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2113 = tosa.mul %2111, %2112 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2114 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2115 = tosa.transpose %arg176, %2114 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2116 = tosa.reshape %2106 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_370 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2117 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2116, %2115 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_370 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2118 = tosa.reshape %2117 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2119 = tosa.mul %2113, %2118 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2120 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2121 = tosa.transpose %arg177, %2120 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2122 = tosa.reshape %2119 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_371 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2123 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2122, %2121 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_371 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2124 = tosa.reshape %2123 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2125 = tosa.add %2094, %2124 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2126 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_372 = arith.constant 2 : i32
    %2127 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2125 : tensor<1x40x4096xf32>) outs(%2126 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_372 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2128 = tosa.reduce_sum %2127 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2129 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2130 = tosa.reciprocal %2129 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2131 = tosa.mul %2130, %2128 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2132 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2133 = tosa.add %2131, %2132 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2134 = tosa.rsqrt %2133 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2135 = tosa.mul %2125, %2134 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2136 = tosa.reshape %arg178 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2137 = tosa.mul %2136, %2135 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2138 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2139 = tosa.transpose %arg179, %2138 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2140 = tosa.reshape %2137 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_373 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2141 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2140, %2139 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_373 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2142 = tosa.reshape %2141 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2143 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2144 = tosa.transpose %arg180, %2143 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2145 = tosa.reshape %2137 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_374 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2146 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2145, %2144 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_374 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2147 = tosa.reshape %2146 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2148 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2149 = tosa.transpose %arg181, %2148 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2150 = tosa.reshape %2137 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_375 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2151 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2150, %2149 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_375 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2152 = tosa.reshape %2151 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2153 = tosa.reshape %2142 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2154 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2155 = tosa.transpose %2153, %2154 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2156 = tosa.reshape %2147 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2157 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2158 = tosa.transpose %2156, %2157 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2159 = tosa.reshape %2152 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2160 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2161 = tosa.transpose %2159, %2160 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_376 = tensor.extract_slice %arg182[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_377 = tensor.extract_slice %extracted_slice_376[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_378 = tensor.extract_slice %extracted_slice_377[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_379 = tensor.extract_slice %arg183[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_380 = tensor.extract_slice %extracted_slice_379[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_381 = tensor.extract_slice %extracted_slice_380[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2162 = tensor.empty() : tensor<1x40x128xf32>
    %2163 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_378 : tensor<1x1x40x128xf32>) outs(%2162 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2164 = tensor.empty() : tensor<40x128xf32>
    %2165 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2163 : tensor<1x40x128xf32>) outs(%2164 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2166 = tensor.empty() : tensor<1x40x128xf32>
    %2167 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_381 : tensor<1x1x40x128xf32>) outs(%2166 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2168 = tensor.empty() : tensor<40x128xf32>
    %2169 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2167 : tensor<1x40x128xf32>) outs(%2168 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2170 = tensor.empty() : tensor<1x40x128xf32>
    %2171 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2170 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2165[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2172 = tosa.reshape %2171 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %2173 = tensor.empty() : tensor<1x40x128xf32>
    %2174 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2173 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2169[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2175 = tosa.reshape %2174 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2176 = tosa.mul %2155, %2172 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_382 = tensor.extract_slice %2155[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_383 = tensor.extract_slice %2155[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2177 = tensor.empty() : tensor<1x32x40x64xf32>
    %2178 = linalg.negf ins(%extracted_slice_383 : tensor<1x32x40x64xf32>) outs(%2177 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2179 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_384 = tensor.insert_slice %2178 into %2179[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_385 = tensor.insert_slice %extracted_slice_382 into %inserted_slice_384[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2180 = tosa.mul %inserted_slice_385, %2175 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2181 = tosa.add %2176, %2180 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2182 = tosa.mul %2158, %2172 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_386 = tensor.extract_slice %2158[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_387 = tensor.extract_slice %2158[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2183 = tensor.empty() : tensor<1x32x40x64xf32>
    %2184 = linalg.negf ins(%extracted_slice_387 : tensor<1x32x40x64xf32>) outs(%2183 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2185 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_388 = tensor.insert_slice %2184 into %2185[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_389 = tensor.insert_slice %extracted_slice_386 into %inserted_slice_388[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2186 = tosa.mul %inserted_slice_389, %2175 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2187 = tosa.add %2182, %2186 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2188 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2189 = tosa.transpose %2187, %2188 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2190 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2191 = tosa.add %2181, %2190 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2192 = tosa.reshape %2191 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2193 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2194 = tosa.add %2189, %2193 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2195 = tosa.reshape %2194 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2196 = tosa.matmul %2192, %2195 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2197 = tosa.reshape %2196 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2198 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2199 = tosa.reciprocal %2198 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2200 = tosa.mul %2197, %2199 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2201 = tosa.add %2200, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2202 = tosa.reduce_max %2201 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2203 = tosa.sub %2201, %2202 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2204 = tosa.exp %2203 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2205 = tosa.reduce_sum %2204 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2206 = tosa.reciprocal %2205 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2207 = tosa.mul %2204, %2206 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2208 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2209 = tosa.add %2207, %2208 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2210 = tosa.reshape %2209 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2211 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2212 = tosa.add %2161, %2211 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2213 = tosa.reshape %2212 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2214 = tosa.matmul %2210, %2213 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2215 = tosa.reshape %2214 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2216 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2217 = tosa.transpose %2215, %2216 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2218 = tosa.identity %2217 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2219 = tosa.reshape %2218 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2220 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2221 = tosa.transpose %arg184, %2220 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2222 = tosa.reshape %2219 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_390 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2223 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2222, %2221 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_390 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2224 = tosa.reshape %2223 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2225 = tosa.add %2125, %2224 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2226 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_391 = arith.constant 2 : i32
    %2227 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2225 : tensor<1x40x4096xf32>) outs(%2226 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_391 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2228 = tosa.reduce_sum %2227 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2229 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2230 = tosa.reciprocal %2229 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2231 = tosa.mul %2230, %2228 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2232 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2233 = tosa.add %2231, %2232 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2234 = tosa.rsqrt %2233 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2235 = tosa.mul %2225, %2234 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2236 = tosa.reshape %arg185 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2237 = tosa.mul %2236, %2235 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2238 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2239 = tosa.transpose %arg186, %2238 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2240 = tosa.reshape %2237 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_392 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2241 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2240, %2239 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_392 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2242 = tosa.reshape %2241 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2243 = tosa.sigmoid %2242 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2244 = tosa.mul %2242, %2243 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2245 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2246 = tosa.transpose %arg187, %2245 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2247 = tosa.reshape %2237 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_393 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2248 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2247, %2246 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_393 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2249 = tosa.reshape %2248 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2250 = tosa.mul %2244, %2249 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2251 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2252 = tosa.transpose %arg188, %2251 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2253 = tosa.reshape %2250 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_394 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2254 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2253, %2252 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_394 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2255 = tosa.reshape %2254 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2256 = tosa.add %2225, %2255 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2257 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_395 = arith.constant 2 : i32
    %2258 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2256 : tensor<1x40x4096xf32>) outs(%2257 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_395 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2259 = tosa.reduce_sum %2258 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2260 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2261 = tosa.reciprocal %2260 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2262 = tosa.mul %2261, %2259 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2263 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2264 = tosa.add %2262, %2263 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2265 = tosa.rsqrt %2264 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2266 = tosa.mul %2256, %2265 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2267 = tosa.reshape %arg189 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2268 = tosa.mul %2267, %2266 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2269 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2270 = tosa.transpose %arg190, %2269 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2271 = tosa.reshape %2268 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_396 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2272 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2271, %2270 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_396 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2273 = tosa.reshape %2272 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2274 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2275 = tosa.transpose %arg191, %2274 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2276 = tosa.reshape %2268 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_397 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2277 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2276, %2275 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_397 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2278 = tosa.reshape %2277 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2279 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2280 = tosa.transpose %arg192, %2279 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2281 = tosa.reshape %2268 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_398 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2282 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2281, %2280 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_398 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2283 = tosa.reshape %2282 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2284 = tosa.reshape %2273 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2285 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2286 = tosa.transpose %2284, %2285 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2287 = tosa.reshape %2278 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2288 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2289 = tosa.transpose %2287, %2288 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2290 = tosa.reshape %2283 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2291 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2292 = tosa.transpose %2290, %2291 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_399 = tensor.extract_slice %arg193[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_400 = tensor.extract_slice %extracted_slice_399[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_401 = tensor.extract_slice %extracted_slice_400[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_402 = tensor.extract_slice %arg194[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_403 = tensor.extract_slice %extracted_slice_402[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_404 = tensor.extract_slice %extracted_slice_403[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2293 = tensor.empty() : tensor<1x40x128xf32>
    %2294 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_401 : tensor<1x1x40x128xf32>) outs(%2293 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2295 = tensor.empty() : tensor<40x128xf32>
    %2296 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2294 : tensor<1x40x128xf32>) outs(%2295 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2297 = tensor.empty() : tensor<1x40x128xf32>
    %2298 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_404 : tensor<1x1x40x128xf32>) outs(%2297 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2299 = tensor.empty() : tensor<40x128xf32>
    %2300 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2298 : tensor<1x40x128xf32>) outs(%2299 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2301 = tensor.empty() : tensor<1x40x128xf32>
    %2302 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2301 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2296[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2303 = tosa.reshape %2302 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %2304 = tensor.empty() : tensor<1x40x128xf32>
    %2305 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2304 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2300[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2306 = tosa.reshape %2305 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2307 = tosa.mul %2286, %2303 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_405 = tensor.extract_slice %2286[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_406 = tensor.extract_slice %2286[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2308 = tensor.empty() : tensor<1x32x40x64xf32>
    %2309 = linalg.negf ins(%extracted_slice_406 : tensor<1x32x40x64xf32>) outs(%2308 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2310 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_407 = tensor.insert_slice %2309 into %2310[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_408 = tensor.insert_slice %extracted_slice_405 into %inserted_slice_407[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2311 = tosa.mul %inserted_slice_408, %2306 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2312 = tosa.add %2307, %2311 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2313 = tosa.mul %2289, %2303 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_409 = tensor.extract_slice %2289[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_410 = tensor.extract_slice %2289[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2314 = tensor.empty() : tensor<1x32x40x64xf32>
    %2315 = linalg.negf ins(%extracted_slice_410 : tensor<1x32x40x64xf32>) outs(%2314 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2316 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_411 = tensor.insert_slice %2315 into %2316[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_412 = tensor.insert_slice %extracted_slice_409 into %inserted_slice_411[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2317 = tosa.mul %inserted_slice_412, %2306 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2318 = tosa.add %2313, %2317 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2319 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2320 = tosa.transpose %2318, %2319 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2321 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2322 = tosa.add %2312, %2321 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2323 = tosa.reshape %2322 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2324 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2325 = tosa.add %2320, %2324 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2326 = tosa.reshape %2325 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2327 = tosa.matmul %2323, %2326 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2328 = tosa.reshape %2327 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2329 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2330 = tosa.reciprocal %2329 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2331 = tosa.mul %2328, %2330 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2332 = tosa.add %2331, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2333 = tosa.reduce_max %2332 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2334 = tosa.sub %2332, %2333 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2335 = tosa.exp %2334 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2336 = tosa.reduce_sum %2335 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2337 = tosa.reciprocal %2336 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2338 = tosa.mul %2335, %2337 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2339 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2340 = tosa.add %2338, %2339 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2341 = tosa.reshape %2340 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2342 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2343 = tosa.add %2292, %2342 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2344 = tosa.reshape %2343 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2345 = tosa.matmul %2341, %2344 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2346 = tosa.reshape %2345 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2347 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2348 = tosa.transpose %2346, %2347 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2349 = tosa.identity %2348 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2350 = tosa.reshape %2349 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2351 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2352 = tosa.transpose %arg195, %2351 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2353 = tosa.reshape %2350 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_413 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2354 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2353, %2352 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_413 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2355 = tosa.reshape %2354 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2356 = tosa.add %2256, %2355 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2357 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_414 = arith.constant 2 : i32
    %2358 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2356 : tensor<1x40x4096xf32>) outs(%2357 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_414 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2359 = tosa.reduce_sum %2358 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2360 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2361 = tosa.reciprocal %2360 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2362 = tosa.mul %2361, %2359 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2363 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2364 = tosa.add %2362, %2363 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2365 = tosa.rsqrt %2364 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2366 = tosa.mul %2356, %2365 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2367 = tosa.reshape %arg196 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2368 = tosa.mul %2367, %2366 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2369 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2370 = tosa.transpose %arg197, %2369 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2371 = tosa.reshape %2368 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_415 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2372 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2371, %2370 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_415 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2373 = tosa.reshape %2372 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2374 = tosa.sigmoid %2373 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2375 = tosa.mul %2373, %2374 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2376 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2377 = tosa.transpose %arg198, %2376 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2378 = tosa.reshape %2368 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_416 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2379 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2378, %2377 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_416 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2380 = tosa.reshape %2379 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2381 = tosa.mul %2375, %2380 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2382 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2383 = tosa.transpose %arg199, %2382 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2384 = tosa.reshape %2381 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_417 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2385 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2384, %2383 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_417 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2386 = tosa.reshape %2385 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2387 = tosa.add %2356, %2386 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2388 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_418 = arith.constant 2 : i32
    %2389 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2387 : tensor<1x40x4096xf32>) outs(%2388 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_418 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2390 = tosa.reduce_sum %2389 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2391 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2392 = tosa.reciprocal %2391 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2393 = tosa.mul %2392, %2390 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2394 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2395 = tosa.add %2393, %2394 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2396 = tosa.rsqrt %2395 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2397 = tosa.mul %2387, %2396 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2398 = tosa.reshape %arg200 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2399 = tosa.mul %2398, %2397 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2400 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2401 = tosa.transpose %arg201, %2400 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2402 = tosa.reshape %2399 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_419 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2403 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2402, %2401 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_419 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2404 = tosa.reshape %2403 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2405 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2406 = tosa.transpose %arg202, %2405 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2407 = tosa.reshape %2399 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_420 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2408 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2407, %2406 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_420 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2409 = tosa.reshape %2408 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2410 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2411 = tosa.transpose %arg203, %2410 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2412 = tosa.reshape %2399 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_421 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2413 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2412, %2411 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_421 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2414 = tosa.reshape %2413 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2415 = tosa.reshape %2404 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2416 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2417 = tosa.transpose %2415, %2416 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2418 = tosa.reshape %2409 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2419 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2420 = tosa.transpose %2418, %2419 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2421 = tosa.reshape %2414 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2422 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2423 = tosa.transpose %2421, %2422 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_422 = tensor.extract_slice %arg204[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_423 = tensor.extract_slice %extracted_slice_422[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_424 = tensor.extract_slice %extracted_slice_423[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_425 = tensor.extract_slice %arg205[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_426 = tensor.extract_slice %extracted_slice_425[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_427 = tensor.extract_slice %extracted_slice_426[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2424 = tensor.empty() : tensor<1x40x128xf32>
    %2425 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_424 : tensor<1x1x40x128xf32>) outs(%2424 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2426 = tensor.empty() : tensor<40x128xf32>
    %2427 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2425 : tensor<1x40x128xf32>) outs(%2426 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2428 = tensor.empty() : tensor<1x40x128xf32>
    %2429 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_427 : tensor<1x1x40x128xf32>) outs(%2428 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2430 = tensor.empty() : tensor<40x128xf32>
    %2431 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2429 : tensor<1x40x128xf32>) outs(%2430 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2432 = tensor.empty() : tensor<1x40x128xf32>
    %2433 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2432 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2427[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2434 = tosa.reshape %2433 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %2435 = tensor.empty() : tensor<1x40x128xf32>
    %2436 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2435 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2431[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2437 = tosa.reshape %2436 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2438 = tosa.mul %2417, %2434 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_428 = tensor.extract_slice %2417[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_429 = tensor.extract_slice %2417[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2439 = tensor.empty() : tensor<1x32x40x64xf32>
    %2440 = linalg.negf ins(%extracted_slice_429 : tensor<1x32x40x64xf32>) outs(%2439 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2441 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_430 = tensor.insert_slice %2440 into %2441[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_431 = tensor.insert_slice %extracted_slice_428 into %inserted_slice_430[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2442 = tosa.mul %inserted_slice_431, %2437 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2443 = tosa.add %2438, %2442 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2444 = tosa.mul %2420, %2434 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_432 = tensor.extract_slice %2420[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_433 = tensor.extract_slice %2420[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2445 = tensor.empty() : tensor<1x32x40x64xf32>
    %2446 = linalg.negf ins(%extracted_slice_433 : tensor<1x32x40x64xf32>) outs(%2445 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2447 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_434 = tensor.insert_slice %2446 into %2447[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_435 = tensor.insert_slice %extracted_slice_432 into %inserted_slice_434[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2448 = tosa.mul %inserted_slice_435, %2437 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2449 = tosa.add %2444, %2448 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2450 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2451 = tosa.transpose %2449, %2450 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2452 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2453 = tosa.add %2443, %2452 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2454 = tosa.reshape %2453 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2455 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2456 = tosa.add %2451, %2455 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2457 = tosa.reshape %2456 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2458 = tosa.matmul %2454, %2457 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2459 = tosa.reshape %2458 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2460 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2461 = tosa.reciprocal %2460 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2462 = tosa.mul %2459, %2461 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2463 = tosa.add %2462, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2464 = tosa.reduce_max %2463 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2465 = tosa.sub %2463, %2464 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2466 = tosa.exp %2465 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2467 = tosa.reduce_sum %2466 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2468 = tosa.reciprocal %2467 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2469 = tosa.mul %2466, %2468 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2470 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2471 = tosa.add %2469, %2470 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2472 = tosa.reshape %2471 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2473 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2474 = tosa.add %2423, %2473 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2475 = tosa.reshape %2474 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2476 = tosa.matmul %2472, %2475 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2477 = tosa.reshape %2476 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2478 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2479 = tosa.transpose %2477, %2478 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2480 = tosa.identity %2479 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2481 = tosa.reshape %2480 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2482 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2483 = tosa.transpose %arg206, %2482 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2484 = tosa.reshape %2481 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_436 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2485 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2484, %2483 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_436 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2486 = tosa.reshape %2485 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2487 = tosa.add %2387, %2486 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2488 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_437 = arith.constant 2 : i32
    %2489 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2487 : tensor<1x40x4096xf32>) outs(%2488 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_437 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2490 = tosa.reduce_sum %2489 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2491 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2492 = tosa.reciprocal %2491 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2493 = tosa.mul %2492, %2490 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2494 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2495 = tosa.add %2493, %2494 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2496 = tosa.rsqrt %2495 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2497 = tosa.mul %2487, %2496 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2498 = tosa.reshape %arg207 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2499 = tosa.mul %2498, %2497 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2500 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2501 = tosa.transpose %arg208, %2500 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2502 = tosa.reshape %2499 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_438 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2503 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2502, %2501 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_438 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2504 = tosa.reshape %2503 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2505 = tosa.sigmoid %2504 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2506 = tosa.mul %2504, %2505 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2507 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2508 = tosa.transpose %arg209, %2507 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2509 = tosa.reshape %2499 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_439 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2510 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2509, %2508 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_439 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2511 = tosa.reshape %2510 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2512 = tosa.mul %2506, %2511 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2513 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2514 = tosa.transpose %arg210, %2513 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2515 = tosa.reshape %2512 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_440 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2516 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2515, %2514 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_440 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2517 = tosa.reshape %2516 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2518 = tosa.add %2487, %2517 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2519 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_441 = arith.constant 2 : i32
    %2520 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2518 : tensor<1x40x4096xf32>) outs(%2519 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_441 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2521 = tosa.reduce_sum %2520 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2522 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2523 = tosa.reciprocal %2522 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2524 = tosa.mul %2523, %2521 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2525 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2526 = tosa.add %2524, %2525 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2527 = tosa.rsqrt %2526 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2528 = tosa.mul %2518, %2527 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2529 = tosa.reshape %arg211 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2530 = tosa.mul %2529, %2528 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2531 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2532 = tosa.transpose %arg212, %2531 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2533 = tosa.reshape %2530 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_442 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2534 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2533, %2532 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_442 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2535 = tosa.reshape %2534 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2536 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2537 = tosa.transpose %arg213, %2536 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2538 = tosa.reshape %2530 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_443 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2539 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2538, %2537 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_443 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2540 = tosa.reshape %2539 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2541 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2542 = tosa.transpose %arg214, %2541 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2543 = tosa.reshape %2530 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_444 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2544 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2543, %2542 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_444 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2545 = tosa.reshape %2544 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2546 = tosa.reshape %2535 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2547 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2548 = tosa.transpose %2546, %2547 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2549 = tosa.reshape %2540 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2550 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2551 = tosa.transpose %2549, %2550 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2552 = tosa.reshape %2545 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2553 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2554 = tosa.transpose %2552, %2553 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_445 = tensor.extract_slice %arg215[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_446 = tensor.extract_slice %extracted_slice_445[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_447 = tensor.extract_slice %extracted_slice_446[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_448 = tensor.extract_slice %arg216[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_449 = tensor.extract_slice %extracted_slice_448[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_450 = tensor.extract_slice %extracted_slice_449[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2555 = tensor.empty() : tensor<1x40x128xf32>
    %2556 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_447 : tensor<1x1x40x128xf32>) outs(%2555 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2557 = tensor.empty() : tensor<40x128xf32>
    %2558 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2556 : tensor<1x40x128xf32>) outs(%2557 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2559 = tensor.empty() : tensor<1x40x128xf32>
    %2560 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_450 : tensor<1x1x40x128xf32>) outs(%2559 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2561 = tensor.empty() : tensor<40x128xf32>
    %2562 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2560 : tensor<1x40x128xf32>) outs(%2561 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2563 = tensor.empty() : tensor<1x40x128xf32>
    %2564 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2563 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2558[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2565 = tosa.reshape %2564 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %2566 = tensor.empty() : tensor<1x40x128xf32>
    %2567 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2566 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2562[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2568 = tosa.reshape %2567 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2569 = tosa.mul %2548, %2565 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_451 = tensor.extract_slice %2548[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_452 = tensor.extract_slice %2548[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2570 = tensor.empty() : tensor<1x32x40x64xf32>
    %2571 = linalg.negf ins(%extracted_slice_452 : tensor<1x32x40x64xf32>) outs(%2570 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2572 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_453 = tensor.insert_slice %2571 into %2572[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_454 = tensor.insert_slice %extracted_slice_451 into %inserted_slice_453[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2573 = tosa.mul %inserted_slice_454, %2568 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2574 = tosa.add %2569, %2573 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2575 = tosa.mul %2551, %2565 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_455 = tensor.extract_slice %2551[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_456 = tensor.extract_slice %2551[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2576 = tensor.empty() : tensor<1x32x40x64xf32>
    %2577 = linalg.negf ins(%extracted_slice_456 : tensor<1x32x40x64xf32>) outs(%2576 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2578 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_457 = tensor.insert_slice %2577 into %2578[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_458 = tensor.insert_slice %extracted_slice_455 into %inserted_slice_457[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2579 = tosa.mul %inserted_slice_458, %2568 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2580 = tosa.add %2575, %2579 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2581 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2582 = tosa.transpose %2580, %2581 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2583 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2584 = tosa.add %2574, %2583 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2585 = tosa.reshape %2584 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2586 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2587 = tosa.add %2582, %2586 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2588 = tosa.reshape %2587 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2589 = tosa.matmul %2585, %2588 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2590 = tosa.reshape %2589 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2591 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2592 = tosa.reciprocal %2591 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2593 = tosa.mul %2590, %2592 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2594 = tosa.add %2593, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2595 = tosa.reduce_max %2594 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2596 = tosa.sub %2594, %2595 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2597 = tosa.exp %2596 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2598 = tosa.reduce_sum %2597 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2599 = tosa.reciprocal %2598 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2600 = tosa.mul %2597, %2599 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2601 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2602 = tosa.add %2600, %2601 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2603 = tosa.reshape %2602 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2604 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2605 = tosa.add %2554, %2604 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2606 = tosa.reshape %2605 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2607 = tosa.matmul %2603, %2606 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2608 = tosa.reshape %2607 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2609 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2610 = tosa.transpose %2608, %2609 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2611 = tosa.identity %2610 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2612 = tosa.reshape %2611 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2613 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2614 = tosa.transpose %arg217, %2613 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2615 = tosa.reshape %2612 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_459 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2616 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2615, %2614 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_459 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2617 = tosa.reshape %2616 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2618 = tosa.add %2518, %2617 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2619 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_460 = arith.constant 2 : i32
    %2620 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2618 : tensor<1x40x4096xf32>) outs(%2619 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_460 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2621 = tosa.reduce_sum %2620 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2622 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2623 = tosa.reciprocal %2622 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2624 = tosa.mul %2623, %2621 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2625 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2626 = tosa.add %2624, %2625 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2627 = tosa.rsqrt %2626 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2628 = tosa.mul %2618, %2627 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2629 = tosa.reshape %arg218 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2630 = tosa.mul %2629, %2628 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2631 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2632 = tosa.transpose %arg219, %2631 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2633 = tosa.reshape %2630 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_461 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2634 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2633, %2632 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_461 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2635 = tosa.reshape %2634 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2636 = tosa.sigmoid %2635 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2637 = tosa.mul %2635, %2636 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2638 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2639 = tosa.transpose %arg220, %2638 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2640 = tosa.reshape %2630 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_462 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2641 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2640, %2639 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_462 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2642 = tosa.reshape %2641 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2643 = tosa.mul %2637, %2642 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2644 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2645 = tosa.transpose %arg221, %2644 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2646 = tosa.reshape %2643 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_463 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2647 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2646, %2645 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_463 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2648 = tosa.reshape %2647 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2649 = tosa.add %2618, %2648 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2650 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_464 = arith.constant 2 : i32
    %2651 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2649 : tensor<1x40x4096xf32>) outs(%2650 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_464 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2652 = tosa.reduce_sum %2651 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2653 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2654 = tosa.reciprocal %2653 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2655 = tosa.mul %2654, %2652 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2656 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2657 = tosa.add %2655, %2656 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2658 = tosa.rsqrt %2657 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2659 = tosa.mul %2649, %2658 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2660 = tosa.reshape %arg222 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2661 = tosa.mul %2660, %2659 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2662 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2663 = tosa.transpose %arg223, %2662 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2664 = tosa.reshape %2661 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_465 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2665 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2664, %2663 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_465 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2666 = tosa.reshape %2665 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2667 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2668 = tosa.transpose %arg224, %2667 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2669 = tosa.reshape %2661 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_466 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2670 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2669, %2668 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_466 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2671 = tosa.reshape %2670 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2672 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2673 = tosa.transpose %arg225, %2672 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2674 = tosa.reshape %2661 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_467 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2675 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2674, %2673 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_467 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2676 = tosa.reshape %2675 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2677 = tosa.reshape %2666 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2678 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2679 = tosa.transpose %2677, %2678 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2680 = tosa.reshape %2671 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2681 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2682 = tosa.transpose %2680, %2681 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2683 = tosa.reshape %2676 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2684 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2685 = tosa.transpose %2683, %2684 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_468 = tensor.extract_slice %arg226[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_469 = tensor.extract_slice %extracted_slice_468[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_470 = tensor.extract_slice %extracted_slice_469[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_471 = tensor.extract_slice %arg227[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_472 = tensor.extract_slice %extracted_slice_471[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_473 = tensor.extract_slice %extracted_slice_472[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2686 = tensor.empty() : tensor<1x40x128xf32>
    %2687 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_470 : tensor<1x1x40x128xf32>) outs(%2686 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2688 = tensor.empty() : tensor<40x128xf32>
    %2689 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2687 : tensor<1x40x128xf32>) outs(%2688 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2690 = tensor.empty() : tensor<1x40x128xf32>
    %2691 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_473 : tensor<1x1x40x128xf32>) outs(%2690 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2692 = tensor.empty() : tensor<40x128xf32>
    %2693 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2691 : tensor<1x40x128xf32>) outs(%2692 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2694 = tensor.empty() : tensor<1x40x128xf32>
    %2695 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2694 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2689[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2696 = tosa.reshape %2695 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %2697 = tensor.empty() : tensor<1x40x128xf32>
    %2698 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2697 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2693[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2699 = tosa.reshape %2698 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2700 = tosa.mul %2679, %2696 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_474 = tensor.extract_slice %2679[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_475 = tensor.extract_slice %2679[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2701 = tensor.empty() : tensor<1x32x40x64xf32>
    %2702 = linalg.negf ins(%extracted_slice_475 : tensor<1x32x40x64xf32>) outs(%2701 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2703 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_476 = tensor.insert_slice %2702 into %2703[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_477 = tensor.insert_slice %extracted_slice_474 into %inserted_slice_476[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2704 = tosa.mul %inserted_slice_477, %2699 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2705 = tosa.add %2700, %2704 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2706 = tosa.mul %2682, %2696 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_478 = tensor.extract_slice %2682[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_479 = tensor.extract_slice %2682[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2707 = tensor.empty() : tensor<1x32x40x64xf32>
    %2708 = linalg.negf ins(%extracted_slice_479 : tensor<1x32x40x64xf32>) outs(%2707 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2709 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_480 = tensor.insert_slice %2708 into %2709[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_481 = tensor.insert_slice %extracted_slice_478 into %inserted_slice_480[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2710 = tosa.mul %inserted_slice_481, %2699 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2711 = tosa.add %2706, %2710 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2712 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2713 = tosa.transpose %2711, %2712 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2714 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2715 = tosa.add %2705, %2714 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2716 = tosa.reshape %2715 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2717 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2718 = tosa.add %2713, %2717 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2719 = tosa.reshape %2718 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2720 = tosa.matmul %2716, %2719 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2721 = tosa.reshape %2720 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2722 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2723 = tosa.reciprocal %2722 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2724 = tosa.mul %2721, %2723 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2725 = tosa.add %2724, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2726 = tosa.reduce_max %2725 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2727 = tosa.sub %2725, %2726 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2728 = tosa.exp %2727 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2729 = tosa.reduce_sum %2728 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2730 = tosa.reciprocal %2729 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2731 = tosa.mul %2728, %2730 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2732 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2733 = tosa.add %2731, %2732 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2734 = tosa.reshape %2733 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2735 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2736 = tosa.add %2685, %2735 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2737 = tosa.reshape %2736 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2738 = tosa.matmul %2734, %2737 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2739 = tosa.reshape %2738 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2740 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2741 = tosa.transpose %2739, %2740 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2742 = tosa.identity %2741 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2743 = tosa.reshape %2742 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2744 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2745 = tosa.transpose %arg228, %2744 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2746 = tosa.reshape %2743 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_482 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2747 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2746, %2745 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_482 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2748 = tosa.reshape %2747 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2749 = tosa.add %2649, %2748 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2750 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_483 = arith.constant 2 : i32
    %2751 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2749 : tensor<1x40x4096xf32>) outs(%2750 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_483 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2752 = tosa.reduce_sum %2751 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2753 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2754 = tosa.reciprocal %2753 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2755 = tosa.mul %2754, %2752 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2756 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2757 = tosa.add %2755, %2756 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2758 = tosa.rsqrt %2757 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2759 = tosa.mul %2749, %2758 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2760 = tosa.reshape %arg229 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2761 = tosa.mul %2760, %2759 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2762 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2763 = tosa.transpose %arg230, %2762 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2764 = tosa.reshape %2761 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_484 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2765 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2764, %2763 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_484 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2766 = tosa.reshape %2765 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2767 = tosa.sigmoid %2766 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2768 = tosa.mul %2766, %2767 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2769 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2770 = tosa.transpose %arg231, %2769 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2771 = tosa.reshape %2761 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_485 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2772 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2771, %2770 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_485 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2773 = tosa.reshape %2772 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2774 = tosa.mul %2768, %2773 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2775 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2776 = tosa.transpose %arg232, %2775 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2777 = tosa.reshape %2774 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_486 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2778 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2777, %2776 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_486 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2779 = tosa.reshape %2778 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2780 = tosa.add %2749, %2779 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2781 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_487 = arith.constant 2 : i32
    %2782 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2780 : tensor<1x40x4096xf32>) outs(%2781 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_487 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2783 = tosa.reduce_sum %2782 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2784 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2785 = tosa.reciprocal %2784 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2786 = tosa.mul %2785, %2783 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2787 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2788 = tosa.add %2786, %2787 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2789 = tosa.rsqrt %2788 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2790 = tosa.mul %2780, %2789 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2791 = tosa.reshape %arg233 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2792 = tosa.mul %2791, %2790 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2793 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2794 = tosa.transpose %arg234, %2793 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2795 = tosa.reshape %2792 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_488 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2796 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2795, %2794 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_488 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2797 = tosa.reshape %2796 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2798 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2799 = tosa.transpose %arg235, %2798 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2800 = tosa.reshape %2792 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_489 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2801 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2800, %2799 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_489 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2802 = tosa.reshape %2801 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2803 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2804 = tosa.transpose %arg236, %2803 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2805 = tosa.reshape %2792 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_490 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2806 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2805, %2804 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_490 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2807 = tosa.reshape %2806 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2808 = tosa.reshape %2797 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2809 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2810 = tosa.transpose %2808, %2809 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2811 = tosa.reshape %2802 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2812 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2813 = tosa.transpose %2811, %2812 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2814 = tosa.reshape %2807 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2815 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2816 = tosa.transpose %2814, %2815 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_491 = tensor.extract_slice %arg237[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_492 = tensor.extract_slice %extracted_slice_491[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_493 = tensor.extract_slice %extracted_slice_492[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_494 = tensor.extract_slice %arg238[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_495 = tensor.extract_slice %extracted_slice_494[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_496 = tensor.extract_slice %extracted_slice_495[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2817 = tensor.empty() : tensor<1x40x128xf32>
    %2818 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_493 : tensor<1x1x40x128xf32>) outs(%2817 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2819 = tensor.empty() : tensor<40x128xf32>
    %2820 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2818 : tensor<1x40x128xf32>) outs(%2819 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2821 = tensor.empty() : tensor<1x40x128xf32>
    %2822 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_496 : tensor<1x1x40x128xf32>) outs(%2821 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2823 = tensor.empty() : tensor<40x128xf32>
    %2824 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2822 : tensor<1x40x128xf32>) outs(%2823 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2825 = tensor.empty() : tensor<1x40x128xf32>
    %2826 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2825 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2820[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2827 = tosa.reshape %2826 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %2828 = tensor.empty() : tensor<1x40x128xf32>
    %2829 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2828 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2824[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2830 = tosa.reshape %2829 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2831 = tosa.mul %2810, %2827 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_497 = tensor.extract_slice %2810[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_498 = tensor.extract_slice %2810[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2832 = tensor.empty() : tensor<1x32x40x64xf32>
    %2833 = linalg.negf ins(%extracted_slice_498 : tensor<1x32x40x64xf32>) outs(%2832 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2834 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_499 = tensor.insert_slice %2833 into %2834[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_500 = tensor.insert_slice %extracted_slice_497 into %inserted_slice_499[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2835 = tosa.mul %inserted_slice_500, %2830 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2836 = tosa.add %2831, %2835 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2837 = tosa.mul %2813, %2827 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_501 = tensor.extract_slice %2813[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_502 = tensor.extract_slice %2813[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2838 = tensor.empty() : tensor<1x32x40x64xf32>
    %2839 = linalg.negf ins(%extracted_slice_502 : tensor<1x32x40x64xf32>) outs(%2838 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2840 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_503 = tensor.insert_slice %2839 into %2840[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_504 = tensor.insert_slice %extracted_slice_501 into %inserted_slice_503[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2841 = tosa.mul %inserted_slice_504, %2830 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2842 = tosa.add %2837, %2841 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2843 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2844 = tosa.transpose %2842, %2843 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2845 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2846 = tosa.add %2836, %2845 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2847 = tosa.reshape %2846 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2848 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2849 = tosa.add %2844, %2848 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2850 = tosa.reshape %2849 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2851 = tosa.matmul %2847, %2850 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2852 = tosa.reshape %2851 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2853 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2854 = tosa.reciprocal %2853 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2855 = tosa.mul %2852, %2854 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2856 = tosa.add %2855, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2857 = tosa.reduce_max %2856 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2858 = tosa.sub %2856, %2857 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2859 = tosa.exp %2858 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2860 = tosa.reduce_sum %2859 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2861 = tosa.reciprocal %2860 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2862 = tosa.mul %2859, %2861 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2863 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2864 = tosa.add %2862, %2863 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2865 = tosa.reshape %2864 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2866 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2867 = tosa.add %2816, %2866 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2868 = tosa.reshape %2867 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2869 = tosa.matmul %2865, %2868 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2870 = tosa.reshape %2869 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2871 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2872 = tosa.transpose %2870, %2871 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2873 = tosa.identity %2872 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2874 = tosa.reshape %2873 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2875 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2876 = tosa.transpose %arg239, %2875 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2877 = tosa.reshape %2874 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_505 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2878 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2877, %2876 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_505 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2879 = tosa.reshape %2878 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2880 = tosa.add %2780, %2879 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2881 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_506 = arith.constant 2 : i32
    %2882 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2880 : tensor<1x40x4096xf32>) outs(%2881 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_506 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2883 = tosa.reduce_sum %2882 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2884 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2885 = tosa.reciprocal %2884 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2886 = tosa.mul %2885, %2883 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2887 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2888 = tosa.add %2886, %2887 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2889 = tosa.rsqrt %2888 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2890 = tosa.mul %2880, %2889 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2891 = tosa.reshape %arg240 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2892 = tosa.mul %2891, %2890 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2893 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2894 = tosa.transpose %arg241, %2893 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2895 = tosa.reshape %2892 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_507 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2896 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2895, %2894 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_507 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2897 = tosa.reshape %2896 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2898 = tosa.sigmoid %2897 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2899 = tosa.mul %2897, %2898 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2900 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2901 = tosa.transpose %arg242, %2900 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2902 = tosa.reshape %2892 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_508 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %2903 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2902, %2901 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_508 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %2904 = tosa.reshape %2903 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2905 = tosa.mul %2899, %2904 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2906 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2907 = tosa.transpose %arg243, %2906 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2908 = tosa.reshape %2905 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_509 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2909 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2908, %2907 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_509 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2910 = tosa.reshape %2909 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2911 = tosa.add %2880, %2910 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %2912 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_510 = arith.constant 2 : i32
    %2913 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2911 : tensor<1x40x4096xf32>) outs(%2912 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_510 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %2914 = tosa.reduce_sum %2913 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2915 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2916 = tosa.reciprocal %2915 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2917 = tosa.mul %2916, %2914 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2918 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2919 = tosa.add %2917, %2918 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2920 = tosa.rsqrt %2919 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2921 = tosa.mul %2911, %2920 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2922 = tosa.reshape %arg244 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2923 = tosa.mul %2922, %2921 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2924 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2925 = tosa.transpose %arg245, %2924 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2926 = tosa.reshape %2923 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_511 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2927 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2926, %2925 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_511 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2928 = tosa.reshape %2927 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2929 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2930 = tosa.transpose %arg246, %2929 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2931 = tosa.reshape %2923 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_512 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2932 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2931, %2930 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_512 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2933 = tosa.reshape %2932 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2934 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2935 = tosa.transpose %arg247, %2934 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2936 = tosa.reshape %2923 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_513 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %2937 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2936, %2935 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_513 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %2938 = tosa.reshape %2937 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2939 = tosa.reshape %2928 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2940 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2941 = tosa.transpose %2939, %2940 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2942 = tosa.reshape %2933 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2943 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2944 = tosa.transpose %2942, %2943 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2945 = tosa.reshape %2938 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2946 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2947 = tosa.transpose %2945, %2946 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_514 = tensor.extract_slice %arg248[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_515 = tensor.extract_slice %extracted_slice_514[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_516 = tensor.extract_slice %extracted_slice_515[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_517 = tensor.extract_slice %arg249[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_518 = tensor.extract_slice %extracted_slice_517[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_519 = tensor.extract_slice %extracted_slice_518[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %2948 = tensor.empty() : tensor<1x40x128xf32>
    %2949 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_516 : tensor<1x1x40x128xf32>) outs(%2948 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2950 = tensor.empty() : tensor<40x128xf32>
    %2951 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2949 : tensor<1x40x128xf32>) outs(%2950 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2952 = tensor.empty() : tensor<1x40x128xf32>
    %2953 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_519 : tensor<1x1x40x128xf32>) outs(%2952 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %2954 = tensor.empty() : tensor<40x128xf32>
    %2955 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%2953 : tensor<1x40x128xf32>) outs(%2954 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %2956 = tensor.empty() : tensor<1x40x128xf32>
    %2957 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2956 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2951[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2958 = tosa.reshape %2957 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %2959 = tensor.empty() : tensor<1x40x128xf32>
    %2960 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%2959 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %2955[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %2961 = tosa.reshape %2960 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2962 = tosa.mul %2941, %2958 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_520 = tensor.extract_slice %2941[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_521 = tensor.extract_slice %2941[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2963 = tensor.empty() : tensor<1x32x40x64xf32>
    %2964 = linalg.negf ins(%extracted_slice_521 : tensor<1x32x40x64xf32>) outs(%2963 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2965 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_522 = tensor.insert_slice %2964 into %2965[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_523 = tensor.insert_slice %extracted_slice_520 into %inserted_slice_522[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2966 = tosa.mul %inserted_slice_523, %2961 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2967 = tosa.add %2962, %2966 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2968 = tosa.mul %2944, %2958 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_524 = tensor.extract_slice %2944[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_525 = tensor.extract_slice %2944[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %2969 = tensor.empty() : tensor<1x32x40x64xf32>
    %2970 = linalg.negf ins(%extracted_slice_525 : tensor<1x32x40x64xf32>) outs(%2969 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2971 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_526 = tensor.insert_slice %2970 into %2971[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_527 = tensor.insert_slice %extracted_slice_524 into %inserted_slice_526[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %2972 = tosa.mul %inserted_slice_527, %2961 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2973 = tosa.add %2968, %2972 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2974 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2975 = tosa.transpose %2973, %2974 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2976 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2977 = tosa.add %2967, %2976 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2978 = tosa.reshape %2977 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2979 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2980 = tosa.add %2975, %2979 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2981 = tosa.reshape %2980 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2982 = tosa.matmul %2978, %2981 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2983 = tosa.reshape %2982 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2984 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2985 = tosa.reciprocal %2984 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2986 = tosa.mul %2983, %2985 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2987 = tosa.add %2986, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2988 = tosa.reduce_max %2987 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2989 = tosa.sub %2987, %2988 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2990 = tosa.exp %2989 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2991 = tosa.reduce_sum %2990 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2992 = tosa.reciprocal %2991 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2993 = tosa.mul %2990, %2992 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2994 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2995 = tosa.add %2993, %2994 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2996 = tosa.reshape %2995 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2997 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2998 = tosa.add %2947, %2997 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %2999 = tosa.reshape %2998 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3000 = tosa.matmul %2996, %2999 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3001 = tosa.reshape %3000 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3002 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3003 = tosa.transpose %3001, %3002 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3004 = tosa.identity %3003 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3005 = tosa.reshape %3004 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3006 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3007 = tosa.transpose %arg250, %3006 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3008 = tosa.reshape %3005 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_528 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3009 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3008, %3007 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_528 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3010 = tosa.reshape %3009 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3011 = tosa.add %2911, %3010 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3012 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_529 = arith.constant 2 : i32
    %3013 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3011 : tensor<1x40x4096xf32>) outs(%3012 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_529 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3014 = tosa.reduce_sum %3013 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3015 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3016 = tosa.reciprocal %3015 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3017 = tosa.mul %3016, %3014 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3018 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3019 = tosa.add %3017, %3018 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3020 = tosa.rsqrt %3019 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3021 = tosa.mul %3011, %3020 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3022 = tosa.reshape %arg251 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3023 = tosa.mul %3022, %3021 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3024 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3025 = tosa.transpose %arg252, %3024 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3026 = tosa.reshape %3023 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_530 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3027 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3026, %3025 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_530 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3028 = tosa.reshape %3027 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3029 = tosa.sigmoid %3028 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3030 = tosa.mul %3028, %3029 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3031 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3032 = tosa.transpose %arg253, %3031 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3033 = tosa.reshape %3023 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_531 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3034 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3033, %3032 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_531 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3035 = tosa.reshape %3034 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3036 = tosa.mul %3030, %3035 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3037 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3038 = tosa.transpose %arg254, %3037 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3039 = tosa.reshape %3036 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_532 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3040 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3039, %3038 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_532 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3041 = tosa.reshape %3040 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3042 = tosa.add %3011, %3041 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3043 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_533 = arith.constant 2 : i32
    %3044 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3042 : tensor<1x40x4096xf32>) outs(%3043 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_533 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3045 = tosa.reduce_sum %3044 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3046 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3047 = tosa.reciprocal %3046 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3048 = tosa.mul %3047, %3045 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3049 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3050 = tosa.add %3048, %3049 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3051 = tosa.rsqrt %3050 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3052 = tosa.mul %3042, %3051 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3053 = tosa.reshape %arg255 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3054 = tosa.mul %3053, %3052 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3055 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3056 = tosa.transpose %arg256, %3055 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3057 = tosa.reshape %3054 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_534 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3058 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3057, %3056 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_534 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3059 = tosa.reshape %3058 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3060 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3061 = tosa.transpose %arg257, %3060 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3062 = tosa.reshape %3054 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_535 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3063 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3062, %3061 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_535 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3064 = tosa.reshape %3063 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3065 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3066 = tosa.transpose %arg258, %3065 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3067 = tosa.reshape %3054 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_536 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3068 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3067, %3066 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_536 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3069 = tosa.reshape %3068 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3070 = tosa.reshape %3059 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3071 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3072 = tosa.transpose %3070, %3071 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3073 = tosa.reshape %3064 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3074 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3075 = tosa.transpose %3073, %3074 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3076 = tosa.reshape %3069 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3077 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3078 = tosa.transpose %3076, %3077 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_537 = tensor.extract_slice %arg259[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_538 = tensor.extract_slice %extracted_slice_537[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_539 = tensor.extract_slice %extracted_slice_538[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_540 = tensor.extract_slice %arg260[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_541 = tensor.extract_slice %extracted_slice_540[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_542 = tensor.extract_slice %extracted_slice_541[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3079 = tensor.empty() : tensor<1x40x128xf32>
    %3080 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_539 : tensor<1x1x40x128xf32>) outs(%3079 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3081 = tensor.empty() : tensor<40x128xf32>
    %3082 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3080 : tensor<1x40x128xf32>) outs(%3081 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3083 = tensor.empty() : tensor<1x40x128xf32>
    %3084 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_542 : tensor<1x1x40x128xf32>) outs(%3083 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3085 = tensor.empty() : tensor<40x128xf32>
    %3086 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3084 : tensor<1x40x128xf32>) outs(%3085 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3087 = tensor.empty() : tensor<1x40x128xf32>
    %3088 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3087 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3082[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3089 = tosa.reshape %3088 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %3090 = tensor.empty() : tensor<1x40x128xf32>
    %3091 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3090 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3086[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3092 = tosa.reshape %3091 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3093 = tosa.mul %3072, %3089 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_543 = tensor.extract_slice %3072[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_544 = tensor.extract_slice %3072[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3094 = tensor.empty() : tensor<1x32x40x64xf32>
    %3095 = linalg.negf ins(%extracted_slice_544 : tensor<1x32x40x64xf32>) outs(%3094 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3096 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_545 = tensor.insert_slice %3095 into %3096[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_546 = tensor.insert_slice %extracted_slice_543 into %inserted_slice_545[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3097 = tosa.mul %inserted_slice_546, %3092 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3098 = tosa.add %3093, %3097 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3099 = tosa.mul %3075, %3089 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_547 = tensor.extract_slice %3075[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_548 = tensor.extract_slice %3075[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3100 = tensor.empty() : tensor<1x32x40x64xf32>
    %3101 = linalg.negf ins(%extracted_slice_548 : tensor<1x32x40x64xf32>) outs(%3100 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3102 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_549 = tensor.insert_slice %3101 into %3102[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_550 = tensor.insert_slice %extracted_slice_547 into %inserted_slice_549[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3103 = tosa.mul %inserted_slice_550, %3092 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3104 = tosa.add %3099, %3103 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3105 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3106 = tosa.transpose %3104, %3105 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3107 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3108 = tosa.add %3098, %3107 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3109 = tosa.reshape %3108 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3110 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3111 = tosa.add %3106, %3110 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3112 = tosa.reshape %3111 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3113 = tosa.matmul %3109, %3112 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3114 = tosa.reshape %3113 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3115 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3116 = tosa.reciprocal %3115 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3117 = tosa.mul %3114, %3116 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3118 = tosa.add %3117, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3119 = tosa.reduce_max %3118 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3120 = tosa.sub %3118, %3119 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3121 = tosa.exp %3120 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3122 = tosa.reduce_sum %3121 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3123 = tosa.reciprocal %3122 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3124 = tosa.mul %3121, %3123 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3125 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3126 = tosa.add %3124, %3125 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3127 = tosa.reshape %3126 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3128 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3129 = tosa.add %3078, %3128 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3130 = tosa.reshape %3129 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3131 = tosa.matmul %3127, %3130 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3132 = tosa.reshape %3131 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3133 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3134 = tosa.transpose %3132, %3133 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3135 = tosa.identity %3134 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3136 = tosa.reshape %3135 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3137 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3138 = tosa.transpose %arg261, %3137 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3139 = tosa.reshape %3136 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_551 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3140 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3139, %3138 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_551 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3141 = tosa.reshape %3140 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3142 = tosa.add %3042, %3141 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3143 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_552 = arith.constant 2 : i32
    %3144 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3142 : tensor<1x40x4096xf32>) outs(%3143 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_552 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3145 = tosa.reduce_sum %3144 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3146 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3147 = tosa.reciprocal %3146 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3148 = tosa.mul %3147, %3145 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3149 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3150 = tosa.add %3148, %3149 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3151 = tosa.rsqrt %3150 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3152 = tosa.mul %3142, %3151 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3153 = tosa.reshape %arg262 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3154 = tosa.mul %3153, %3152 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3155 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3156 = tosa.transpose %arg263, %3155 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3157 = tosa.reshape %3154 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_553 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3158 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3157, %3156 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_553 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3159 = tosa.reshape %3158 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3160 = tosa.sigmoid %3159 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3161 = tosa.mul %3159, %3160 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3162 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3163 = tosa.transpose %arg264, %3162 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3164 = tosa.reshape %3154 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_554 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3165 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3164, %3163 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_554 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3166 = tosa.reshape %3165 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3167 = tosa.mul %3161, %3166 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3168 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3169 = tosa.transpose %arg265, %3168 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3170 = tosa.reshape %3167 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_555 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3171 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3170, %3169 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_555 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3172 = tosa.reshape %3171 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3173 = tosa.add %3142, %3172 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3174 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_556 = arith.constant 2 : i32
    %3175 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3173 : tensor<1x40x4096xf32>) outs(%3174 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_556 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3176 = tosa.reduce_sum %3175 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3177 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3178 = tosa.reciprocal %3177 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3179 = tosa.mul %3178, %3176 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3180 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3181 = tosa.add %3179, %3180 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3182 = tosa.rsqrt %3181 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3183 = tosa.mul %3173, %3182 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3184 = tosa.reshape %arg266 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3185 = tosa.mul %3184, %3183 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3186 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3187 = tosa.transpose %arg267, %3186 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3188 = tosa.reshape %3185 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_557 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3189 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3188, %3187 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_557 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3190 = tosa.reshape %3189 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3191 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3192 = tosa.transpose %arg268, %3191 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3193 = tosa.reshape %3185 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_558 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3194 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3193, %3192 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_558 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3195 = tosa.reshape %3194 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3196 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3197 = tosa.transpose %arg269, %3196 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3198 = tosa.reshape %3185 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_559 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3199 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3198, %3197 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_559 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3200 = tosa.reshape %3199 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3201 = tosa.reshape %3190 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3202 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3203 = tosa.transpose %3201, %3202 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3204 = tosa.reshape %3195 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3205 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3206 = tosa.transpose %3204, %3205 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3207 = tosa.reshape %3200 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3208 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3209 = tosa.transpose %3207, %3208 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_560 = tensor.extract_slice %arg270[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_561 = tensor.extract_slice %extracted_slice_560[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_562 = tensor.extract_slice %extracted_slice_561[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_563 = tensor.extract_slice %arg271[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_564 = tensor.extract_slice %extracted_slice_563[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_565 = tensor.extract_slice %extracted_slice_564[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3210 = tensor.empty() : tensor<1x40x128xf32>
    %3211 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_562 : tensor<1x1x40x128xf32>) outs(%3210 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3212 = tensor.empty() : tensor<40x128xf32>
    %3213 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3211 : tensor<1x40x128xf32>) outs(%3212 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3214 = tensor.empty() : tensor<1x40x128xf32>
    %3215 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_565 : tensor<1x1x40x128xf32>) outs(%3214 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3216 = tensor.empty() : tensor<40x128xf32>
    %3217 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3215 : tensor<1x40x128xf32>) outs(%3216 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3218 = tensor.empty() : tensor<1x40x128xf32>
    %3219 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3218 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3213[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3220 = tosa.reshape %3219 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %3221 = tensor.empty() : tensor<1x40x128xf32>
    %3222 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3221 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3217[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3223 = tosa.reshape %3222 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3224 = tosa.mul %3203, %3220 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_566 = tensor.extract_slice %3203[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_567 = tensor.extract_slice %3203[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3225 = tensor.empty() : tensor<1x32x40x64xf32>
    %3226 = linalg.negf ins(%extracted_slice_567 : tensor<1x32x40x64xf32>) outs(%3225 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3227 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_568 = tensor.insert_slice %3226 into %3227[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_569 = tensor.insert_slice %extracted_slice_566 into %inserted_slice_568[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3228 = tosa.mul %inserted_slice_569, %3223 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3229 = tosa.add %3224, %3228 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3230 = tosa.mul %3206, %3220 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_570 = tensor.extract_slice %3206[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_571 = tensor.extract_slice %3206[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3231 = tensor.empty() : tensor<1x32x40x64xf32>
    %3232 = linalg.negf ins(%extracted_slice_571 : tensor<1x32x40x64xf32>) outs(%3231 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3233 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_572 = tensor.insert_slice %3232 into %3233[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_573 = tensor.insert_slice %extracted_slice_570 into %inserted_slice_572[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3234 = tosa.mul %inserted_slice_573, %3223 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3235 = tosa.add %3230, %3234 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3236 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3237 = tosa.transpose %3235, %3236 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3238 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3239 = tosa.add %3229, %3238 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3240 = tosa.reshape %3239 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3241 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3242 = tosa.add %3237, %3241 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3243 = tosa.reshape %3242 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3244 = tosa.matmul %3240, %3243 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3245 = tosa.reshape %3244 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3246 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3247 = tosa.reciprocal %3246 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3248 = tosa.mul %3245, %3247 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3249 = tosa.add %3248, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3250 = tosa.reduce_max %3249 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3251 = tosa.sub %3249, %3250 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3252 = tosa.exp %3251 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3253 = tosa.reduce_sum %3252 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3254 = tosa.reciprocal %3253 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3255 = tosa.mul %3252, %3254 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3256 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3257 = tosa.add %3255, %3256 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3258 = tosa.reshape %3257 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3259 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3260 = tosa.add %3209, %3259 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3261 = tosa.reshape %3260 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3262 = tosa.matmul %3258, %3261 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3263 = tosa.reshape %3262 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3264 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3265 = tosa.transpose %3263, %3264 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3266 = tosa.identity %3265 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3267 = tosa.reshape %3266 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3268 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3269 = tosa.transpose %arg272, %3268 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3270 = tosa.reshape %3267 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_574 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3271 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3270, %3269 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_574 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3272 = tosa.reshape %3271 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3273 = tosa.add %3173, %3272 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3274 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_575 = arith.constant 2 : i32
    %3275 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3273 : tensor<1x40x4096xf32>) outs(%3274 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_575 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3276 = tosa.reduce_sum %3275 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3277 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3278 = tosa.reciprocal %3277 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3279 = tosa.mul %3278, %3276 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3280 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3281 = tosa.add %3279, %3280 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3282 = tosa.rsqrt %3281 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3283 = tosa.mul %3273, %3282 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3284 = tosa.reshape %arg273 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3285 = tosa.mul %3284, %3283 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3286 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3287 = tosa.transpose %arg274, %3286 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3288 = tosa.reshape %3285 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_576 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3289 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3288, %3287 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_576 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3290 = tosa.reshape %3289 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3291 = tosa.sigmoid %3290 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3292 = tosa.mul %3290, %3291 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3293 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3294 = tosa.transpose %arg275, %3293 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3295 = tosa.reshape %3285 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_577 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3296 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3295, %3294 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_577 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3297 = tosa.reshape %3296 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3298 = tosa.mul %3292, %3297 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3299 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3300 = tosa.transpose %arg276, %3299 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3301 = tosa.reshape %3298 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_578 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3302 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3301, %3300 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_578 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3303 = tosa.reshape %3302 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3304 = tosa.add %3273, %3303 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3305 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_579 = arith.constant 2 : i32
    %3306 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3304 : tensor<1x40x4096xf32>) outs(%3305 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_579 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3307 = tosa.reduce_sum %3306 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3308 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3309 = tosa.reciprocal %3308 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3310 = tosa.mul %3309, %3307 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3311 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3312 = tosa.add %3310, %3311 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3313 = tosa.rsqrt %3312 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3314 = tosa.mul %3304, %3313 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3315 = tosa.reshape %arg277 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3316 = tosa.mul %3315, %3314 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3317 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3318 = tosa.transpose %arg278, %3317 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3319 = tosa.reshape %3316 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_580 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3320 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3319, %3318 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_580 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3321 = tosa.reshape %3320 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3322 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3323 = tosa.transpose %arg279, %3322 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3324 = tosa.reshape %3316 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_581 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3325 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3324, %3323 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_581 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3326 = tosa.reshape %3325 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3327 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3328 = tosa.transpose %arg280, %3327 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3329 = tosa.reshape %3316 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_582 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3330 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3329, %3328 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_582 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3331 = tosa.reshape %3330 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3332 = tosa.reshape %3321 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3333 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3334 = tosa.transpose %3332, %3333 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3335 = tosa.reshape %3326 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3336 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3337 = tosa.transpose %3335, %3336 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3338 = tosa.reshape %3331 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3339 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3340 = tosa.transpose %3338, %3339 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_583 = tensor.extract_slice %arg281[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_584 = tensor.extract_slice %extracted_slice_583[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_585 = tensor.extract_slice %extracted_slice_584[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_586 = tensor.extract_slice %arg282[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_587 = tensor.extract_slice %extracted_slice_586[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_588 = tensor.extract_slice %extracted_slice_587[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3341 = tensor.empty() : tensor<1x40x128xf32>
    %3342 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_585 : tensor<1x1x40x128xf32>) outs(%3341 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3343 = tensor.empty() : tensor<40x128xf32>
    %3344 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3342 : tensor<1x40x128xf32>) outs(%3343 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3345 = tensor.empty() : tensor<1x40x128xf32>
    %3346 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_588 : tensor<1x1x40x128xf32>) outs(%3345 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3347 = tensor.empty() : tensor<40x128xf32>
    %3348 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3346 : tensor<1x40x128xf32>) outs(%3347 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3349 = tensor.empty() : tensor<1x40x128xf32>
    %3350 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3349 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3344[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3351 = tosa.reshape %3350 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %3352 = tensor.empty() : tensor<1x40x128xf32>
    %3353 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3352 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3348[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3354 = tosa.reshape %3353 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3355 = tosa.mul %3334, %3351 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_589 = tensor.extract_slice %3334[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_590 = tensor.extract_slice %3334[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3356 = tensor.empty() : tensor<1x32x40x64xf32>
    %3357 = linalg.negf ins(%extracted_slice_590 : tensor<1x32x40x64xf32>) outs(%3356 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3358 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_591 = tensor.insert_slice %3357 into %3358[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_592 = tensor.insert_slice %extracted_slice_589 into %inserted_slice_591[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3359 = tosa.mul %inserted_slice_592, %3354 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3360 = tosa.add %3355, %3359 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3361 = tosa.mul %3337, %3351 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_593 = tensor.extract_slice %3337[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_594 = tensor.extract_slice %3337[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3362 = tensor.empty() : tensor<1x32x40x64xf32>
    %3363 = linalg.negf ins(%extracted_slice_594 : tensor<1x32x40x64xf32>) outs(%3362 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3364 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_595 = tensor.insert_slice %3363 into %3364[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_596 = tensor.insert_slice %extracted_slice_593 into %inserted_slice_595[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3365 = tosa.mul %inserted_slice_596, %3354 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3366 = tosa.add %3361, %3365 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3367 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3368 = tosa.transpose %3366, %3367 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3369 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3370 = tosa.add %3360, %3369 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3371 = tosa.reshape %3370 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3372 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3373 = tosa.add %3368, %3372 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3374 = tosa.reshape %3373 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3375 = tosa.matmul %3371, %3374 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3376 = tosa.reshape %3375 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3377 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3378 = tosa.reciprocal %3377 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3379 = tosa.mul %3376, %3378 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3380 = tosa.add %3379, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3381 = tosa.reduce_max %3380 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3382 = tosa.sub %3380, %3381 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3383 = tosa.exp %3382 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3384 = tosa.reduce_sum %3383 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3385 = tosa.reciprocal %3384 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3386 = tosa.mul %3383, %3385 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3387 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3388 = tosa.add %3386, %3387 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3389 = tosa.reshape %3388 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3390 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3391 = tosa.add %3340, %3390 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3392 = tosa.reshape %3391 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3393 = tosa.matmul %3389, %3392 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3394 = tosa.reshape %3393 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3395 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3396 = tosa.transpose %3394, %3395 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3397 = tosa.identity %3396 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3398 = tosa.reshape %3397 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3399 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3400 = tosa.transpose %arg283, %3399 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3401 = tosa.reshape %3398 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_597 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3402 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3401, %3400 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_597 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3403 = tosa.reshape %3402 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3404 = tosa.add %3304, %3403 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3405 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_598 = arith.constant 2 : i32
    %3406 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3404 : tensor<1x40x4096xf32>) outs(%3405 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_598 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3407 = tosa.reduce_sum %3406 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3408 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3409 = tosa.reciprocal %3408 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3410 = tosa.mul %3409, %3407 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3411 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3412 = tosa.add %3410, %3411 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3413 = tosa.rsqrt %3412 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3414 = tosa.mul %3404, %3413 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3415 = tosa.reshape %arg284 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3416 = tosa.mul %3415, %3414 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3417 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3418 = tosa.transpose %arg285, %3417 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3419 = tosa.reshape %3416 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_599 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3420 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3419, %3418 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_599 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3421 = tosa.reshape %3420 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3422 = tosa.sigmoid %3421 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3423 = tosa.mul %3421, %3422 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3424 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3425 = tosa.transpose %arg286, %3424 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3426 = tosa.reshape %3416 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_600 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3427 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3426, %3425 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_600 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3428 = tosa.reshape %3427 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3429 = tosa.mul %3423, %3428 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3430 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3431 = tosa.transpose %arg287, %3430 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3432 = tosa.reshape %3429 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_601 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3433 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3432, %3431 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_601 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3434 = tosa.reshape %3433 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3435 = tosa.add %3404, %3434 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3436 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_602 = arith.constant 2 : i32
    %3437 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3435 : tensor<1x40x4096xf32>) outs(%3436 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_602 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3438 = tosa.reduce_sum %3437 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3439 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3440 = tosa.reciprocal %3439 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3441 = tosa.mul %3440, %3438 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3442 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3443 = tosa.add %3441, %3442 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3444 = tosa.rsqrt %3443 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3445 = tosa.mul %3435, %3444 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3446 = tosa.reshape %arg288 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3447 = tosa.mul %3446, %3445 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3448 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3449 = tosa.transpose %arg289, %3448 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3450 = tosa.reshape %3447 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_603 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3451 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3450, %3449 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_603 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3452 = tosa.reshape %3451 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3453 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3454 = tosa.transpose %arg290, %3453 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3455 = tosa.reshape %3447 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_604 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3456 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3455, %3454 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_604 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3457 = tosa.reshape %3456 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3458 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3459 = tosa.transpose %arg291, %3458 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3460 = tosa.reshape %3447 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_605 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3461 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3460, %3459 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_605 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3462 = tosa.reshape %3461 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3463 = tosa.reshape %3452 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3464 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3465 = tosa.transpose %3463, %3464 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3466 = tosa.reshape %3457 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3467 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3468 = tosa.transpose %3466, %3467 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3469 = tosa.reshape %3462 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3470 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3471 = tosa.transpose %3469, %3470 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_606 = tensor.extract_slice %arg292[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_607 = tensor.extract_slice %extracted_slice_606[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_608 = tensor.extract_slice %extracted_slice_607[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_609 = tensor.extract_slice %arg293[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_610 = tensor.extract_slice %extracted_slice_609[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_611 = tensor.extract_slice %extracted_slice_610[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3472 = tensor.empty() : tensor<1x40x128xf32>
    %3473 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_608 : tensor<1x1x40x128xf32>) outs(%3472 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3474 = tensor.empty() : tensor<40x128xf32>
    %3475 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3473 : tensor<1x40x128xf32>) outs(%3474 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3476 = tensor.empty() : tensor<1x40x128xf32>
    %3477 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_611 : tensor<1x1x40x128xf32>) outs(%3476 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3478 = tensor.empty() : tensor<40x128xf32>
    %3479 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3477 : tensor<1x40x128xf32>) outs(%3478 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3480 = tensor.empty() : tensor<1x40x128xf32>
    %3481 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3480 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3475[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3482 = tosa.reshape %3481 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %3483 = tensor.empty() : tensor<1x40x128xf32>
    %3484 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3483 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3479[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3485 = tosa.reshape %3484 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3486 = tosa.mul %3465, %3482 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_612 = tensor.extract_slice %3465[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_613 = tensor.extract_slice %3465[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3487 = tensor.empty() : tensor<1x32x40x64xf32>
    %3488 = linalg.negf ins(%extracted_slice_613 : tensor<1x32x40x64xf32>) outs(%3487 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3489 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_614 = tensor.insert_slice %3488 into %3489[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_615 = tensor.insert_slice %extracted_slice_612 into %inserted_slice_614[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3490 = tosa.mul %inserted_slice_615, %3485 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3491 = tosa.add %3486, %3490 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3492 = tosa.mul %3468, %3482 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_616 = tensor.extract_slice %3468[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_617 = tensor.extract_slice %3468[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3493 = tensor.empty() : tensor<1x32x40x64xf32>
    %3494 = linalg.negf ins(%extracted_slice_617 : tensor<1x32x40x64xf32>) outs(%3493 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3495 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_618 = tensor.insert_slice %3494 into %3495[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_619 = tensor.insert_slice %extracted_slice_616 into %inserted_slice_618[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3496 = tosa.mul %inserted_slice_619, %3485 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3497 = tosa.add %3492, %3496 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3498 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3499 = tosa.transpose %3497, %3498 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3500 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3501 = tosa.add %3491, %3500 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3502 = tosa.reshape %3501 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3503 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3504 = tosa.add %3499, %3503 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3505 = tosa.reshape %3504 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3506 = tosa.matmul %3502, %3505 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3507 = tosa.reshape %3506 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3508 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3509 = tosa.reciprocal %3508 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3510 = tosa.mul %3507, %3509 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3511 = tosa.add %3510, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3512 = tosa.reduce_max %3511 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3513 = tosa.sub %3511, %3512 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3514 = tosa.exp %3513 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3515 = tosa.reduce_sum %3514 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3516 = tosa.reciprocal %3515 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3517 = tosa.mul %3514, %3516 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3518 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3519 = tosa.add %3517, %3518 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3520 = tosa.reshape %3519 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3521 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3522 = tosa.add %3471, %3521 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3523 = tosa.reshape %3522 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3524 = tosa.matmul %3520, %3523 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3525 = tosa.reshape %3524 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3526 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3527 = tosa.transpose %3525, %3526 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3528 = tosa.identity %3527 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3529 = tosa.reshape %3528 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3530 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3531 = tosa.transpose %arg294, %3530 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3532 = tosa.reshape %3529 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_620 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3533 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3532, %3531 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_620 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3534 = tosa.reshape %3533 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3535 = tosa.add %3435, %3534 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3536 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_621 = arith.constant 2 : i32
    %3537 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3535 : tensor<1x40x4096xf32>) outs(%3536 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_621 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3538 = tosa.reduce_sum %3537 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3539 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3540 = tosa.reciprocal %3539 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3541 = tosa.mul %3540, %3538 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3542 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3543 = tosa.add %3541, %3542 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3544 = tosa.rsqrt %3543 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3545 = tosa.mul %3535, %3544 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3546 = tosa.reshape %arg295 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3547 = tosa.mul %3546, %3545 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3548 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3549 = tosa.transpose %arg296, %3548 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3550 = tosa.reshape %3547 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_622 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3551 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3550, %3549 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_622 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3552 = tosa.reshape %3551 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3553 = tosa.sigmoid %3552 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3554 = tosa.mul %3552, %3553 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3555 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3556 = tosa.transpose %arg297, %3555 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3557 = tosa.reshape %3547 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_623 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3558 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3557, %3556 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_623 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3559 = tosa.reshape %3558 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3560 = tosa.mul %3554, %3559 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3561 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3562 = tosa.transpose %arg298, %3561 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3563 = tosa.reshape %3560 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_624 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3564 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3563, %3562 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_624 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3565 = tosa.reshape %3564 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3566 = tosa.add %3535, %3565 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3567 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_625 = arith.constant 2 : i32
    %3568 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3566 : tensor<1x40x4096xf32>) outs(%3567 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_625 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3569 = tosa.reduce_sum %3568 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3570 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3571 = tosa.reciprocal %3570 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3572 = tosa.mul %3571, %3569 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3573 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3574 = tosa.add %3572, %3573 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3575 = tosa.rsqrt %3574 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3576 = tosa.mul %3566, %3575 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3577 = tosa.reshape %arg299 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3578 = tosa.mul %3577, %3576 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3579 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3580 = tosa.transpose %arg300, %3579 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3581 = tosa.reshape %3578 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_626 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3582 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3581, %3580 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_626 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3583 = tosa.reshape %3582 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3584 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3585 = tosa.transpose %arg301, %3584 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3586 = tosa.reshape %3578 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_627 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3587 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3586, %3585 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_627 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3588 = tosa.reshape %3587 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3589 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3590 = tosa.transpose %arg302, %3589 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3591 = tosa.reshape %3578 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_628 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3592 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3591, %3590 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_628 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3593 = tosa.reshape %3592 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3594 = tosa.reshape %3583 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3595 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3596 = tosa.transpose %3594, %3595 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3597 = tosa.reshape %3588 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3598 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3599 = tosa.transpose %3597, %3598 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3600 = tosa.reshape %3593 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3601 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3602 = tosa.transpose %3600, %3601 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_629 = tensor.extract_slice %arg303[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_630 = tensor.extract_slice %extracted_slice_629[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_631 = tensor.extract_slice %extracted_slice_630[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_632 = tensor.extract_slice %arg304[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_633 = tensor.extract_slice %extracted_slice_632[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_634 = tensor.extract_slice %extracted_slice_633[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3603 = tensor.empty() : tensor<1x40x128xf32>
    %3604 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_631 : tensor<1x1x40x128xf32>) outs(%3603 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3605 = tensor.empty() : tensor<40x128xf32>
    %3606 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3604 : tensor<1x40x128xf32>) outs(%3605 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3607 = tensor.empty() : tensor<1x40x128xf32>
    %3608 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_634 : tensor<1x1x40x128xf32>) outs(%3607 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3609 = tensor.empty() : tensor<40x128xf32>
    %3610 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3608 : tensor<1x40x128xf32>) outs(%3609 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3611 = tensor.empty() : tensor<1x40x128xf32>
    %3612 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3611 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3606[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3613 = tosa.reshape %3612 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %3614 = tensor.empty() : tensor<1x40x128xf32>
    %3615 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3614 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3610[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3616 = tosa.reshape %3615 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3617 = tosa.mul %3596, %3613 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_635 = tensor.extract_slice %3596[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_636 = tensor.extract_slice %3596[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3618 = tensor.empty() : tensor<1x32x40x64xf32>
    %3619 = linalg.negf ins(%extracted_slice_636 : tensor<1x32x40x64xf32>) outs(%3618 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3620 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_637 = tensor.insert_slice %3619 into %3620[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_638 = tensor.insert_slice %extracted_slice_635 into %inserted_slice_637[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3621 = tosa.mul %inserted_slice_638, %3616 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3622 = tosa.add %3617, %3621 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3623 = tosa.mul %3599, %3613 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_639 = tensor.extract_slice %3599[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_640 = tensor.extract_slice %3599[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3624 = tensor.empty() : tensor<1x32x40x64xf32>
    %3625 = linalg.negf ins(%extracted_slice_640 : tensor<1x32x40x64xf32>) outs(%3624 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3626 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_641 = tensor.insert_slice %3625 into %3626[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_642 = tensor.insert_slice %extracted_slice_639 into %inserted_slice_641[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3627 = tosa.mul %inserted_slice_642, %3616 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3628 = tosa.add %3623, %3627 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3629 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3630 = tosa.transpose %3628, %3629 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3631 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3632 = tosa.add %3622, %3631 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3633 = tosa.reshape %3632 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3634 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3635 = tosa.add %3630, %3634 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3636 = tosa.reshape %3635 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3637 = tosa.matmul %3633, %3636 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3638 = tosa.reshape %3637 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3639 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3640 = tosa.reciprocal %3639 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3641 = tosa.mul %3638, %3640 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3642 = tosa.add %3641, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3643 = tosa.reduce_max %3642 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3644 = tosa.sub %3642, %3643 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3645 = tosa.exp %3644 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3646 = tosa.reduce_sum %3645 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3647 = tosa.reciprocal %3646 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3648 = tosa.mul %3645, %3647 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3649 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3650 = tosa.add %3648, %3649 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3651 = tosa.reshape %3650 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3652 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3653 = tosa.add %3602, %3652 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3654 = tosa.reshape %3653 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3655 = tosa.matmul %3651, %3654 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3656 = tosa.reshape %3655 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3657 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3658 = tosa.transpose %3656, %3657 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3659 = tosa.identity %3658 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3660 = tosa.reshape %3659 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3661 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3662 = tosa.transpose %arg305, %3661 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3663 = tosa.reshape %3660 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_643 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3664 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3663, %3662 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_643 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3665 = tosa.reshape %3664 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3666 = tosa.add %3566, %3665 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3667 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_644 = arith.constant 2 : i32
    %3668 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3666 : tensor<1x40x4096xf32>) outs(%3667 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_644 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3669 = tosa.reduce_sum %3668 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3670 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3671 = tosa.reciprocal %3670 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3672 = tosa.mul %3671, %3669 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3673 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3674 = tosa.add %3672, %3673 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3675 = tosa.rsqrt %3674 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3676 = tosa.mul %3666, %3675 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3677 = tosa.reshape %arg306 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3678 = tosa.mul %3677, %3676 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3679 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3680 = tosa.transpose %arg307, %3679 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3681 = tosa.reshape %3678 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_645 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3682 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3681, %3680 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_645 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3683 = tosa.reshape %3682 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3684 = tosa.sigmoid %3683 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3685 = tosa.mul %3683, %3684 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3686 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3687 = tosa.transpose %arg308, %3686 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3688 = tosa.reshape %3678 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_646 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3689 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3688, %3687 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_646 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3690 = tosa.reshape %3689 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3691 = tosa.mul %3685, %3690 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3692 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3693 = tosa.transpose %arg309, %3692 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3694 = tosa.reshape %3691 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_647 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3695 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3694, %3693 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_647 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3696 = tosa.reshape %3695 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3697 = tosa.add %3666, %3696 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3698 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_648 = arith.constant 2 : i32
    %3699 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3697 : tensor<1x40x4096xf32>) outs(%3698 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_648 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3700 = tosa.reduce_sum %3699 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3701 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3702 = tosa.reciprocal %3701 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3703 = tosa.mul %3702, %3700 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3704 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3705 = tosa.add %3703, %3704 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3706 = tosa.rsqrt %3705 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3707 = tosa.mul %3697, %3706 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3708 = tosa.reshape %arg310 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3709 = tosa.mul %3708, %3707 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3710 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3711 = tosa.transpose %arg311, %3710 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3712 = tosa.reshape %3709 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_649 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3713 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3712, %3711 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_649 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3714 = tosa.reshape %3713 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3715 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3716 = tosa.transpose %arg312, %3715 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3717 = tosa.reshape %3709 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_650 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3718 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3717, %3716 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_650 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3719 = tosa.reshape %3718 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3720 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3721 = tosa.transpose %arg313, %3720 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3722 = tosa.reshape %3709 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_651 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3723 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3722, %3721 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_651 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3724 = tosa.reshape %3723 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3725 = tosa.reshape %3714 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3726 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3727 = tosa.transpose %3725, %3726 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3728 = tosa.reshape %3719 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3729 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3730 = tosa.transpose %3728, %3729 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3731 = tosa.reshape %3724 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3732 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3733 = tosa.transpose %3731, %3732 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_652 = tensor.extract_slice %arg314[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_653 = tensor.extract_slice %extracted_slice_652[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_654 = tensor.extract_slice %extracted_slice_653[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_655 = tensor.extract_slice %arg315[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_656 = tensor.extract_slice %extracted_slice_655[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_657 = tensor.extract_slice %extracted_slice_656[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3734 = tensor.empty() : tensor<1x40x128xf32>
    %3735 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_654 : tensor<1x1x40x128xf32>) outs(%3734 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3736 = tensor.empty() : tensor<40x128xf32>
    %3737 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3735 : tensor<1x40x128xf32>) outs(%3736 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3738 = tensor.empty() : tensor<1x40x128xf32>
    %3739 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_657 : tensor<1x1x40x128xf32>) outs(%3738 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3740 = tensor.empty() : tensor<40x128xf32>
    %3741 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3739 : tensor<1x40x128xf32>) outs(%3740 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3742 = tensor.empty() : tensor<1x40x128xf32>
    %3743 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3742 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3737[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3744 = tosa.reshape %3743 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %3745 = tensor.empty() : tensor<1x40x128xf32>
    %3746 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3745 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3741[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3747 = tosa.reshape %3746 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3748 = tosa.mul %3727, %3744 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_658 = tensor.extract_slice %3727[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_659 = tensor.extract_slice %3727[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3749 = tensor.empty() : tensor<1x32x40x64xf32>
    %3750 = linalg.negf ins(%extracted_slice_659 : tensor<1x32x40x64xf32>) outs(%3749 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3751 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_660 = tensor.insert_slice %3750 into %3751[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_661 = tensor.insert_slice %extracted_slice_658 into %inserted_slice_660[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3752 = tosa.mul %inserted_slice_661, %3747 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3753 = tosa.add %3748, %3752 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3754 = tosa.mul %3730, %3744 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_662 = tensor.extract_slice %3730[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_663 = tensor.extract_slice %3730[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3755 = tensor.empty() : tensor<1x32x40x64xf32>
    %3756 = linalg.negf ins(%extracted_slice_663 : tensor<1x32x40x64xf32>) outs(%3755 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3757 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_664 = tensor.insert_slice %3756 into %3757[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_665 = tensor.insert_slice %extracted_slice_662 into %inserted_slice_664[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3758 = tosa.mul %inserted_slice_665, %3747 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3759 = tosa.add %3754, %3758 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3760 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3761 = tosa.transpose %3759, %3760 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3762 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3763 = tosa.add %3753, %3762 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3764 = tosa.reshape %3763 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3765 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3766 = tosa.add %3761, %3765 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3767 = tosa.reshape %3766 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3768 = tosa.matmul %3764, %3767 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3769 = tosa.reshape %3768 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3770 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3771 = tosa.reciprocal %3770 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3772 = tosa.mul %3769, %3771 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3773 = tosa.add %3772, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3774 = tosa.reduce_max %3773 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3775 = tosa.sub %3773, %3774 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3776 = tosa.exp %3775 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3777 = tosa.reduce_sum %3776 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3778 = tosa.reciprocal %3777 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3779 = tosa.mul %3776, %3778 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3780 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3781 = tosa.add %3779, %3780 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3782 = tosa.reshape %3781 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3783 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3784 = tosa.add %3733, %3783 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3785 = tosa.reshape %3784 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3786 = tosa.matmul %3782, %3785 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3787 = tosa.reshape %3786 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3788 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3789 = tosa.transpose %3787, %3788 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3790 = tosa.identity %3789 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3791 = tosa.reshape %3790 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3792 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3793 = tosa.transpose %arg316, %3792 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3794 = tosa.reshape %3791 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_666 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3795 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3794, %3793 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_666 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3796 = tosa.reshape %3795 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3797 = tosa.add %3697, %3796 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3798 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_667 = arith.constant 2 : i32
    %3799 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3797 : tensor<1x40x4096xf32>) outs(%3798 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_667 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3800 = tosa.reduce_sum %3799 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3801 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3802 = tosa.reciprocal %3801 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3803 = tosa.mul %3802, %3800 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3804 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3805 = tosa.add %3803, %3804 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3806 = tosa.rsqrt %3805 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3807 = tosa.mul %3797, %3806 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3808 = tosa.reshape %arg317 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3809 = tosa.mul %3808, %3807 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3810 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3811 = tosa.transpose %arg318, %3810 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3812 = tosa.reshape %3809 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_668 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3813 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3812, %3811 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_668 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3814 = tosa.reshape %3813 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3815 = tosa.sigmoid %3814 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3816 = tosa.mul %3814, %3815 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3817 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3818 = tosa.transpose %arg319, %3817 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3819 = tosa.reshape %3809 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_669 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3820 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3819, %3818 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_669 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3821 = tosa.reshape %3820 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3822 = tosa.mul %3816, %3821 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3823 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3824 = tosa.transpose %arg320, %3823 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3825 = tosa.reshape %3822 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_670 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3826 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3825, %3824 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_670 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3827 = tosa.reshape %3826 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3828 = tosa.add %3797, %3827 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3829 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_671 = arith.constant 2 : i32
    %3830 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3828 : tensor<1x40x4096xf32>) outs(%3829 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_671 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3831 = tosa.reduce_sum %3830 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3832 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3833 = tosa.reciprocal %3832 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3834 = tosa.mul %3833, %3831 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3835 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3836 = tosa.add %3834, %3835 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3837 = tosa.rsqrt %3836 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3838 = tosa.mul %3828, %3837 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3839 = tosa.reshape %arg321 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3840 = tosa.mul %3839, %3838 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3841 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3842 = tosa.transpose %arg322, %3841 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3843 = tosa.reshape %3840 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_672 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3844 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3843, %3842 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_672 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3845 = tosa.reshape %3844 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3846 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3847 = tosa.transpose %arg323, %3846 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3848 = tosa.reshape %3840 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_673 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3849 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3848, %3847 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_673 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3850 = tosa.reshape %3849 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3851 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3852 = tosa.transpose %arg324, %3851 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3853 = tosa.reshape %3840 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_674 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3854 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3853, %3852 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_674 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3855 = tosa.reshape %3854 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3856 = tosa.reshape %3845 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3857 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3858 = tosa.transpose %3856, %3857 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3859 = tosa.reshape %3850 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3860 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3861 = tosa.transpose %3859, %3860 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3862 = tosa.reshape %3855 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3863 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3864 = tosa.transpose %3862, %3863 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_675 = tensor.extract_slice %arg325[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_676 = tensor.extract_slice %extracted_slice_675[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_677 = tensor.extract_slice %extracted_slice_676[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_678 = tensor.extract_slice %arg326[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_679 = tensor.extract_slice %extracted_slice_678[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_680 = tensor.extract_slice %extracted_slice_679[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3865 = tensor.empty() : tensor<1x40x128xf32>
    %3866 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_677 : tensor<1x1x40x128xf32>) outs(%3865 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3867 = tensor.empty() : tensor<40x128xf32>
    %3868 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3866 : tensor<1x40x128xf32>) outs(%3867 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3869 = tensor.empty() : tensor<1x40x128xf32>
    %3870 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_680 : tensor<1x1x40x128xf32>) outs(%3869 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3871 = tensor.empty() : tensor<40x128xf32>
    %3872 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3870 : tensor<1x40x128xf32>) outs(%3871 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %3873 = tensor.empty() : tensor<1x40x128xf32>
    %3874 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3873 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3868[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3875 = tosa.reshape %3874 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %3876 = tensor.empty() : tensor<1x40x128xf32>
    %3877 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%3876 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3872[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %3878 = tosa.reshape %3877 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3879 = tosa.mul %3858, %3875 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_681 = tensor.extract_slice %3858[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_682 = tensor.extract_slice %3858[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3880 = tensor.empty() : tensor<1x32x40x64xf32>
    %3881 = linalg.negf ins(%extracted_slice_682 : tensor<1x32x40x64xf32>) outs(%3880 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3882 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_683 = tensor.insert_slice %3881 into %3882[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_684 = tensor.insert_slice %extracted_slice_681 into %inserted_slice_683[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3883 = tosa.mul %inserted_slice_684, %3878 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3884 = tosa.add %3879, %3883 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3885 = tosa.mul %3861, %3875 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_685 = tensor.extract_slice %3861[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_686 = tensor.extract_slice %3861[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %3886 = tensor.empty() : tensor<1x32x40x64xf32>
    %3887 = linalg.negf ins(%extracted_slice_686 : tensor<1x32x40x64xf32>) outs(%3886 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3888 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_687 = tensor.insert_slice %3887 into %3888[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_688 = tensor.insert_slice %extracted_slice_685 into %inserted_slice_687[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %3889 = tosa.mul %inserted_slice_688, %3878 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3890 = tosa.add %3885, %3889 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3891 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3892 = tosa.transpose %3890, %3891 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3893 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3894 = tosa.add %3884, %3893 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3895 = tosa.reshape %3894 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3896 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3897 = tosa.add %3892, %3896 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3898 = tosa.reshape %3897 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3899 = tosa.matmul %3895, %3898 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3900 = tosa.reshape %3899 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3901 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3902 = tosa.reciprocal %3901 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3903 = tosa.mul %3900, %3902 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3904 = tosa.add %3903, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3905 = tosa.reduce_max %3904 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3906 = tosa.sub %3904, %3905 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3907 = tosa.exp %3906 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3908 = tosa.reduce_sum %3907 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3909 = tosa.reciprocal %3908 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3910 = tosa.mul %3907, %3909 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3911 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3912 = tosa.add %3910, %3911 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3913 = tosa.reshape %3912 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3914 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3915 = tosa.add %3864, %3914 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3916 = tosa.reshape %3915 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3917 = tosa.matmul %3913, %3916 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3918 = tosa.reshape %3917 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3919 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3920 = tosa.transpose %3918, %3919 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3921 = tosa.identity %3920 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3922 = tosa.reshape %3921 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3923 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3924 = tosa.transpose %arg327, %3923 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3925 = tosa.reshape %3922 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_689 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3926 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3925, %3924 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_689 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3927 = tosa.reshape %3926 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3928 = tosa.add %3828, %3927 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3929 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_690 = arith.constant 2 : i32
    %3930 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3928 : tensor<1x40x4096xf32>) outs(%3929 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_690 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3931 = tosa.reduce_sum %3930 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3932 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3933 = tosa.reciprocal %3932 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3934 = tosa.mul %3933, %3931 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3935 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3936 = tosa.add %3934, %3935 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3937 = tosa.rsqrt %3936 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3938 = tosa.mul %3928, %3937 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3939 = tosa.reshape %arg328 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3940 = tosa.mul %3939, %3938 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3941 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3942 = tosa.transpose %arg329, %3941 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3943 = tosa.reshape %3940 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_691 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3944 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3943, %3942 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_691 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3945 = tosa.reshape %3944 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3946 = tosa.sigmoid %3945 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3947 = tosa.mul %3945, %3946 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3948 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3949 = tosa.transpose %arg330, %3948 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3950 = tosa.reshape %3940 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_692 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %3951 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3950, %3949 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_692 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %3952 = tosa.reshape %3951 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3953 = tosa.mul %3947, %3952 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3954 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3955 = tosa.transpose %arg331, %3954 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3956 = tosa.reshape %3953 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_693 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3957 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3956, %3955 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_693 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3958 = tosa.reshape %3957 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3959 = tosa.add %3928, %3958 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %3960 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_694 = arith.constant 2 : i32
    %3961 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3959 : tensor<1x40x4096xf32>) outs(%3960 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_694 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %3962 = tosa.reduce_sum %3961 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3963 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3964 = tosa.reciprocal %3963 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3965 = tosa.mul %3964, %3962 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3966 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3967 = tosa.add %3965, %3966 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3968 = tosa.rsqrt %3967 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3969 = tosa.mul %3959, %3968 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3970 = tosa.reshape %arg332 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3971 = tosa.mul %3970, %3969 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3972 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3973 = tosa.transpose %arg333, %3972 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3974 = tosa.reshape %3971 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_695 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3975 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3974, %3973 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_695 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3976 = tosa.reshape %3975 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3977 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3978 = tosa.transpose %arg334, %3977 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3979 = tosa.reshape %3971 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_696 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3980 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3979, %3978 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_696 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3981 = tosa.reshape %3980 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3982 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3983 = tosa.transpose %arg335, %3982 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3984 = tosa.reshape %3971 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_697 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %3985 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3984, %3983 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_697 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %3986 = tosa.reshape %3985 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3987 = tosa.reshape %3976 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3988 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3989 = tosa.transpose %3987, %3988 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3990 = tosa.reshape %3981 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3991 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3992 = tosa.transpose %3990, %3991 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3993 = tosa.reshape %3986 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3994 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %3995 = tosa.transpose %3993, %3994 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_698 = tensor.extract_slice %arg336[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_699 = tensor.extract_slice %extracted_slice_698[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_700 = tensor.extract_slice %extracted_slice_699[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_701 = tensor.extract_slice %arg337[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_702 = tensor.extract_slice %extracted_slice_701[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_703 = tensor.extract_slice %extracted_slice_702[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %3996 = tensor.empty() : tensor<1x40x128xf32>
    %3997 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_700 : tensor<1x1x40x128xf32>) outs(%3996 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %3998 = tensor.empty() : tensor<40x128xf32>
    %3999 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%3997 : tensor<1x40x128xf32>) outs(%3998 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %4000 = tensor.empty() : tensor<1x40x128xf32>
    %4001 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_703 : tensor<1x1x40x128xf32>) outs(%4000 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %4002 = tensor.empty() : tensor<40x128xf32>
    %4003 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%4001 : tensor<1x40x128xf32>) outs(%4002 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %4004 = tensor.empty() : tensor<1x40x128xf32>
    %4005 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%4004 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %3999[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %4006 = tosa.reshape %4005 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %4007 = tensor.empty() : tensor<1x40x128xf32>
    %4008 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%4007 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %4003[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %4009 = tosa.reshape %4008 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4010 = tosa.mul %3989, %4006 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_704 = tensor.extract_slice %3989[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_705 = tensor.extract_slice %3989[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %4011 = tensor.empty() : tensor<1x32x40x64xf32>
    %4012 = linalg.negf ins(%extracted_slice_705 : tensor<1x32x40x64xf32>) outs(%4011 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %4013 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_706 = tensor.insert_slice %4012 into %4013[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_707 = tensor.insert_slice %extracted_slice_704 into %inserted_slice_706[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %4014 = tosa.mul %inserted_slice_707, %4009 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4015 = tosa.add %4010, %4014 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4016 = tosa.mul %3992, %4006 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_708 = tensor.extract_slice %3992[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_709 = tensor.extract_slice %3992[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %4017 = tensor.empty() : tensor<1x32x40x64xf32>
    %4018 = linalg.negf ins(%extracted_slice_709 : tensor<1x32x40x64xf32>) outs(%4017 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %4019 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_710 = tensor.insert_slice %4018 into %4019[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_711 = tensor.insert_slice %extracted_slice_708 into %inserted_slice_710[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %4020 = tosa.mul %inserted_slice_711, %4009 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4021 = tosa.add %4016, %4020 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4022 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4023 = tosa.transpose %4021, %4022 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4024 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4025 = tosa.add %4015, %4024 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4026 = tosa.reshape %4025 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4027 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4028 = tosa.add %4023, %4027 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4029 = tosa.reshape %4028 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4030 = tosa.matmul %4026, %4029 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4031 = tosa.reshape %4030 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4032 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4033 = tosa.reciprocal %4032 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4034 = tosa.mul %4031, %4033 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4035 = tosa.add %4034, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4036 = tosa.reduce_max %4035 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4037 = tosa.sub %4035, %4036 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4038 = tosa.exp %4037 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4039 = tosa.reduce_sum %4038 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4040 = tosa.reciprocal %4039 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4041 = tosa.mul %4038, %4040 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4042 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4043 = tosa.add %4041, %4042 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4044 = tosa.reshape %4043 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4045 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4046 = tosa.add %3995, %4045 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4047 = tosa.reshape %4046 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4048 = tosa.matmul %4044, %4047 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4049 = tosa.reshape %4048 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4050 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4051 = tosa.transpose %4049, %4050 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4052 = tosa.identity %4051 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4053 = tosa.reshape %4052 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4054 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4055 = tosa.transpose %arg338, %4054 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4056 = tosa.reshape %4053 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_712 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4057 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4056, %4055 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_712 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %4058 = tosa.reshape %4057 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4059 = tosa.add %3959, %4058 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %4060 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_713 = arith.constant 2 : i32
    %4061 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4059 : tensor<1x40x4096xf32>) outs(%4060 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_713 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %4062 = tosa.reduce_sum %4061 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4063 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4064 = tosa.reciprocal %4063 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4065 = tosa.mul %4064, %4062 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4066 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4067 = tosa.add %4065, %4066 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4068 = tosa.rsqrt %4067 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4069 = tosa.mul %4059, %4068 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4070 = tosa.reshape %arg339 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4071 = tosa.mul %4070, %4069 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4072 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4073 = tosa.transpose %arg340, %4072 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4074 = tosa.reshape %4071 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_714 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %4075 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4074, %4073 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_714 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %4076 = tosa.reshape %4075 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4077 = tosa.sigmoid %4076 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4078 = tosa.mul %4076, %4077 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4079 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4080 = tosa.transpose %arg341, %4079 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4081 = tosa.reshape %4071 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_715 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %4082 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4081, %4080 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_715 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %4083 = tosa.reshape %4082 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4084 = tosa.mul %4078, %4083 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4085 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4086 = tosa.transpose %arg342, %4085 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4087 = tosa.reshape %4084 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_716 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4088 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4087, %4086 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_716 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %4089 = tosa.reshape %4088 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4090 = tosa.add %4059, %4089 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %4091 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_717 = arith.constant 2 : i32
    %4092 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4090 : tensor<1x40x4096xf32>) outs(%4091 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_717 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %4093 = tosa.reduce_sum %4092 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4094 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4095 = tosa.reciprocal %4094 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4096 = tosa.mul %4095, %4093 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4097 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4098 = tosa.add %4096, %4097 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4099 = tosa.rsqrt %4098 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4100 = tosa.mul %4090, %4099 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4101 = tosa.reshape %arg343 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4102 = tosa.mul %4101, %4100 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4103 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4104 = tosa.transpose %arg344, %4103 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4105 = tosa.reshape %4102 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_718 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4106 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4105, %4104 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_718 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %4107 = tosa.reshape %4106 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4108 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4109 = tosa.transpose %arg345, %4108 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4110 = tosa.reshape %4102 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_719 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4111 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4110, %4109 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_719 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %4112 = tosa.reshape %4111 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4113 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4114 = tosa.transpose %arg346, %4113 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4115 = tosa.reshape %4102 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_720 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4116 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4115, %4114 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_720 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %4117 = tosa.reshape %4116 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4118 = tosa.reshape %4107 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4119 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4120 = tosa.transpose %4118, %4119 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4121 = tosa.reshape %4112 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4122 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4123 = tosa.transpose %4121, %4122 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4124 = tosa.reshape %4117 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4125 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4126 = tosa.transpose %4124, %4125 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_721 = tensor.extract_slice %arg347[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_722 = tensor.extract_slice %extracted_slice_721[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_723 = tensor.extract_slice %extracted_slice_722[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_724 = tensor.extract_slice %arg348[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_725 = tensor.extract_slice %extracted_slice_724[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_726 = tensor.extract_slice %extracted_slice_725[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %4127 = tensor.empty() : tensor<1x40x128xf32>
    %4128 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_723 : tensor<1x1x40x128xf32>) outs(%4127 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %4129 = tensor.empty() : tensor<40x128xf32>
    %4130 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%4128 : tensor<1x40x128xf32>) outs(%4129 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %4131 = tensor.empty() : tensor<1x40x128xf32>
    %4132 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_726 : tensor<1x1x40x128xf32>) outs(%4131 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %4133 = tensor.empty() : tensor<40x128xf32>
    %4134 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%4132 : tensor<1x40x128xf32>) outs(%4133 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %4135 = tensor.empty() : tensor<1x40x128xf32>
    %4136 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%4135 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %4130[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %4137 = tosa.reshape %4136 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    %4138 = tensor.empty() : tensor<1x40x128xf32>
    %4139 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x40xi64>) outs(%4138 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4239 = arith.index_cast %in : i64 to index
      %4240 = linalg.index 2 : index
      %extracted = tensor.extract %4134[%4239, %4240] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    call @timingStart() : () -> ()
    %4140 = tosa.reshape %4139 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4141 = tosa.mul %4120, %4137 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_727 = tensor.extract_slice %4120[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_728 = tensor.extract_slice %4120[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %4142 = tensor.empty() : tensor<1x32x40x64xf32>
    %4143 = linalg.negf ins(%extracted_slice_728 : tensor<1x32x40x64xf32>) outs(%4142 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %4144 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_729 = tensor.insert_slice %4143 into %4144[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_730 = tensor.insert_slice %extracted_slice_727 into %inserted_slice_729[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %4145 = tosa.mul %inserted_slice_730, %4140 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4146 = tosa.add %4141, %4145 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4147 = tosa.mul %4123, %4137 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    %extracted_slice_731 = tensor.extract_slice %4123[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_732 = tensor.extract_slice %4123[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %4148 = tensor.empty() : tensor<1x32x40x64xf32>
    %4149 = linalg.negf ins(%extracted_slice_732 : tensor<1x32x40x64xf32>) outs(%4148 : tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %4150 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_733 = tensor.insert_slice %4149 into %4150[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_734 = tensor.insert_slice %extracted_slice_731 into %inserted_slice_733[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    call @timingStart() : () -> ()
    %4151 = tosa.mul %inserted_slice_734, %4140 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4152 = tosa.add %4147, %4151 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4153 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4154 = tosa.transpose %4152, %4153 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4155 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4156 = tosa.add %4146, %4155 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4157 = tosa.reshape %4156 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4158 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xf32>}> : () -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4159 = tosa.add %4154, %4158 : (tensor<1x32x128x40xf32>, tensor<1x32x128x40xf32>) -> tensor<1x32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4160 = tosa.reshape %4159 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xf32>) -> tensor<32x128x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4161 = tosa.matmul %4157, %4160 : (tensor<32x40x128xf32>, tensor<32x128x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4162 = tosa.reshape %4161 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4163 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4164 = tosa.reciprocal %4163 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4165 = tosa.mul %4162, %4164 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4166 = tosa.add %4165, %29 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4167 = tosa.reduce_max %4166 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4168 = tosa.sub %4166, %4167 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4169 = tosa.exp %4168 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4170 = tosa.reduce_sum %4169 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4171 = tosa.reciprocal %4170 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4172 = tosa.mul %4169, %4171 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4173 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4174 = tosa.add %4172, %4173 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4175 = tosa.reshape %4174 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4176 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4177 = tosa.add %4126, %4176 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4178 = tosa.reshape %4177 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4179 = tosa.matmul %4175, %4178 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4180 = tosa.reshape %4179 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4181 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4182 = tosa.transpose %4180, %4181 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4183 = tosa.identity %4182 : (tensor<1x40x32x128xf32>) -> tensor<1x40x32x128xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4184 = tosa.reshape %4183 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4185 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4186 = tosa.transpose %arg349, %4185 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4187 = tosa.reshape %4184 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_735 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4188 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4187, %4186 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_735 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %4189 = tosa.reshape %4188 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4190 = tosa.add %4090, %4189 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %4191 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_736 = arith.constant 2 : i32
    %4192 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4190 : tensor<1x40x4096xf32>) outs(%4191 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_736 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %4193 = tosa.reduce_sum %4192 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4194 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4195 = tosa.reciprocal %4194 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4196 = tosa.mul %4195, %4193 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4197 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4198 = tosa.add %4196, %4197 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4199 = tosa.rsqrt %4198 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4200 = tosa.mul %4190, %4199 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4201 = tosa.reshape %arg350 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4202 = tosa.mul %4201, %4200 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4203 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4204 = tosa.transpose %arg351, %4203 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4205 = tosa.reshape %4202 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_737 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %4206 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4205, %4204 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_737 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %4207 = tosa.reshape %4206 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4208 = tosa.sigmoid %4207 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4209 = tosa.mul %4207, %4208 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4210 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4211 = tosa.transpose %arg352, %4210 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4212 = tosa.reshape %4202 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_738 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
    %4213 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4212, %4211 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_738 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
    call @timingStart() : () -> ()
    %4214 = tosa.reshape %4213 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4215 = tosa.mul %4209, %4214 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4216 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4217 = tosa.transpose %arg353, %4216 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4218 = tosa.reshape %4215 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
    call @timingEnd() : () -> ()
    %cst_739 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
    %4219 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4218, %4217 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_739 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
    call @timingStart() : () -> ()
    %4220 = tosa.reshape %4219 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4221 = tosa.add %4190, %4220 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    %4222 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_740 = arith.constant 2 : i32
    %4223 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4221 : tensor<1x40x4096xf32>) outs(%4222 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4239 = math.fpowi %in, %c2_i32_740 : f32, i32
      linalg.yield %4239 : f32
    } -> tensor<1x40x4096xf32>
    call @timingStart() : () -> ()
    %4224 = tosa.reduce_sum %4223 {axis = 2 : i32} : (tensor<1x40x4096xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4225 = "tosa.const"() <{value = dense<4.096000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4226 = tosa.reciprocal %4225 : (tensor<1xf32>) -> tensor<1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4227 = tosa.mul %4226, %4224 {shift = 0 : i8} : (tensor<1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4228 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4229 = tosa.add %4227, %4228 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4230 = tosa.rsqrt %4229 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4231 = tosa.mul %4221, %4230 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4232 = tosa.reshape %arg354 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4233 = tosa.mul %4232, %4231 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4234 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4235 = tosa.transpose %arg355, %4234 : (tensor<32000x4096xf32>, tensor<2xi32>) -> tensor<4096x32000xf32>
    call @timingEnd() : () -> ()
    call @timingStart() : () -> ()
    %4236 = tosa.reshape %4233 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
    call @timingEnd() : () -> ()
    %cst_741 = arith.constant dense<0.000000e+00> : tensor<40x32000xf32>
    %4237 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4236, %4235 : tensor<40x4096xf32>, tensor<4096x32000xf32>) outs(%cst_741 : tensor<40x32000xf32>) -> tensor<40x32000xf32>
    call @timingStart() : () -> ()
    %4238 = tosa.reshape %4237 {new_shape = array<i64: 1, 40, 32000>} : (tensor<40x32000xf32>) -> tensor<1x40x32000xf32>
    call @timingEnd() : () -> ()
    return %4233, %4238 : tensor<1x40x4096xf32>, tensor<1x40x32000xf32>
  }
}

