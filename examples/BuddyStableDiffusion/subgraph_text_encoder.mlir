#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1) -> (d0)>
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x12xi64>, tensor<1x77xi64>, tensor<49408x1024xf32>, tensor<77x1024xf32>, tensor<1x12xi64>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<4096x1024xf32>, tensor<4096xf32>, tensor<1024x4096xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1x12x1024xf32>, tensor<1x1024xf32>), sym_name = "subgraph0"}> ({
  ^bb0(%arg0: tensor<1x12xi64>, %arg1: tensor<1x77xi64>, %arg2: tensor<49408x1024xf32>, %arg3: tensor<77x1024xf32>, %arg4: tensor<1x12xi64>, %arg5: tensor<1024xf32>, %arg6: tensor<1024xf32>, %arg7: tensor<1024x1024xf32>, %arg8: tensor<1024xf32>, %arg9: tensor<1024x1024xf32>, %arg10: tensor<1024xf32>, %arg11: tensor<1024x1024xf32>, %arg12: tensor<1024xf32>, %arg13: tensor<1024x1024xf32>, %arg14: tensor<1024xf32>, %arg15: tensor<1024xf32>, %arg16: tensor<1024xf32>, %arg17: tensor<4096x1024xf32>, %arg18: tensor<4096xf32>, %arg19: tensor<1024x4096xf32>, %arg20: tensor<1024xf32>, %arg21: tensor<1024xf32>, %arg22: tensor<1024xf32>, %arg23: tensor<1024x1024xf32>, %arg24: tensor<1024xf32>, %arg25: tensor<1024x1024xf32>, %arg26: tensor<1024xf32>, %arg27: tensor<1024x1024xf32>, %arg28: tensor<1024xf32>, %arg29: tensor<1024x1024xf32>, %arg30: tensor<1024xf32>, %arg31: tensor<1024xf32>, %arg32: tensor<1024xf32>, %arg33: tensor<4096x1024xf32>, %arg34: tensor<4096xf32>, %arg35: tensor<1024x4096xf32>, %arg36: tensor<1024xf32>, %arg37: tensor<1024xf32>, %arg38: tensor<1024xf32>, %arg39: tensor<1024x1024xf32>, %arg40: tensor<1024xf32>, %arg41: tensor<1024x1024xf32>, %arg42: tensor<1024xf32>, %arg43: tensor<1024x1024xf32>, %arg44: tensor<1024xf32>, %arg45: tensor<1024x1024xf32>, %arg46: tensor<1024xf32>, %arg47: tensor<1024xf32>, %arg48: tensor<1024xf32>, %arg49: tensor<4096x1024xf32>, %arg50: tensor<4096xf32>, %arg51: tensor<1024x4096xf32>, %arg52: tensor<1024xf32>, %arg53: tensor<1024xf32>, %arg54: tensor<1024xf32>, %arg55: tensor<1024x1024xf32>, %arg56: tensor<1024xf32>, %arg57: tensor<1024x1024xf32>, %arg58: tensor<1024xf32>, %arg59: tensor<1024x1024xf32>, %arg60: tensor<1024xf32>, %arg61: tensor<1024x1024xf32>, %arg62: tensor<1024xf32>, %arg63: tensor<1024xf32>, %arg64: tensor<1024xf32>, %arg65: tensor<4096x1024xf32>, %arg66: tensor<4096xf32>, %arg67: tensor<1024x4096xf32>, %arg68: tensor<1024xf32>, %arg69: tensor<1024xf32>, %arg70: tensor<1024xf32>, %arg71: tensor<1024x1024xf32>, %arg72: tensor<1024xf32>, %arg73: tensor<1024x1024xf32>, %arg74: tensor<1024xf32>, %arg75: tensor<1024x1024xf32>, %arg76: tensor<1024xf32>, %arg77: tensor<1024x1024xf32>, %arg78: tensor<1024xf32>, %arg79: tensor<1024xf32>, %arg80: tensor<1024xf32>, %arg81: tensor<4096x1024xf32>, %arg82: tensor<4096xf32>, %arg83: tensor<1024x4096xf32>, %arg84: tensor<1024xf32>, %arg85: tensor<1024xf32>, %arg86: tensor<1024xf32>, %arg87: tensor<1024x1024xf32>, %arg88: tensor<1024xf32>, %arg89: tensor<1024x1024xf32>, %arg90: tensor<1024xf32>, %arg91: tensor<1024x1024xf32>, %arg92: tensor<1024xf32>, %arg93: tensor<1024x1024xf32>, %arg94: tensor<1024xf32>, %arg95: tensor<1024xf32>, %arg96: tensor<1024xf32>, %arg97: tensor<4096x1024xf32>, %arg98: tensor<4096xf32>, %arg99: tensor<1024x4096xf32>, %arg100: tensor<1024xf32>, %arg101: tensor<1024xf32>, %arg102: tensor<1024xf32>, %arg103: tensor<1024x1024xf32>, %arg104: tensor<1024xf32>, %arg105: tensor<1024x1024xf32>, %arg106: tensor<1024xf32>, %arg107: tensor<1024x1024xf32>, %arg108: tensor<1024xf32>, %arg109: tensor<1024x1024xf32>, %arg110: tensor<1024xf32>, %arg111: tensor<1024xf32>, %arg112: tensor<1024xf32>, %arg113: tensor<4096x1024xf32>, %arg114: tensor<4096xf32>, %arg115: tensor<1024x4096xf32>, %arg116: tensor<1024xf32>, %arg117: tensor<1024xf32>, %arg118: tensor<1024xf32>, %arg119: tensor<1024x1024xf32>, %arg120: tensor<1024xf32>, %arg121: tensor<1024x1024xf32>, %arg122: tensor<1024xf32>, %arg123: tensor<1024x1024xf32>, %arg124: tensor<1024xf32>, %arg125: tensor<1024x1024xf32>, %arg126: tensor<1024xf32>, %arg127: tensor<1024xf32>, %arg128: tensor<1024xf32>, %arg129: tensor<4096x1024xf32>, %arg130: tensor<4096xf32>, %arg131: tensor<1024x4096xf32>, %arg132: tensor<1024xf32>, %arg133: tensor<1024xf32>, %arg134: tensor<1024xf32>, %arg135: tensor<1024x1024xf32>, %arg136: tensor<1024xf32>, %arg137: tensor<1024x1024xf32>, %arg138: tensor<1024xf32>, %arg139: tensor<1024x1024xf32>, %arg140: tensor<1024xf32>, %arg141: tensor<1024x1024xf32>, %arg142: tensor<1024xf32>, %arg143: tensor<1024xf32>, %arg144: tensor<1024xf32>, %arg145: tensor<4096x1024xf32>, %arg146: tensor<4096xf32>, %arg147: tensor<1024x4096xf32>, %arg148: tensor<1024xf32>, %arg149: tensor<1024xf32>, %arg150: tensor<1024xf32>, %arg151: tensor<1024x1024xf32>, %arg152: tensor<1024xf32>, %arg153: tensor<1024x1024xf32>, %arg154: tensor<1024xf32>, %arg155: tensor<1024x1024xf32>, %arg156: tensor<1024xf32>, %arg157: tensor<1024x1024xf32>, %arg158: tensor<1024xf32>, %arg159: tensor<1024xf32>, %arg160: tensor<1024xf32>, %arg161: tensor<4096x1024xf32>, %arg162: tensor<4096xf32>, %arg163: tensor<1024x4096xf32>, %arg164: tensor<1024xf32>, %arg165: tensor<1024xf32>, %arg166: tensor<1024xf32>, %arg167: tensor<1024x1024xf32>, %arg168: tensor<1024xf32>, %arg169: tensor<1024x1024xf32>, %arg170: tensor<1024xf32>, %arg171: tensor<1024x1024xf32>, %arg172: tensor<1024xf32>, %arg173: tensor<1024x1024xf32>, %arg174: tensor<1024xf32>, %arg175: tensor<1024xf32>, %arg176: tensor<1024xf32>, %arg177: tensor<4096x1024xf32>, %arg178: tensor<4096xf32>, %arg179: tensor<1024x4096xf32>, %arg180: tensor<1024xf32>, %arg181: tensor<1024xf32>, %arg182: tensor<1024xf32>, %arg183: tensor<1024x1024xf32>, %arg184: tensor<1024xf32>, %arg185: tensor<1024x1024xf32>, %arg186: tensor<1024xf32>, %arg187: tensor<1024x1024xf32>, %arg188: tensor<1024xf32>, %arg189: tensor<1024x1024xf32>, %arg190: tensor<1024xf32>, %arg191: tensor<1024xf32>, %arg192: tensor<1024xf32>, %arg193: tensor<4096x1024xf32>, %arg194: tensor<4096xf32>, %arg195: tensor<1024x4096xf32>, %arg196: tensor<1024xf32>, %arg197: tensor<1024xf32>, %arg198: tensor<1024xf32>, %arg199: tensor<1024x1024xf32>, %arg200: tensor<1024xf32>, %arg201: tensor<1024x1024xf32>, %arg202: tensor<1024xf32>, %arg203: tensor<1024x1024xf32>, %arg204: tensor<1024xf32>, %arg205: tensor<1024x1024xf32>, %arg206: tensor<1024xf32>, %arg207: tensor<1024xf32>, %arg208: tensor<1024xf32>, %arg209: tensor<4096x1024xf32>, %arg210: tensor<4096xf32>, %arg211: tensor<1024x4096xf32>, %arg212: tensor<1024xf32>, %arg213: tensor<1024xf32>, %arg214: tensor<1024xf32>, %arg215: tensor<1024x1024xf32>, %arg216: tensor<1024xf32>, %arg217: tensor<1024x1024xf32>, %arg218: tensor<1024xf32>, %arg219: tensor<1024x1024xf32>, %arg220: tensor<1024xf32>, %arg221: tensor<1024x1024xf32>, %arg222: tensor<1024xf32>, %arg223: tensor<1024xf32>, %arg224: tensor<1024xf32>, %arg225: tensor<4096x1024xf32>, %arg226: tensor<4096xf32>, %arg227: tensor<1024x4096xf32>, %arg228: tensor<1024xf32>, %arg229: tensor<1024xf32>, %arg230: tensor<1024xf32>, %arg231: tensor<1024x1024xf32>, %arg232: tensor<1024xf32>, %arg233: tensor<1024x1024xf32>, %arg234: tensor<1024xf32>, %arg235: tensor<1024x1024xf32>, %arg236: tensor<1024xf32>, %arg237: tensor<1024x1024xf32>, %arg238: tensor<1024xf32>, %arg239: tensor<1024xf32>, %arg240: tensor<1024xf32>, %arg241: tensor<4096x1024xf32>, %arg242: tensor<4096xf32>, %arg243: tensor<1024x4096xf32>, %arg244: tensor<1024xf32>, %arg245: tensor<1024xf32>, %arg246: tensor<1024xf32>, %arg247: tensor<1024x1024xf32>, %arg248: tensor<1024xf32>, %arg249: tensor<1024x1024xf32>, %arg250: tensor<1024xf32>, %arg251: tensor<1024x1024xf32>, %arg252: tensor<1024xf32>, %arg253: tensor<1024x1024xf32>, %arg254: tensor<1024xf32>, %arg255: tensor<1024xf32>, %arg256: tensor<1024xf32>, %arg257: tensor<4096x1024xf32>, %arg258: tensor<4096xf32>, %arg259: tensor<1024x4096xf32>, %arg260: tensor<1024xf32>, %arg261: tensor<1024xf32>, %arg262: tensor<1024xf32>, %arg263: tensor<1024x1024xf32>, %arg264: tensor<1024xf32>, %arg265: tensor<1024x1024xf32>, %arg266: tensor<1024xf32>, %arg267: tensor<1024x1024xf32>, %arg268: tensor<1024xf32>, %arg269: tensor<1024x1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<4096x1024xf32>, %arg274: tensor<4096xf32>, %arg275: tensor<1024x4096xf32>, %arg276: tensor<1024xf32>, %arg277: tensor<1024xf32>, %arg278: tensor<1024xf32>, %arg279: tensor<1024x1024xf32>, %arg280: tensor<1024xf32>, %arg281: tensor<1024x1024xf32>, %arg282: tensor<1024xf32>, %arg283: tensor<1024x1024xf32>, %arg284: tensor<1024xf32>, %arg285: tensor<1024x1024xf32>, %arg286: tensor<1024xf32>, %arg287: tensor<1024xf32>, %arg288: tensor<1024xf32>, %arg289: tensor<4096x1024xf32>, %arg290: tensor<4096xf32>, %arg291: tensor<1024x4096xf32>, %arg292: tensor<1024xf32>, %arg293: tensor<1024xf32>, %arg294: tensor<1024xf32>, %arg295: tensor<1024x1024xf32>, %arg296: tensor<1024xf32>, %arg297: tensor<1024x1024xf32>, %arg298: tensor<1024xf32>, %arg299: tensor<1024x1024xf32>, %arg300: tensor<1024xf32>, %arg301: tensor<1024x1024xf32>, %arg302: tensor<1024xf32>, %arg303: tensor<1024xf32>, %arg304: tensor<1024xf32>, %arg305: tensor<4096x1024xf32>, %arg306: tensor<4096xf32>, %arg307: tensor<1024x4096xf32>, %arg308: tensor<1024xf32>, %arg309: tensor<1024xf32>, %arg310: tensor<1024xf32>, %arg311: tensor<1024x1024xf32>, %arg312: tensor<1024xf32>, %arg313: tensor<1024x1024xf32>, %arg314: tensor<1024xf32>, %arg315: tensor<1024x1024xf32>, %arg316: tensor<1024xf32>, %arg317: tensor<1024x1024xf32>, %arg318: tensor<1024xf32>, %arg319: tensor<1024xf32>, %arg320: tensor<1024xf32>, %arg321: tensor<4096x1024xf32>, %arg322: tensor<4096xf32>, %arg323: tensor<1024x4096xf32>, %arg324: tensor<1024xf32>, %arg325: tensor<1024xf32>, %arg326: tensor<1024xf32>, %arg327: tensor<1024x1024xf32>, %arg328: tensor<1024xf32>, %arg329: tensor<1024x1024xf32>, %arg330: tensor<1024xf32>, %arg331: tensor<1024x1024xf32>, %arg332: tensor<1024xf32>, %arg333: tensor<1024x1024xf32>, %arg334: tensor<1024xf32>, %arg335: tensor<1024xf32>, %arg336: tensor<1024xf32>, %arg337: tensor<4096x1024xf32>, %arg338: tensor<4096xf32>, %arg339: tensor<1024x4096xf32>, %arg340: tensor<1024xf32>, %arg341: tensor<1024xf32>, %arg342: tensor<1024xf32>, %arg343: tensor<1024x1024xf32>, %arg344: tensor<1024xf32>, %arg345: tensor<1024x1024xf32>, %arg346: tensor<1024xf32>, %arg347: tensor<1024x1024xf32>, %arg348: tensor<1024xf32>, %arg349: tensor<1024x1024xf32>, %arg350: tensor<1024xf32>, %arg351: tensor<1024xf32>, %arg352: tensor<1024xf32>, %arg353: tensor<4096x1024xf32>, %arg354: tensor<4096xf32>, %arg355: tensor<1024x4096xf32>, %arg356: tensor<1024xf32>, %arg357: tensor<1024xf32>, %arg358: tensor<1024xf32>, %arg359: tensor<1024x1024xf32>, %arg360: tensor<1024xf32>, %arg361: tensor<1024x1024xf32>, %arg362: tensor<1024xf32>, %arg363: tensor<1024x1024xf32>, %arg364: tensor<1024xf32>, %arg365: tensor<1024x1024xf32>, %arg366: tensor<1024xf32>, %arg367: tensor<1024xf32>, %arg368: tensor<1024xf32>, %arg369: tensor<4096x1024xf32>, %arg370: tensor<4096xf32>, %arg371: tensor<1024x4096xf32>, %arg372: tensor<1024xf32>, %arg373: tensor<1024xf32>, %arg374: tensor<1024xf32>):
    %0 = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 12>}> : (tensor<1x12xi64>) -> tensor<1x12xi64>
    %1 = "tensor.extract_slice"(%arg1) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 77>, static_strides = array<i64: 1, 1>}> : (tensor<1x77xi64>) -> tensor<1x77xi64>
    %2 = "tensor.extract_slice"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 12>, static_strides = array<i64: 1, 1>}> : (tensor<1x77xi64>) -> tensor<1x12xi64>
    %3 = "tosa.cast"(%0) : (tensor<1x12xi64>) -> tensor<1x12xi32>
    %4 = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 49408, 1024>}> : (tensor<49408x1024xf32>) -> tensor<1x49408x1024xf32>
    %5 = "tosa.gather"(%4, %3) : (tensor<1x49408x1024xf32>, tensor<1x12xi32>) -> tensor<1x12x1024xf32>
    %6 = "tosa.reshape"(%5) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %7 = "tosa.cast"(%2) : (tensor<1x12xi64>) -> tensor<1x12xi32>
    %8 = "tosa.reshape"(%arg3) <{new_shape = array<i64: 1, 77, 1024>}> : (tensor<77x1024xf32>) -> tensor<1x77x1024xf32>
    %9 = "tosa.gather"(%8, %7) : (tensor<1x77x1024xf32>, tensor<1x12xi32>) -> tensor<1x12x1024xf32>
    %10 = "tosa.reshape"(%9) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %11 = "tosa.add"(%6, %10) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %12 = "arith.constant"() <{value = dense<-3.40282347E+38> : tensor<12x12xf32>}> : () -> tensor<12x12xf32>
    %13 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]> : tensor<12xi64>}> : () -> tensor<12xi64>
    %14 = "tosa.const"() <{value = dense<1> : tensor<12xi64>}> : () -> tensor<12xi64>
    %15 = "tosa.add"(%13, %14) : (tensor<12xi64>, tensor<12xi64>) -> tensor<12xi64>
    %16 = "tosa.reshape"(%15) <{new_shape = array<i64: 12, 1>}> : (tensor<12xi64>) -> tensor<12x1xi64>
    %17 = "tensor.empty"() : () -> tensor<12x12xi1>
    %18 = "linalg.generic"(%13, %16, %17) <{indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg375: i64, %arg376: i64, %arg377: i1):
      %3446 = "arith.cmpi"(%arg375, %arg376) <{predicate = 2 : i64}> : (i64, i64) -> i1
      "linalg.yield"(%3446) : (i1) -> ()
    }) : (tensor<12xi64>, tensor<12x1xi64>, tensor<12x12xi1>) -> tensor<12x12xi1>
    %19 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %20 = "tensor.empty"() : () -> tensor<12x12xf32>
    %21 = "linalg.generic"(%18, %12, %20) <{indexing_maps = [#map3, #map3, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg375: i1, %arg376: f32, %arg377: f32):
      %3446 = "arith.select"(%arg375, %19, %arg376) : (i1, f32, f32) -> f32
      "linalg.yield"(%3446) : (f32) -> ()
    }) : (tensor<12x12xi1>, tensor<12x12xf32>, tensor<12x12xf32>) -> tensor<12x12xf32>
    %22 = "tensor.extract_slice"(%arg4) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 12>, static_strides = array<i64: 1, 1>}> : (tensor<1x12xi64>) -> tensor<1x12xi64>
    %23 = "tosa.reshape"(%22) <{new_shape = array<i64: 1, 1, 12>}> : (tensor<1x12xi64>) -> tensor<1x1x12xi64>
    %24 = "tosa.reshape"(%23) <{new_shape = array<i64: 1, 1, 1, 12>}> : (tensor<1x1x12xi64>) -> tensor<1x1x1x12xi64>
    %25 = "tensor.extract_slice"(%24) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 12>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x1x12xi64>) -> tensor<1x1x1x12xi64>
    %26 = "tosa.const"() <{value = dense<0> : tensor<1x1x12x12xi64>}> : () -> tensor<1x1x12x12xi64>
    %27 = "tosa.add"(%25, %26) : (tensor<1x1x1x12xi64>, tensor<1x1x12x12xi64>) -> tensor<1x1x12x12xi64>
    %28 = "tosa.cast"(%27) : (tensor<1x1x12x12xi64>) -> tensor<1x1x12x12xf32>
    %29 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x12x12xf32>}> : () -> tensor<1x1x12x12xf32>
    %30 = "tosa.sub"(%29, %28) : (tensor<1x1x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x1x12x12xf32>
    %31 = "tosa.cast"(%30) : (tensor<1x1x12x12xf32>) -> tensor<1x1x12x12xi1>
    %32 = "arith.constant"() <{value = -3.40282347E+38 : f32}> : () -> f32
    %33 = "tensor.empty"() : () -> tensor<1x1x12x12xf32>
    %34 = "linalg.generic"(%31, %30, %33) <{indexing_maps = [#map4, #map4, #map4], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg375: i1, %arg376: f32, %arg377: f32):
      %3446 = "arith.select"(%arg375, %32, %arg376) : (i1, f32, f32) -> f32
      "linalg.yield"(%3446) : (f32) -> ()
    }) : (tensor<1x1x12x12xi1>, tensor<1x1x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x1x12x12xf32>
    %35 = "tosa.reduce_sum"(%11) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %36 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %37 = "tosa.reciprocal"(%36) : (tensor<1xf32>) -> tensor<1xf32>
    %38 = "tosa.mul"(%37, %35) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %39 = "tosa.sub"(%11, %38) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %40 = "tosa.mul"(%39, %39) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %41 = "tosa.reduce_sum"(%40) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %42 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %43 = "tosa.reciprocal"(%42) : (tensor<1xf32>) -> tensor<1xf32>
    %44 = "tosa.mul"(%43, %41) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %45 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %46 = "tosa.add"(%44, %45) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %47 = "tosa.rsqrt"(%46) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %48 = "tosa.sub"(%11, %38) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %49 = "tosa.mul"(%48, %47) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %50 = "tosa.reshape"(%arg5) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %51 = "tosa.mul"(%49, %50) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %52 = "tosa.reshape"(%arg6) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %53 = "tosa.add"(%51, %52) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %54 = "tosa.reshape"(%53) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %55 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %56 = "tosa.transpose"(%arg7, %55) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %57 = "tosa.reshape"(%54) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %58 = "tosa.reshape"(%56) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %59 = "tosa.matmul"(%57, %58) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %60 = "tosa.reshape"(%59) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %61 = "tosa.reshape"(%arg8) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %62 = "tosa.add"(%61, %60) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %63 = "tosa.reshape"(%62) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %64 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %65 = "tosa.mul"(%63, %64) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %66 = "tosa.reshape"(%53) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %67 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %68 = "tosa.transpose"(%arg9, %67) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %69 = "tosa.reshape"(%66) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %70 = "tosa.reshape"(%68) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %71 = "tosa.matmul"(%69, %70) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %72 = "tosa.reshape"(%71) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %73 = "tosa.reshape"(%arg10) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %74 = "tosa.add"(%73, %72) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %75 = "tosa.reshape"(%74) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %76 = "tosa.reshape"(%75) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %77 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %78 = "tosa.transpose"(%76, %77) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %79 = "tosa.identity"(%78) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %80 = "tosa.reshape"(%53) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %81 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %82 = "tosa.transpose"(%arg11, %81) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %83 = "tosa.reshape"(%80) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %84 = "tosa.reshape"(%82) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %85 = "tosa.matmul"(%83, %84) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %86 = "tosa.reshape"(%85) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %87 = "tosa.reshape"(%arg12) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %88 = "tosa.add"(%87, %86) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %89 = "tosa.reshape"(%88) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %90 = "tosa.reshape"(%89) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %91 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %92 = "tosa.transpose"(%90, %91) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %93 = "tosa.identity"(%92) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %94 = "tosa.reshape"(%65) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %95 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %96 = "tosa.transpose"(%94, %95) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %97 = "tosa.identity"(%96) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %98 = "tosa.reshape"(%97) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %99 = "tosa.reshape"(%79) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %100 = "tosa.reshape"(%93) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %101 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %102 = "tosa.transpose"(%99, %101) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %103 = "tosa.matmul"(%98, %102) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %104 = "tosa.reshape"(%103) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %105 = "tosa.reshape"(%21) <{new_shape = array<i64: 1, 12, 12>}> : (tensor<12x12xf32>) -> tensor<1x12x12xf32>
    %106 = "tosa.reshape"(%105) <{new_shape = array<i64: 1, 1, 12, 12>}> : (tensor<1x12x12xf32>) -> tensor<1x1x12x12xf32>
    %107 = "tensor.extract_slice"(%106) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 12, 12>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x12x12xf32>) -> tensor<1x1x12x12xf32>
    %108 = "tensor.extract_slice"(%107) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 12, 12>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1x12x12xf32>) -> tensor<1x1x12x12xf32>
    %109 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x12x12xf32>}> : () -> tensor<1x1x12x12xf32>
    %110 = "tosa.add"(%108, %109) : (tensor<1x1x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x1x12x12xf32>
    %111 = "tosa.add"(%104, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %112 = "tosa.reshape"(%111) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %113 = "tosa.reshape"(%112) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %114 = "tosa.add"(%113, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %115 = "tosa.reshape"(%114) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %116 = "tosa.reduce_max"(%115) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %117 = "tosa.sub"(%115, %116) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %118 = "tosa.exp"(%117) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %119 = "tosa.reduce_sum"(%118) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %120 = "tosa.reciprocal"(%119) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %121 = "tosa.mul"(%118, %120) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %122 = "tosa.identity"(%121) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %123 = "tosa.matmul"(%122, %100) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %124 = "tosa.reshape"(%123) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %125 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %126 = "tosa.transpose"(%124, %125) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %127 = "tosa.identity"(%126) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %128 = "tosa.reshape"(%127) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %129 = "tosa.reshape"(%128) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %130 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %131 = "tosa.transpose"(%arg13, %130) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %132 = "tosa.reshape"(%129) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %133 = "tosa.reshape"(%131) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %134 = "tosa.matmul"(%132, %133) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %135 = "tosa.reshape"(%134) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %136 = "tosa.reshape"(%arg14) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %137 = "tosa.add"(%136, %135) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %138 = "tosa.reshape"(%137) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %139 = "tosa.add"(%11, %138) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %140 = "tosa.reduce_sum"(%139) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %141 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %142 = "tosa.reciprocal"(%141) : (tensor<1xf32>) -> tensor<1xf32>
    %143 = "tosa.mul"(%142, %140) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %144 = "tosa.sub"(%139, %143) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %145 = "tosa.mul"(%144, %144) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %146 = "tosa.reduce_sum"(%145) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %147 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %148 = "tosa.reciprocal"(%147) : (tensor<1xf32>) -> tensor<1xf32>
    %149 = "tosa.mul"(%148, %146) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %150 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %151 = "tosa.add"(%149, %150) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %152 = "tosa.rsqrt"(%151) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %153 = "tosa.sub"(%139, %143) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %154 = "tosa.mul"(%153, %152) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %155 = "tosa.reshape"(%arg15) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %156 = "tosa.mul"(%154, %155) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %157 = "tosa.reshape"(%arg16) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %158 = "tosa.add"(%156, %157) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %159 = "tosa.reshape"(%158) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %160 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %161 = "tosa.transpose"(%arg17, %160) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %162 = "tosa.reshape"(%159) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %163 = "tosa.reshape"(%161) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %164 = "tosa.matmul"(%162, %163) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %165 = "tosa.reshape"(%164) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %166 = "tosa.reshape"(%arg18) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %167 = "tosa.add"(%166, %165) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %168 = "tosa.reshape"(%167) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %169 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %170 = "tosa.mul"(%168, %169) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %171 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %172 = "tosa.mul"(%168, %171) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %173 = "math.erf"(%172) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %174 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %175 = "tosa.add"(%173, %174) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %176 = "tosa.mul"(%170, %175) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %177 = "tosa.reshape"(%176) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %178 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %179 = "tosa.transpose"(%arg19, %178) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %180 = "tosa.reshape"(%177) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %181 = "tosa.reshape"(%179) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %182 = "tosa.matmul"(%180, %181) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %183 = "tosa.reshape"(%182) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %184 = "tosa.reshape"(%arg20) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %185 = "tosa.add"(%184, %183) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %186 = "tosa.reshape"(%185) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %187 = "tosa.add"(%139, %186) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %188 = "tosa.reduce_sum"(%187) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %189 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %190 = "tosa.reciprocal"(%189) : (tensor<1xf32>) -> tensor<1xf32>
    %191 = "tosa.mul"(%190, %188) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %192 = "tosa.sub"(%187, %191) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %193 = "tosa.mul"(%192, %192) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %194 = "tosa.reduce_sum"(%193) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %195 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %196 = "tosa.reciprocal"(%195) : (tensor<1xf32>) -> tensor<1xf32>
    %197 = "tosa.mul"(%196, %194) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %198 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %199 = "tosa.add"(%197, %198) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %200 = "tosa.rsqrt"(%199) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %201 = "tosa.sub"(%187, %191) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %202 = "tosa.mul"(%201, %200) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %203 = "tosa.reshape"(%arg21) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %204 = "tosa.mul"(%202, %203) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %205 = "tosa.reshape"(%arg22) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %206 = "tosa.add"(%204, %205) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %207 = "tosa.reshape"(%206) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %208 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %209 = "tosa.transpose"(%arg23, %208) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %210 = "tosa.reshape"(%207) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %211 = "tosa.reshape"(%209) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %212 = "tosa.matmul"(%210, %211) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %213 = "tosa.reshape"(%212) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %214 = "tosa.reshape"(%arg24) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %215 = "tosa.add"(%214, %213) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %216 = "tosa.reshape"(%215) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %217 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %218 = "tosa.mul"(%216, %217) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %219 = "tosa.reshape"(%206) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %220 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %221 = "tosa.transpose"(%arg25, %220) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %222 = "tosa.reshape"(%219) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %223 = "tosa.reshape"(%221) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %224 = "tosa.matmul"(%222, %223) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %225 = "tosa.reshape"(%224) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %226 = "tosa.reshape"(%arg26) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %227 = "tosa.add"(%226, %225) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %228 = "tosa.reshape"(%227) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %229 = "tosa.reshape"(%228) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %230 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %231 = "tosa.transpose"(%229, %230) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %232 = "tosa.identity"(%231) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %233 = "tosa.reshape"(%206) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %234 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %235 = "tosa.transpose"(%arg27, %234) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %236 = "tosa.reshape"(%233) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %237 = "tosa.reshape"(%235) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %238 = "tosa.matmul"(%236, %237) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %239 = "tosa.reshape"(%238) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %240 = "tosa.reshape"(%arg28) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %241 = "tosa.add"(%240, %239) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %242 = "tosa.reshape"(%241) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %243 = "tosa.reshape"(%242) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %244 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %245 = "tosa.transpose"(%243, %244) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %246 = "tosa.identity"(%245) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %247 = "tosa.reshape"(%218) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %248 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %249 = "tosa.transpose"(%247, %248) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %250 = "tosa.identity"(%249) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %251 = "tosa.reshape"(%250) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %252 = "tosa.reshape"(%232) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %253 = "tosa.reshape"(%246) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %254 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %255 = "tosa.transpose"(%252, %254) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %256 = "tosa.matmul"(%251, %255) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %257 = "tosa.reshape"(%256) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %258 = "tosa.add"(%257, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %259 = "tosa.reshape"(%258) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %260 = "tosa.reshape"(%259) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %261 = "tosa.add"(%260, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %262 = "tosa.reshape"(%261) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %263 = "tosa.reduce_max"(%262) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %264 = "tosa.sub"(%262, %263) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %265 = "tosa.exp"(%264) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %266 = "tosa.reduce_sum"(%265) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %267 = "tosa.reciprocal"(%266) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %268 = "tosa.mul"(%265, %267) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %269 = "tosa.identity"(%268) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %270 = "tosa.matmul"(%269, %253) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %271 = "tosa.reshape"(%270) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %272 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %273 = "tosa.transpose"(%271, %272) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %274 = "tosa.identity"(%273) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %275 = "tosa.reshape"(%274) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %276 = "tosa.reshape"(%275) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %277 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %278 = "tosa.transpose"(%arg29, %277) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %279 = "tosa.reshape"(%276) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %280 = "tosa.reshape"(%278) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %281 = "tosa.matmul"(%279, %280) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %282 = "tosa.reshape"(%281) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %283 = "tosa.reshape"(%arg30) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %284 = "tosa.add"(%283, %282) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %285 = "tosa.reshape"(%284) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %286 = "tosa.add"(%187, %285) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %287 = "tosa.reduce_sum"(%286) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %288 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %289 = "tosa.reciprocal"(%288) : (tensor<1xf32>) -> tensor<1xf32>
    %290 = "tosa.mul"(%289, %287) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %291 = "tosa.sub"(%286, %290) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %292 = "tosa.mul"(%291, %291) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %293 = "tosa.reduce_sum"(%292) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %294 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %295 = "tosa.reciprocal"(%294) : (tensor<1xf32>) -> tensor<1xf32>
    %296 = "tosa.mul"(%295, %293) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %297 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %298 = "tosa.add"(%296, %297) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %299 = "tosa.rsqrt"(%298) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %300 = "tosa.sub"(%286, %290) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %301 = "tosa.mul"(%300, %299) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %302 = "tosa.reshape"(%arg31) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %303 = "tosa.mul"(%301, %302) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %304 = "tosa.reshape"(%arg32) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %305 = "tosa.add"(%303, %304) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %306 = "tosa.reshape"(%305) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %307 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %308 = "tosa.transpose"(%arg33, %307) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %309 = "tosa.reshape"(%306) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %310 = "tosa.reshape"(%308) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %311 = "tosa.matmul"(%309, %310) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %312 = "tosa.reshape"(%311) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %313 = "tosa.reshape"(%arg34) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %314 = "tosa.add"(%313, %312) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %315 = "tosa.reshape"(%314) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %316 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %317 = "tosa.mul"(%315, %316) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %318 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %319 = "tosa.mul"(%315, %318) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %320 = "math.erf"(%319) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %321 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %322 = "tosa.add"(%320, %321) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %323 = "tosa.mul"(%317, %322) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %324 = "tosa.reshape"(%323) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %325 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %326 = "tosa.transpose"(%arg35, %325) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %327 = "tosa.reshape"(%324) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %328 = "tosa.reshape"(%326) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %329 = "tosa.matmul"(%327, %328) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %330 = "tosa.reshape"(%329) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %331 = "tosa.reshape"(%arg36) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %332 = "tosa.add"(%331, %330) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %333 = "tosa.reshape"(%332) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %334 = "tosa.add"(%286, %333) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %335 = "tosa.reduce_sum"(%334) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %336 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %337 = "tosa.reciprocal"(%336) : (tensor<1xf32>) -> tensor<1xf32>
    %338 = "tosa.mul"(%337, %335) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %339 = "tosa.sub"(%334, %338) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %340 = "tosa.mul"(%339, %339) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %341 = "tosa.reduce_sum"(%340) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %342 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %343 = "tosa.reciprocal"(%342) : (tensor<1xf32>) -> tensor<1xf32>
    %344 = "tosa.mul"(%343, %341) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %345 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %346 = "tosa.add"(%344, %345) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %347 = "tosa.rsqrt"(%346) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %348 = "tosa.sub"(%334, %338) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %349 = "tosa.mul"(%348, %347) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %350 = "tosa.reshape"(%arg37) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %351 = "tosa.mul"(%349, %350) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %352 = "tosa.reshape"(%arg38) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %353 = "tosa.add"(%351, %352) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %354 = "tosa.reshape"(%353) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %355 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %356 = "tosa.transpose"(%arg39, %355) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %357 = "tosa.reshape"(%354) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %358 = "tosa.reshape"(%356) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %359 = "tosa.matmul"(%357, %358) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %360 = "tosa.reshape"(%359) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %361 = "tosa.reshape"(%arg40) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %362 = "tosa.add"(%361, %360) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %363 = "tosa.reshape"(%362) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %364 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %365 = "tosa.mul"(%363, %364) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %366 = "tosa.reshape"(%353) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %367 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %368 = "tosa.transpose"(%arg41, %367) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %369 = "tosa.reshape"(%366) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %370 = "tosa.reshape"(%368) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %371 = "tosa.matmul"(%369, %370) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %372 = "tosa.reshape"(%371) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %373 = "tosa.reshape"(%arg42) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %374 = "tosa.add"(%373, %372) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %375 = "tosa.reshape"(%374) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %376 = "tosa.reshape"(%375) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %377 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %378 = "tosa.transpose"(%376, %377) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %379 = "tosa.identity"(%378) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %380 = "tosa.reshape"(%353) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %381 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %382 = "tosa.transpose"(%arg43, %381) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %383 = "tosa.reshape"(%380) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %384 = "tosa.reshape"(%382) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %385 = "tosa.matmul"(%383, %384) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %386 = "tosa.reshape"(%385) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %387 = "tosa.reshape"(%arg44) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %388 = "tosa.add"(%387, %386) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %389 = "tosa.reshape"(%388) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %390 = "tosa.reshape"(%389) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %391 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %392 = "tosa.transpose"(%390, %391) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %393 = "tosa.identity"(%392) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %394 = "tosa.reshape"(%365) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %395 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %396 = "tosa.transpose"(%394, %395) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %397 = "tosa.identity"(%396) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %398 = "tosa.reshape"(%397) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %399 = "tosa.reshape"(%379) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %400 = "tosa.reshape"(%393) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %401 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %402 = "tosa.transpose"(%399, %401) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %403 = "tosa.matmul"(%398, %402) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %404 = "tosa.reshape"(%403) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %405 = "tosa.add"(%404, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %406 = "tosa.reshape"(%405) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %407 = "tosa.reshape"(%406) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %408 = "tosa.add"(%407, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %409 = "tosa.reshape"(%408) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %410 = "tosa.reduce_max"(%409) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %411 = "tosa.sub"(%409, %410) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %412 = "tosa.exp"(%411) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %413 = "tosa.reduce_sum"(%412) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %414 = "tosa.reciprocal"(%413) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %415 = "tosa.mul"(%412, %414) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %416 = "tosa.identity"(%415) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %417 = "tosa.matmul"(%416, %400) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %418 = "tosa.reshape"(%417) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %419 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %420 = "tosa.transpose"(%418, %419) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %421 = "tosa.identity"(%420) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %422 = "tosa.reshape"(%421) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %423 = "tosa.reshape"(%422) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %424 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %425 = "tosa.transpose"(%arg45, %424) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %426 = "tosa.reshape"(%423) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %427 = "tosa.reshape"(%425) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %428 = "tosa.matmul"(%426, %427) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %429 = "tosa.reshape"(%428) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %430 = "tosa.reshape"(%arg46) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %431 = "tosa.add"(%430, %429) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %432 = "tosa.reshape"(%431) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %433 = "tosa.add"(%334, %432) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %434 = "tosa.reduce_sum"(%433) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %435 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %436 = "tosa.reciprocal"(%435) : (tensor<1xf32>) -> tensor<1xf32>
    %437 = "tosa.mul"(%436, %434) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %438 = "tosa.sub"(%433, %437) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %439 = "tosa.mul"(%438, %438) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %440 = "tosa.reduce_sum"(%439) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %441 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %442 = "tosa.reciprocal"(%441) : (tensor<1xf32>) -> tensor<1xf32>
    %443 = "tosa.mul"(%442, %440) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %444 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %445 = "tosa.add"(%443, %444) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %446 = "tosa.rsqrt"(%445) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %447 = "tosa.sub"(%433, %437) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %448 = "tosa.mul"(%447, %446) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %449 = "tosa.reshape"(%arg47) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %450 = "tosa.mul"(%448, %449) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %451 = "tosa.reshape"(%arg48) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %452 = "tosa.add"(%450, %451) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %453 = "tosa.reshape"(%452) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %454 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %455 = "tosa.transpose"(%arg49, %454) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %456 = "tosa.reshape"(%453) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %457 = "tosa.reshape"(%455) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %458 = "tosa.matmul"(%456, %457) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %459 = "tosa.reshape"(%458) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %460 = "tosa.reshape"(%arg50) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %461 = "tosa.add"(%460, %459) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %462 = "tosa.reshape"(%461) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %463 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %464 = "tosa.mul"(%462, %463) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %465 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %466 = "tosa.mul"(%462, %465) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %467 = "math.erf"(%466) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %468 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %469 = "tosa.add"(%467, %468) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %470 = "tosa.mul"(%464, %469) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %471 = "tosa.reshape"(%470) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %472 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %473 = "tosa.transpose"(%arg51, %472) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %474 = "tosa.reshape"(%471) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %475 = "tosa.reshape"(%473) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %476 = "tosa.matmul"(%474, %475) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %477 = "tosa.reshape"(%476) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %478 = "tosa.reshape"(%arg52) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %479 = "tosa.add"(%478, %477) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %480 = "tosa.reshape"(%479) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %481 = "tosa.add"(%433, %480) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %482 = "tosa.reduce_sum"(%481) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %483 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %484 = "tosa.reciprocal"(%483) : (tensor<1xf32>) -> tensor<1xf32>
    %485 = "tosa.mul"(%484, %482) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %486 = "tosa.sub"(%481, %485) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %487 = "tosa.mul"(%486, %486) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %488 = "tosa.reduce_sum"(%487) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %489 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %490 = "tosa.reciprocal"(%489) : (tensor<1xf32>) -> tensor<1xf32>
    %491 = "tosa.mul"(%490, %488) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %492 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %493 = "tosa.add"(%491, %492) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %494 = "tosa.rsqrt"(%493) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %495 = "tosa.sub"(%481, %485) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %496 = "tosa.mul"(%495, %494) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %497 = "tosa.reshape"(%arg53) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %498 = "tosa.mul"(%496, %497) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %499 = "tosa.reshape"(%arg54) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %500 = "tosa.add"(%498, %499) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %501 = "tosa.reshape"(%500) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %502 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %503 = "tosa.transpose"(%arg55, %502) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %504 = "tosa.reshape"(%501) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %505 = "tosa.reshape"(%503) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %506 = "tosa.matmul"(%504, %505) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %507 = "tosa.reshape"(%506) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %508 = "tosa.reshape"(%arg56) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %509 = "tosa.add"(%508, %507) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %510 = "tosa.reshape"(%509) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %511 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %512 = "tosa.mul"(%510, %511) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %513 = "tosa.reshape"(%500) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %514 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %515 = "tosa.transpose"(%arg57, %514) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %516 = "tosa.reshape"(%513) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %517 = "tosa.reshape"(%515) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %518 = "tosa.matmul"(%516, %517) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %519 = "tosa.reshape"(%518) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %520 = "tosa.reshape"(%arg58) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %521 = "tosa.add"(%520, %519) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %522 = "tosa.reshape"(%521) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %523 = "tosa.reshape"(%522) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %524 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %525 = "tosa.transpose"(%523, %524) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %526 = "tosa.identity"(%525) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %527 = "tosa.reshape"(%500) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %528 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %529 = "tosa.transpose"(%arg59, %528) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %530 = "tosa.reshape"(%527) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %531 = "tosa.reshape"(%529) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %532 = "tosa.matmul"(%530, %531) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %533 = "tosa.reshape"(%532) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %534 = "tosa.reshape"(%arg60) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %535 = "tosa.add"(%534, %533) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %536 = "tosa.reshape"(%535) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %537 = "tosa.reshape"(%536) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %538 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %539 = "tosa.transpose"(%537, %538) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %540 = "tosa.identity"(%539) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %541 = "tosa.reshape"(%512) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %542 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %543 = "tosa.transpose"(%541, %542) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %544 = "tosa.identity"(%543) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %545 = "tosa.reshape"(%544) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %546 = "tosa.reshape"(%526) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %547 = "tosa.reshape"(%540) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %548 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %549 = "tosa.transpose"(%546, %548) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %550 = "tosa.matmul"(%545, %549) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %551 = "tosa.reshape"(%550) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %552 = "tosa.add"(%551, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %553 = "tosa.reshape"(%552) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %554 = "tosa.reshape"(%553) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %555 = "tosa.add"(%554, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %556 = "tosa.reshape"(%555) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %557 = "tosa.reduce_max"(%556) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %558 = "tosa.sub"(%556, %557) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %559 = "tosa.exp"(%558) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %560 = "tosa.reduce_sum"(%559) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %561 = "tosa.reciprocal"(%560) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %562 = "tosa.mul"(%559, %561) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %563 = "tosa.identity"(%562) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %564 = "tosa.matmul"(%563, %547) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %565 = "tosa.reshape"(%564) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %566 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %567 = "tosa.transpose"(%565, %566) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %568 = "tosa.identity"(%567) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %569 = "tosa.reshape"(%568) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %570 = "tosa.reshape"(%569) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %571 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %572 = "tosa.transpose"(%arg61, %571) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %573 = "tosa.reshape"(%570) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %574 = "tosa.reshape"(%572) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %575 = "tosa.matmul"(%573, %574) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %576 = "tosa.reshape"(%575) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %577 = "tosa.reshape"(%arg62) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %578 = "tosa.add"(%577, %576) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %579 = "tosa.reshape"(%578) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %580 = "tosa.add"(%481, %579) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %581 = "tosa.reduce_sum"(%580) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %582 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %583 = "tosa.reciprocal"(%582) : (tensor<1xf32>) -> tensor<1xf32>
    %584 = "tosa.mul"(%583, %581) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %585 = "tosa.sub"(%580, %584) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %586 = "tosa.mul"(%585, %585) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %587 = "tosa.reduce_sum"(%586) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %588 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %589 = "tosa.reciprocal"(%588) : (tensor<1xf32>) -> tensor<1xf32>
    %590 = "tosa.mul"(%589, %587) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %591 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %592 = "tosa.add"(%590, %591) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %593 = "tosa.rsqrt"(%592) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %594 = "tosa.sub"(%580, %584) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %595 = "tosa.mul"(%594, %593) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %596 = "tosa.reshape"(%arg63) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %597 = "tosa.mul"(%595, %596) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %598 = "tosa.reshape"(%arg64) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %599 = "tosa.add"(%597, %598) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %600 = "tosa.reshape"(%599) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %601 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %602 = "tosa.transpose"(%arg65, %601) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %603 = "tosa.reshape"(%600) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %604 = "tosa.reshape"(%602) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %605 = "tosa.matmul"(%603, %604) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %606 = "tosa.reshape"(%605) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %607 = "tosa.reshape"(%arg66) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %608 = "tosa.add"(%607, %606) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %609 = "tosa.reshape"(%608) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %610 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %611 = "tosa.mul"(%609, %610) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %612 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %613 = "tosa.mul"(%609, %612) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %614 = "math.erf"(%613) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %615 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %616 = "tosa.add"(%614, %615) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %617 = "tosa.mul"(%611, %616) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %618 = "tosa.reshape"(%617) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %619 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %620 = "tosa.transpose"(%arg67, %619) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %621 = "tosa.reshape"(%618) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %622 = "tosa.reshape"(%620) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %623 = "tosa.matmul"(%621, %622) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %624 = "tosa.reshape"(%623) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %625 = "tosa.reshape"(%arg68) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %626 = "tosa.add"(%625, %624) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %627 = "tosa.reshape"(%626) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %628 = "tosa.add"(%580, %627) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %629 = "tosa.reduce_sum"(%628) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %630 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %631 = "tosa.reciprocal"(%630) : (tensor<1xf32>) -> tensor<1xf32>
    %632 = "tosa.mul"(%631, %629) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %633 = "tosa.sub"(%628, %632) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %634 = "tosa.mul"(%633, %633) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %635 = "tosa.reduce_sum"(%634) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %636 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %637 = "tosa.reciprocal"(%636) : (tensor<1xf32>) -> tensor<1xf32>
    %638 = "tosa.mul"(%637, %635) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %639 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %640 = "tosa.add"(%638, %639) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %641 = "tosa.rsqrt"(%640) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %642 = "tosa.sub"(%628, %632) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %643 = "tosa.mul"(%642, %641) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %644 = "tosa.reshape"(%arg69) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %645 = "tosa.mul"(%643, %644) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %646 = "tosa.reshape"(%arg70) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %647 = "tosa.add"(%645, %646) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %648 = "tosa.reshape"(%647) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %649 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %650 = "tosa.transpose"(%arg71, %649) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %651 = "tosa.reshape"(%648) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %652 = "tosa.reshape"(%650) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %653 = "tosa.matmul"(%651, %652) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %654 = "tosa.reshape"(%653) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %655 = "tosa.reshape"(%arg72) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %656 = "tosa.add"(%655, %654) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %657 = "tosa.reshape"(%656) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %658 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %659 = "tosa.mul"(%657, %658) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %660 = "tosa.reshape"(%647) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %661 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %662 = "tosa.transpose"(%arg73, %661) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %663 = "tosa.reshape"(%660) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %664 = "tosa.reshape"(%662) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %665 = "tosa.matmul"(%663, %664) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %666 = "tosa.reshape"(%665) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %667 = "tosa.reshape"(%arg74) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %668 = "tosa.add"(%667, %666) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %669 = "tosa.reshape"(%668) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %670 = "tosa.reshape"(%669) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %671 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %672 = "tosa.transpose"(%670, %671) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %673 = "tosa.identity"(%672) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %674 = "tosa.reshape"(%647) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %675 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %676 = "tosa.transpose"(%arg75, %675) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %677 = "tosa.reshape"(%674) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %678 = "tosa.reshape"(%676) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %679 = "tosa.matmul"(%677, %678) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %680 = "tosa.reshape"(%679) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %681 = "tosa.reshape"(%arg76) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %682 = "tosa.add"(%681, %680) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %683 = "tosa.reshape"(%682) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %684 = "tosa.reshape"(%683) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %685 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %686 = "tosa.transpose"(%684, %685) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %687 = "tosa.identity"(%686) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %688 = "tosa.reshape"(%659) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %689 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %690 = "tosa.transpose"(%688, %689) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %691 = "tosa.identity"(%690) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %692 = "tosa.reshape"(%691) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %693 = "tosa.reshape"(%673) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %694 = "tosa.reshape"(%687) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %695 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %696 = "tosa.transpose"(%693, %695) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %697 = "tosa.matmul"(%692, %696) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %698 = "tosa.reshape"(%697) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %699 = "tosa.add"(%698, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %700 = "tosa.reshape"(%699) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %701 = "tosa.reshape"(%700) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %702 = "tosa.add"(%701, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %703 = "tosa.reshape"(%702) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %704 = "tosa.reduce_max"(%703) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %705 = "tosa.sub"(%703, %704) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %706 = "tosa.exp"(%705) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %707 = "tosa.reduce_sum"(%706) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %708 = "tosa.reciprocal"(%707) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %709 = "tosa.mul"(%706, %708) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %710 = "tosa.identity"(%709) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %711 = "tosa.matmul"(%710, %694) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %712 = "tosa.reshape"(%711) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %713 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %714 = "tosa.transpose"(%712, %713) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %715 = "tosa.identity"(%714) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %716 = "tosa.reshape"(%715) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %717 = "tosa.reshape"(%716) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %718 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %719 = "tosa.transpose"(%arg77, %718) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %720 = "tosa.reshape"(%717) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %721 = "tosa.reshape"(%719) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %722 = "tosa.matmul"(%720, %721) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %723 = "tosa.reshape"(%722) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %724 = "tosa.reshape"(%arg78) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %725 = "tosa.add"(%724, %723) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %726 = "tosa.reshape"(%725) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %727 = "tosa.add"(%628, %726) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %728 = "tosa.reduce_sum"(%727) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %729 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %730 = "tosa.reciprocal"(%729) : (tensor<1xf32>) -> tensor<1xf32>
    %731 = "tosa.mul"(%730, %728) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %732 = "tosa.sub"(%727, %731) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %733 = "tosa.mul"(%732, %732) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %734 = "tosa.reduce_sum"(%733) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %735 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %736 = "tosa.reciprocal"(%735) : (tensor<1xf32>) -> tensor<1xf32>
    %737 = "tosa.mul"(%736, %734) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %738 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %739 = "tosa.add"(%737, %738) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %740 = "tosa.rsqrt"(%739) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %741 = "tosa.sub"(%727, %731) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %742 = "tosa.mul"(%741, %740) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %743 = "tosa.reshape"(%arg79) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %744 = "tosa.mul"(%742, %743) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %745 = "tosa.reshape"(%arg80) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %746 = "tosa.add"(%744, %745) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %747 = "tosa.reshape"(%746) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %748 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %749 = "tosa.transpose"(%arg81, %748) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %750 = "tosa.reshape"(%747) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %751 = "tosa.reshape"(%749) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %752 = "tosa.matmul"(%750, %751) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %753 = "tosa.reshape"(%752) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %754 = "tosa.reshape"(%arg82) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %755 = "tosa.add"(%754, %753) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %756 = "tosa.reshape"(%755) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %757 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %758 = "tosa.mul"(%756, %757) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %759 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %760 = "tosa.mul"(%756, %759) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %761 = "math.erf"(%760) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %762 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %763 = "tosa.add"(%761, %762) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %764 = "tosa.mul"(%758, %763) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %765 = "tosa.reshape"(%764) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %766 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %767 = "tosa.transpose"(%arg83, %766) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %768 = "tosa.reshape"(%765) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %769 = "tosa.reshape"(%767) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %770 = "tosa.matmul"(%768, %769) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %771 = "tosa.reshape"(%770) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %772 = "tosa.reshape"(%arg84) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %773 = "tosa.add"(%772, %771) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %774 = "tosa.reshape"(%773) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %775 = "tosa.add"(%727, %774) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %776 = "tosa.reduce_sum"(%775) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %777 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %778 = "tosa.reciprocal"(%777) : (tensor<1xf32>) -> tensor<1xf32>
    %779 = "tosa.mul"(%778, %776) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %780 = "tosa.sub"(%775, %779) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %781 = "tosa.mul"(%780, %780) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %782 = "tosa.reduce_sum"(%781) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %783 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %784 = "tosa.reciprocal"(%783) : (tensor<1xf32>) -> tensor<1xf32>
    %785 = "tosa.mul"(%784, %782) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %786 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %787 = "tosa.add"(%785, %786) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %788 = "tosa.rsqrt"(%787) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %789 = "tosa.sub"(%775, %779) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %790 = "tosa.mul"(%789, %788) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %791 = "tosa.reshape"(%arg85) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %792 = "tosa.mul"(%790, %791) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %793 = "tosa.reshape"(%arg86) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %794 = "tosa.add"(%792, %793) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %795 = "tosa.reshape"(%794) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %796 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %797 = "tosa.transpose"(%arg87, %796) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %798 = "tosa.reshape"(%795) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %799 = "tosa.reshape"(%797) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %800 = "tosa.matmul"(%798, %799) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %801 = "tosa.reshape"(%800) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %802 = "tosa.reshape"(%arg88) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %803 = "tosa.add"(%802, %801) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %804 = "tosa.reshape"(%803) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %805 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %806 = "tosa.mul"(%804, %805) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %807 = "tosa.reshape"(%794) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %808 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %809 = "tosa.transpose"(%arg89, %808) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %810 = "tosa.reshape"(%807) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %811 = "tosa.reshape"(%809) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %812 = "tosa.matmul"(%810, %811) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %813 = "tosa.reshape"(%812) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %814 = "tosa.reshape"(%arg90) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %815 = "tosa.add"(%814, %813) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %816 = "tosa.reshape"(%815) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %817 = "tosa.reshape"(%816) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %818 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %819 = "tosa.transpose"(%817, %818) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %820 = "tosa.identity"(%819) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %821 = "tosa.reshape"(%794) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %822 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %823 = "tosa.transpose"(%arg91, %822) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %824 = "tosa.reshape"(%821) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %825 = "tosa.reshape"(%823) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %826 = "tosa.matmul"(%824, %825) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %827 = "tosa.reshape"(%826) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %828 = "tosa.reshape"(%arg92) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %829 = "tosa.add"(%828, %827) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %830 = "tosa.reshape"(%829) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %831 = "tosa.reshape"(%830) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %832 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %833 = "tosa.transpose"(%831, %832) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %834 = "tosa.identity"(%833) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %835 = "tosa.reshape"(%806) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %836 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %837 = "tosa.transpose"(%835, %836) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %838 = "tosa.identity"(%837) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %839 = "tosa.reshape"(%838) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %840 = "tosa.reshape"(%820) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %841 = "tosa.reshape"(%834) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %842 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %843 = "tosa.transpose"(%840, %842) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %844 = "tosa.matmul"(%839, %843) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %845 = "tosa.reshape"(%844) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %846 = "tosa.add"(%845, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %847 = "tosa.reshape"(%846) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %848 = "tosa.reshape"(%847) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %849 = "tosa.add"(%848, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %850 = "tosa.reshape"(%849) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %851 = "tosa.reduce_max"(%850) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %852 = "tosa.sub"(%850, %851) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %853 = "tosa.exp"(%852) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %854 = "tosa.reduce_sum"(%853) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %855 = "tosa.reciprocal"(%854) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %856 = "tosa.mul"(%853, %855) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %857 = "tosa.identity"(%856) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %858 = "tosa.matmul"(%857, %841) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %859 = "tosa.reshape"(%858) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %860 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %861 = "tosa.transpose"(%859, %860) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %862 = "tosa.identity"(%861) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %863 = "tosa.reshape"(%862) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %864 = "tosa.reshape"(%863) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %865 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %866 = "tosa.transpose"(%arg93, %865) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %867 = "tosa.reshape"(%864) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %868 = "tosa.reshape"(%866) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %869 = "tosa.matmul"(%867, %868) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %870 = "tosa.reshape"(%869) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %871 = "tosa.reshape"(%arg94) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %872 = "tosa.add"(%871, %870) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %873 = "tosa.reshape"(%872) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %874 = "tosa.add"(%775, %873) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %875 = "tosa.reduce_sum"(%874) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %876 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %877 = "tosa.reciprocal"(%876) : (tensor<1xf32>) -> tensor<1xf32>
    %878 = "tosa.mul"(%877, %875) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %879 = "tosa.sub"(%874, %878) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %880 = "tosa.mul"(%879, %879) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %881 = "tosa.reduce_sum"(%880) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %882 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %883 = "tosa.reciprocal"(%882) : (tensor<1xf32>) -> tensor<1xf32>
    %884 = "tosa.mul"(%883, %881) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %885 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %886 = "tosa.add"(%884, %885) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %887 = "tosa.rsqrt"(%886) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %888 = "tosa.sub"(%874, %878) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %889 = "tosa.mul"(%888, %887) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %890 = "tosa.reshape"(%arg95) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %891 = "tosa.mul"(%889, %890) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %892 = "tosa.reshape"(%arg96) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %893 = "tosa.add"(%891, %892) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %894 = "tosa.reshape"(%893) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %895 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %896 = "tosa.transpose"(%arg97, %895) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %897 = "tosa.reshape"(%894) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %898 = "tosa.reshape"(%896) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %899 = "tosa.matmul"(%897, %898) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %900 = "tosa.reshape"(%899) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %901 = "tosa.reshape"(%arg98) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %902 = "tosa.add"(%901, %900) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %903 = "tosa.reshape"(%902) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %904 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %905 = "tosa.mul"(%903, %904) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %906 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %907 = "tosa.mul"(%903, %906) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %908 = "math.erf"(%907) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %909 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %910 = "tosa.add"(%908, %909) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %911 = "tosa.mul"(%905, %910) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %912 = "tosa.reshape"(%911) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %913 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %914 = "tosa.transpose"(%arg99, %913) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %915 = "tosa.reshape"(%912) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %916 = "tosa.reshape"(%914) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %917 = "tosa.matmul"(%915, %916) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %918 = "tosa.reshape"(%917) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %919 = "tosa.reshape"(%arg100) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %920 = "tosa.add"(%919, %918) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %921 = "tosa.reshape"(%920) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %922 = "tosa.add"(%874, %921) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %923 = "tosa.reduce_sum"(%922) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %924 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %925 = "tosa.reciprocal"(%924) : (tensor<1xf32>) -> tensor<1xf32>
    %926 = "tosa.mul"(%925, %923) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %927 = "tosa.sub"(%922, %926) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %928 = "tosa.mul"(%927, %927) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %929 = "tosa.reduce_sum"(%928) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %930 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %931 = "tosa.reciprocal"(%930) : (tensor<1xf32>) -> tensor<1xf32>
    %932 = "tosa.mul"(%931, %929) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %933 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %934 = "tosa.add"(%932, %933) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %935 = "tosa.rsqrt"(%934) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %936 = "tosa.sub"(%922, %926) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %937 = "tosa.mul"(%936, %935) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %938 = "tosa.reshape"(%arg101) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %939 = "tosa.mul"(%937, %938) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %940 = "tosa.reshape"(%arg102) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %941 = "tosa.add"(%939, %940) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %942 = "tosa.reshape"(%941) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %943 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %944 = "tosa.transpose"(%arg103, %943) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %945 = "tosa.reshape"(%942) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %946 = "tosa.reshape"(%944) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %947 = "tosa.matmul"(%945, %946) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %948 = "tosa.reshape"(%947) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %949 = "tosa.reshape"(%arg104) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %950 = "tosa.add"(%949, %948) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %951 = "tosa.reshape"(%950) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %952 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %953 = "tosa.mul"(%951, %952) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %954 = "tosa.reshape"(%941) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %955 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %956 = "tosa.transpose"(%arg105, %955) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %957 = "tosa.reshape"(%954) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %958 = "tosa.reshape"(%956) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %959 = "tosa.matmul"(%957, %958) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %960 = "tosa.reshape"(%959) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %961 = "tosa.reshape"(%arg106) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %962 = "tosa.add"(%961, %960) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %963 = "tosa.reshape"(%962) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %964 = "tosa.reshape"(%963) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %965 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %966 = "tosa.transpose"(%964, %965) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %967 = "tosa.identity"(%966) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %968 = "tosa.reshape"(%941) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %969 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %970 = "tosa.transpose"(%arg107, %969) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %971 = "tosa.reshape"(%968) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %972 = "tosa.reshape"(%970) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %973 = "tosa.matmul"(%971, %972) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %974 = "tosa.reshape"(%973) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %975 = "tosa.reshape"(%arg108) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %976 = "tosa.add"(%975, %974) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %977 = "tosa.reshape"(%976) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %978 = "tosa.reshape"(%977) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %979 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %980 = "tosa.transpose"(%978, %979) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %981 = "tosa.identity"(%980) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %982 = "tosa.reshape"(%953) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %983 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %984 = "tosa.transpose"(%982, %983) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %985 = "tosa.identity"(%984) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %986 = "tosa.reshape"(%985) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %987 = "tosa.reshape"(%967) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %988 = "tosa.reshape"(%981) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %989 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %990 = "tosa.transpose"(%987, %989) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %991 = "tosa.matmul"(%986, %990) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %992 = "tosa.reshape"(%991) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %993 = "tosa.add"(%992, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %994 = "tosa.reshape"(%993) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %995 = "tosa.reshape"(%994) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %996 = "tosa.add"(%995, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %997 = "tosa.reshape"(%996) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %998 = "tosa.reduce_max"(%997) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %999 = "tosa.sub"(%997, %998) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1000 = "tosa.exp"(%999) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1001 = "tosa.reduce_sum"(%1000) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1002 = "tosa.reciprocal"(%1001) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %1003 = "tosa.mul"(%1000, %1002) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1004 = "tosa.identity"(%1003) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1005 = "tosa.matmul"(%1004, %988) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %1006 = "tosa.reshape"(%1005) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1007 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1008 = "tosa.transpose"(%1006, %1007) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %1009 = "tosa.identity"(%1008) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %1010 = "tosa.reshape"(%1009) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %1011 = "tosa.reshape"(%1010) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1012 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1013 = "tosa.transpose"(%arg109, %1012) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1014 = "tosa.reshape"(%1011) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1015 = "tosa.reshape"(%1013) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1016 = "tosa.matmul"(%1014, %1015) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1017 = "tosa.reshape"(%1016) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1018 = "tosa.reshape"(%arg110) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1019 = "tosa.add"(%1018, %1017) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1020 = "tosa.reshape"(%1019) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1021 = "tosa.add"(%922, %1020) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1022 = "tosa.reduce_sum"(%1021) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1023 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1024 = "tosa.reciprocal"(%1023) : (tensor<1xf32>) -> tensor<1xf32>
    %1025 = "tosa.mul"(%1024, %1022) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1026 = "tosa.sub"(%1021, %1025) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1027 = "tosa.mul"(%1026, %1026) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1028 = "tosa.reduce_sum"(%1027) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1029 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1030 = "tosa.reciprocal"(%1029) : (tensor<1xf32>) -> tensor<1xf32>
    %1031 = "tosa.mul"(%1030, %1028) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1032 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1033 = "tosa.add"(%1031, %1032) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1034 = "tosa.rsqrt"(%1033) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1035 = "tosa.sub"(%1021, %1025) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1036 = "tosa.mul"(%1035, %1034) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1037 = "tosa.reshape"(%arg111) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1038 = "tosa.mul"(%1036, %1037) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1039 = "tosa.reshape"(%arg112) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1040 = "tosa.add"(%1038, %1039) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1041 = "tosa.reshape"(%1040) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1042 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1043 = "tosa.transpose"(%arg113, %1042) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %1044 = "tosa.reshape"(%1041) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1045 = "tosa.reshape"(%1043) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %1046 = "tosa.matmul"(%1044, %1045) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %1047 = "tosa.reshape"(%1046) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1048 = "tosa.reshape"(%arg114) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1049 = "tosa.add"(%1048, %1047) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %1050 = "tosa.reshape"(%1049) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1051 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1052 = "tosa.mul"(%1050, %1051) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1053 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1054 = "tosa.mul"(%1050, %1053) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1055 = "math.erf"(%1054) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1056 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1057 = "tosa.add"(%1055, %1056) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1058 = "tosa.mul"(%1052, %1057) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1059 = "tosa.reshape"(%1058) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1060 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1061 = "tosa.transpose"(%arg115, %1060) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %1062 = "tosa.reshape"(%1059) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1063 = "tosa.reshape"(%1061) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %1064 = "tosa.matmul"(%1062, %1063) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %1065 = "tosa.reshape"(%1064) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1066 = "tosa.reshape"(%arg116) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1067 = "tosa.add"(%1066, %1065) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1068 = "tosa.reshape"(%1067) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1069 = "tosa.add"(%1021, %1068) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1070 = "tosa.reduce_sum"(%1069) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1071 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1072 = "tosa.reciprocal"(%1071) : (tensor<1xf32>) -> tensor<1xf32>
    %1073 = "tosa.mul"(%1072, %1070) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1074 = "tosa.sub"(%1069, %1073) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1075 = "tosa.mul"(%1074, %1074) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1076 = "tosa.reduce_sum"(%1075) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1077 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1078 = "tosa.reciprocal"(%1077) : (tensor<1xf32>) -> tensor<1xf32>
    %1079 = "tosa.mul"(%1078, %1076) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1080 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1081 = "tosa.add"(%1079, %1080) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1082 = "tosa.rsqrt"(%1081) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1083 = "tosa.sub"(%1069, %1073) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1084 = "tosa.mul"(%1083, %1082) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1085 = "tosa.reshape"(%arg117) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1086 = "tosa.mul"(%1084, %1085) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1087 = "tosa.reshape"(%arg118) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1088 = "tosa.add"(%1086, %1087) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1089 = "tosa.reshape"(%1088) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1090 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1091 = "tosa.transpose"(%arg119, %1090) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1092 = "tosa.reshape"(%1089) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1093 = "tosa.reshape"(%1091) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1094 = "tosa.matmul"(%1092, %1093) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1095 = "tosa.reshape"(%1094) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1096 = "tosa.reshape"(%arg120) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1097 = "tosa.add"(%1096, %1095) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1098 = "tosa.reshape"(%1097) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1099 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %1100 = "tosa.mul"(%1098, %1099) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1101 = "tosa.reshape"(%1088) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1102 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1103 = "tosa.transpose"(%arg121, %1102) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1104 = "tosa.reshape"(%1101) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1105 = "tosa.reshape"(%1103) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1106 = "tosa.matmul"(%1104, %1105) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1107 = "tosa.reshape"(%1106) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1108 = "tosa.reshape"(%arg122) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1109 = "tosa.add"(%1108, %1107) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1110 = "tosa.reshape"(%1109) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1111 = "tosa.reshape"(%1110) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1112 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1113 = "tosa.transpose"(%1111, %1112) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1114 = "tosa.identity"(%1113) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1115 = "tosa.reshape"(%1088) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1116 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1117 = "tosa.transpose"(%arg123, %1116) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1118 = "tosa.reshape"(%1115) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1119 = "tosa.reshape"(%1117) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1120 = "tosa.matmul"(%1118, %1119) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1121 = "tosa.reshape"(%1120) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1122 = "tosa.reshape"(%arg124) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1123 = "tosa.add"(%1122, %1121) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1124 = "tosa.reshape"(%1123) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1125 = "tosa.reshape"(%1124) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1126 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1127 = "tosa.transpose"(%1125, %1126) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1128 = "tosa.identity"(%1127) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1129 = "tosa.reshape"(%1100) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1130 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1131 = "tosa.transpose"(%1129, %1130) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1132 = "tosa.identity"(%1131) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1133 = "tosa.reshape"(%1132) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1134 = "tosa.reshape"(%1114) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1135 = "tosa.reshape"(%1128) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1136 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1137 = "tosa.transpose"(%1134, %1136) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %1138 = "tosa.matmul"(%1133, %1137) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %1139 = "tosa.reshape"(%1138) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1140 = "tosa.add"(%1139, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1141 = "tosa.reshape"(%1140) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1142 = "tosa.reshape"(%1141) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1143 = "tosa.add"(%1142, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1144 = "tosa.reshape"(%1143) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1145 = "tosa.reduce_max"(%1144) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1146 = "tosa.sub"(%1144, %1145) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1147 = "tosa.exp"(%1146) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1148 = "tosa.reduce_sum"(%1147) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1149 = "tosa.reciprocal"(%1148) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %1150 = "tosa.mul"(%1147, %1149) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1151 = "tosa.identity"(%1150) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1152 = "tosa.matmul"(%1151, %1135) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %1153 = "tosa.reshape"(%1152) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1154 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1155 = "tosa.transpose"(%1153, %1154) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %1156 = "tosa.identity"(%1155) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %1157 = "tosa.reshape"(%1156) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %1158 = "tosa.reshape"(%1157) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1159 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1160 = "tosa.transpose"(%arg125, %1159) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1161 = "tosa.reshape"(%1158) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1162 = "tosa.reshape"(%1160) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1163 = "tosa.matmul"(%1161, %1162) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1164 = "tosa.reshape"(%1163) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1165 = "tosa.reshape"(%arg126) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1166 = "tosa.add"(%1165, %1164) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1167 = "tosa.reshape"(%1166) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1168 = "tosa.add"(%1069, %1167) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1169 = "tosa.reduce_sum"(%1168) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1170 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1171 = "tosa.reciprocal"(%1170) : (tensor<1xf32>) -> tensor<1xf32>
    %1172 = "tosa.mul"(%1171, %1169) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1173 = "tosa.sub"(%1168, %1172) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1174 = "tosa.mul"(%1173, %1173) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1175 = "tosa.reduce_sum"(%1174) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1176 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1177 = "tosa.reciprocal"(%1176) : (tensor<1xf32>) -> tensor<1xf32>
    %1178 = "tosa.mul"(%1177, %1175) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1179 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1180 = "tosa.add"(%1178, %1179) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1181 = "tosa.rsqrt"(%1180) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1182 = "tosa.sub"(%1168, %1172) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1183 = "tosa.mul"(%1182, %1181) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1184 = "tosa.reshape"(%arg127) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1185 = "tosa.mul"(%1183, %1184) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1186 = "tosa.reshape"(%arg128) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1187 = "tosa.add"(%1185, %1186) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1188 = "tosa.reshape"(%1187) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1189 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1190 = "tosa.transpose"(%arg129, %1189) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %1191 = "tosa.reshape"(%1188) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1192 = "tosa.reshape"(%1190) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %1193 = "tosa.matmul"(%1191, %1192) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %1194 = "tosa.reshape"(%1193) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1195 = "tosa.reshape"(%arg130) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1196 = "tosa.add"(%1195, %1194) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %1197 = "tosa.reshape"(%1196) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1198 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1199 = "tosa.mul"(%1197, %1198) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1200 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1201 = "tosa.mul"(%1197, %1200) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1202 = "math.erf"(%1201) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1203 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1204 = "tosa.add"(%1202, %1203) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1205 = "tosa.mul"(%1199, %1204) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1206 = "tosa.reshape"(%1205) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1207 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1208 = "tosa.transpose"(%arg131, %1207) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %1209 = "tosa.reshape"(%1206) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1210 = "tosa.reshape"(%1208) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %1211 = "tosa.matmul"(%1209, %1210) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %1212 = "tosa.reshape"(%1211) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1213 = "tosa.reshape"(%arg132) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1214 = "tosa.add"(%1213, %1212) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1215 = "tosa.reshape"(%1214) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1216 = "tosa.add"(%1168, %1215) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1217 = "tosa.reduce_sum"(%1216) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1218 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1219 = "tosa.reciprocal"(%1218) : (tensor<1xf32>) -> tensor<1xf32>
    %1220 = "tosa.mul"(%1219, %1217) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1221 = "tosa.sub"(%1216, %1220) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1222 = "tosa.mul"(%1221, %1221) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1223 = "tosa.reduce_sum"(%1222) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1224 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1225 = "tosa.reciprocal"(%1224) : (tensor<1xf32>) -> tensor<1xf32>
    %1226 = "tosa.mul"(%1225, %1223) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1227 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1228 = "tosa.add"(%1226, %1227) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1229 = "tosa.rsqrt"(%1228) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1230 = "tosa.sub"(%1216, %1220) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1231 = "tosa.mul"(%1230, %1229) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1232 = "tosa.reshape"(%arg133) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1233 = "tosa.mul"(%1231, %1232) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1234 = "tosa.reshape"(%arg134) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1235 = "tosa.add"(%1233, %1234) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1236 = "tosa.reshape"(%1235) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1237 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1238 = "tosa.transpose"(%arg135, %1237) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1239 = "tosa.reshape"(%1236) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1240 = "tosa.reshape"(%1238) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1241 = "tosa.matmul"(%1239, %1240) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1242 = "tosa.reshape"(%1241) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1243 = "tosa.reshape"(%arg136) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1244 = "tosa.add"(%1243, %1242) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1245 = "tosa.reshape"(%1244) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1246 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %1247 = "tosa.mul"(%1245, %1246) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1248 = "tosa.reshape"(%1235) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1249 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1250 = "tosa.transpose"(%arg137, %1249) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1251 = "tosa.reshape"(%1248) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1252 = "tosa.reshape"(%1250) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1253 = "tosa.matmul"(%1251, %1252) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1254 = "tosa.reshape"(%1253) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1255 = "tosa.reshape"(%arg138) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1256 = "tosa.add"(%1255, %1254) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1257 = "tosa.reshape"(%1256) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1258 = "tosa.reshape"(%1257) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1259 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1260 = "tosa.transpose"(%1258, %1259) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1261 = "tosa.identity"(%1260) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1262 = "tosa.reshape"(%1235) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1263 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1264 = "tosa.transpose"(%arg139, %1263) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1265 = "tosa.reshape"(%1262) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1266 = "tosa.reshape"(%1264) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1267 = "tosa.matmul"(%1265, %1266) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1268 = "tosa.reshape"(%1267) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1269 = "tosa.reshape"(%arg140) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1270 = "tosa.add"(%1269, %1268) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1271 = "tosa.reshape"(%1270) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1272 = "tosa.reshape"(%1271) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1273 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1274 = "tosa.transpose"(%1272, %1273) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1275 = "tosa.identity"(%1274) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1276 = "tosa.reshape"(%1247) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1277 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1278 = "tosa.transpose"(%1276, %1277) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1279 = "tosa.identity"(%1278) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1280 = "tosa.reshape"(%1279) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1281 = "tosa.reshape"(%1261) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1282 = "tosa.reshape"(%1275) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1283 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1284 = "tosa.transpose"(%1281, %1283) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %1285 = "tosa.matmul"(%1280, %1284) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %1286 = "tosa.reshape"(%1285) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1287 = "tosa.add"(%1286, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1288 = "tosa.reshape"(%1287) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1289 = "tosa.reshape"(%1288) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1290 = "tosa.add"(%1289, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1291 = "tosa.reshape"(%1290) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1292 = "tosa.reduce_max"(%1291) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1293 = "tosa.sub"(%1291, %1292) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1294 = "tosa.exp"(%1293) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1295 = "tosa.reduce_sum"(%1294) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1296 = "tosa.reciprocal"(%1295) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %1297 = "tosa.mul"(%1294, %1296) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1298 = "tosa.identity"(%1297) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1299 = "tosa.matmul"(%1298, %1282) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %1300 = "tosa.reshape"(%1299) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1301 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1302 = "tosa.transpose"(%1300, %1301) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %1303 = "tosa.identity"(%1302) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %1304 = "tosa.reshape"(%1303) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %1305 = "tosa.reshape"(%1304) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1306 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1307 = "tosa.transpose"(%arg141, %1306) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1308 = "tosa.reshape"(%1305) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1309 = "tosa.reshape"(%1307) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1310 = "tosa.matmul"(%1308, %1309) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1311 = "tosa.reshape"(%1310) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1312 = "tosa.reshape"(%arg142) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1313 = "tosa.add"(%1312, %1311) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1314 = "tosa.reshape"(%1313) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1315 = "tosa.add"(%1216, %1314) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1316 = "tosa.reduce_sum"(%1315) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1317 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1318 = "tosa.reciprocal"(%1317) : (tensor<1xf32>) -> tensor<1xf32>
    %1319 = "tosa.mul"(%1318, %1316) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1320 = "tosa.sub"(%1315, %1319) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1321 = "tosa.mul"(%1320, %1320) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1322 = "tosa.reduce_sum"(%1321) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1323 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1324 = "tosa.reciprocal"(%1323) : (tensor<1xf32>) -> tensor<1xf32>
    %1325 = "tosa.mul"(%1324, %1322) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1326 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1327 = "tosa.add"(%1325, %1326) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1328 = "tosa.rsqrt"(%1327) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1329 = "tosa.sub"(%1315, %1319) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1330 = "tosa.mul"(%1329, %1328) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1331 = "tosa.reshape"(%arg143) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1332 = "tosa.mul"(%1330, %1331) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1333 = "tosa.reshape"(%arg144) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1334 = "tosa.add"(%1332, %1333) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1335 = "tosa.reshape"(%1334) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1336 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1337 = "tosa.transpose"(%arg145, %1336) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %1338 = "tosa.reshape"(%1335) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1339 = "tosa.reshape"(%1337) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %1340 = "tosa.matmul"(%1338, %1339) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %1341 = "tosa.reshape"(%1340) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1342 = "tosa.reshape"(%arg146) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1343 = "tosa.add"(%1342, %1341) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %1344 = "tosa.reshape"(%1343) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1345 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1346 = "tosa.mul"(%1344, %1345) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1347 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1348 = "tosa.mul"(%1344, %1347) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1349 = "math.erf"(%1348) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1350 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1351 = "tosa.add"(%1349, %1350) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1352 = "tosa.mul"(%1346, %1351) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1353 = "tosa.reshape"(%1352) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1354 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1355 = "tosa.transpose"(%arg147, %1354) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %1356 = "tosa.reshape"(%1353) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1357 = "tosa.reshape"(%1355) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %1358 = "tosa.matmul"(%1356, %1357) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %1359 = "tosa.reshape"(%1358) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1360 = "tosa.reshape"(%arg148) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1361 = "tosa.add"(%1360, %1359) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1362 = "tosa.reshape"(%1361) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1363 = "tosa.add"(%1315, %1362) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1364 = "tosa.reduce_sum"(%1363) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1365 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1366 = "tosa.reciprocal"(%1365) : (tensor<1xf32>) -> tensor<1xf32>
    %1367 = "tosa.mul"(%1366, %1364) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1368 = "tosa.sub"(%1363, %1367) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1369 = "tosa.mul"(%1368, %1368) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1370 = "tosa.reduce_sum"(%1369) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1371 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1372 = "tosa.reciprocal"(%1371) : (tensor<1xf32>) -> tensor<1xf32>
    %1373 = "tosa.mul"(%1372, %1370) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1374 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1375 = "tosa.add"(%1373, %1374) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1376 = "tosa.rsqrt"(%1375) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1377 = "tosa.sub"(%1363, %1367) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1378 = "tosa.mul"(%1377, %1376) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1379 = "tosa.reshape"(%arg149) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1380 = "tosa.mul"(%1378, %1379) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1381 = "tosa.reshape"(%arg150) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1382 = "tosa.add"(%1380, %1381) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1383 = "tosa.reshape"(%1382) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1384 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1385 = "tosa.transpose"(%arg151, %1384) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1386 = "tosa.reshape"(%1383) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1387 = "tosa.reshape"(%1385) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1388 = "tosa.matmul"(%1386, %1387) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1389 = "tosa.reshape"(%1388) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1390 = "tosa.reshape"(%arg152) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1391 = "tosa.add"(%1390, %1389) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1392 = "tosa.reshape"(%1391) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1393 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %1394 = "tosa.mul"(%1392, %1393) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1395 = "tosa.reshape"(%1382) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1396 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1397 = "tosa.transpose"(%arg153, %1396) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1398 = "tosa.reshape"(%1395) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1399 = "tosa.reshape"(%1397) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1400 = "tosa.matmul"(%1398, %1399) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1401 = "tosa.reshape"(%1400) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1402 = "tosa.reshape"(%arg154) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1403 = "tosa.add"(%1402, %1401) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1404 = "tosa.reshape"(%1403) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1405 = "tosa.reshape"(%1404) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1406 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1407 = "tosa.transpose"(%1405, %1406) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1408 = "tosa.identity"(%1407) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1409 = "tosa.reshape"(%1382) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1410 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1411 = "tosa.transpose"(%arg155, %1410) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1412 = "tosa.reshape"(%1409) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1413 = "tosa.reshape"(%1411) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1414 = "tosa.matmul"(%1412, %1413) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1415 = "tosa.reshape"(%1414) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1416 = "tosa.reshape"(%arg156) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1417 = "tosa.add"(%1416, %1415) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1418 = "tosa.reshape"(%1417) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1419 = "tosa.reshape"(%1418) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1420 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1421 = "tosa.transpose"(%1419, %1420) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1422 = "tosa.identity"(%1421) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1423 = "tosa.reshape"(%1394) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1424 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1425 = "tosa.transpose"(%1423, %1424) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1426 = "tosa.identity"(%1425) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1427 = "tosa.reshape"(%1426) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1428 = "tosa.reshape"(%1408) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1429 = "tosa.reshape"(%1422) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1430 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1431 = "tosa.transpose"(%1428, %1430) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %1432 = "tosa.matmul"(%1427, %1431) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %1433 = "tosa.reshape"(%1432) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1434 = "tosa.add"(%1433, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1435 = "tosa.reshape"(%1434) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1436 = "tosa.reshape"(%1435) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1437 = "tosa.add"(%1436, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1438 = "tosa.reshape"(%1437) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1439 = "tosa.reduce_max"(%1438) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1440 = "tosa.sub"(%1438, %1439) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1441 = "tosa.exp"(%1440) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1442 = "tosa.reduce_sum"(%1441) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1443 = "tosa.reciprocal"(%1442) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %1444 = "tosa.mul"(%1441, %1443) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1445 = "tosa.identity"(%1444) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1446 = "tosa.matmul"(%1445, %1429) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %1447 = "tosa.reshape"(%1446) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1448 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1449 = "tosa.transpose"(%1447, %1448) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %1450 = "tosa.identity"(%1449) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %1451 = "tosa.reshape"(%1450) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %1452 = "tosa.reshape"(%1451) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1453 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1454 = "tosa.transpose"(%arg157, %1453) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1455 = "tosa.reshape"(%1452) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1456 = "tosa.reshape"(%1454) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1457 = "tosa.matmul"(%1455, %1456) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1458 = "tosa.reshape"(%1457) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1459 = "tosa.reshape"(%arg158) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1460 = "tosa.add"(%1459, %1458) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1461 = "tosa.reshape"(%1460) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1462 = "tosa.add"(%1363, %1461) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1463 = "tosa.reduce_sum"(%1462) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1464 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1465 = "tosa.reciprocal"(%1464) : (tensor<1xf32>) -> tensor<1xf32>
    %1466 = "tosa.mul"(%1465, %1463) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1467 = "tosa.sub"(%1462, %1466) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1468 = "tosa.mul"(%1467, %1467) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1469 = "tosa.reduce_sum"(%1468) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1470 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1471 = "tosa.reciprocal"(%1470) : (tensor<1xf32>) -> tensor<1xf32>
    %1472 = "tosa.mul"(%1471, %1469) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1473 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1474 = "tosa.add"(%1472, %1473) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1475 = "tosa.rsqrt"(%1474) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1476 = "tosa.sub"(%1462, %1466) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1477 = "tosa.mul"(%1476, %1475) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1478 = "tosa.reshape"(%arg159) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1479 = "tosa.mul"(%1477, %1478) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1480 = "tosa.reshape"(%arg160) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1481 = "tosa.add"(%1479, %1480) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1482 = "tosa.reshape"(%1481) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1483 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1484 = "tosa.transpose"(%arg161, %1483) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %1485 = "tosa.reshape"(%1482) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1486 = "tosa.reshape"(%1484) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %1487 = "tosa.matmul"(%1485, %1486) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %1488 = "tosa.reshape"(%1487) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1489 = "tosa.reshape"(%arg162) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1490 = "tosa.add"(%1489, %1488) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %1491 = "tosa.reshape"(%1490) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1492 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1493 = "tosa.mul"(%1491, %1492) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1494 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1495 = "tosa.mul"(%1491, %1494) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1496 = "math.erf"(%1495) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1497 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1498 = "tosa.add"(%1496, %1497) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1499 = "tosa.mul"(%1493, %1498) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1500 = "tosa.reshape"(%1499) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1501 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1502 = "tosa.transpose"(%arg163, %1501) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %1503 = "tosa.reshape"(%1500) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1504 = "tosa.reshape"(%1502) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %1505 = "tosa.matmul"(%1503, %1504) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %1506 = "tosa.reshape"(%1505) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1507 = "tosa.reshape"(%arg164) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1508 = "tosa.add"(%1507, %1506) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1509 = "tosa.reshape"(%1508) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1510 = "tosa.add"(%1462, %1509) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1511 = "tosa.reduce_sum"(%1510) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1512 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1513 = "tosa.reciprocal"(%1512) : (tensor<1xf32>) -> tensor<1xf32>
    %1514 = "tosa.mul"(%1513, %1511) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1515 = "tosa.sub"(%1510, %1514) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1516 = "tosa.mul"(%1515, %1515) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1517 = "tosa.reduce_sum"(%1516) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1518 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1519 = "tosa.reciprocal"(%1518) : (tensor<1xf32>) -> tensor<1xf32>
    %1520 = "tosa.mul"(%1519, %1517) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1521 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1522 = "tosa.add"(%1520, %1521) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1523 = "tosa.rsqrt"(%1522) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1524 = "tosa.sub"(%1510, %1514) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1525 = "tosa.mul"(%1524, %1523) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1526 = "tosa.reshape"(%arg165) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1527 = "tosa.mul"(%1525, %1526) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1528 = "tosa.reshape"(%arg166) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1529 = "tosa.add"(%1527, %1528) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1530 = "tosa.reshape"(%1529) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1531 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1532 = "tosa.transpose"(%arg167, %1531) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1533 = "tosa.reshape"(%1530) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1534 = "tosa.reshape"(%1532) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1535 = "tosa.matmul"(%1533, %1534) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1536 = "tosa.reshape"(%1535) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1537 = "tosa.reshape"(%arg168) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1538 = "tosa.add"(%1537, %1536) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1539 = "tosa.reshape"(%1538) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1540 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %1541 = "tosa.mul"(%1539, %1540) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1542 = "tosa.reshape"(%1529) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1543 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1544 = "tosa.transpose"(%arg169, %1543) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1545 = "tosa.reshape"(%1542) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1546 = "tosa.reshape"(%1544) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1547 = "tosa.matmul"(%1545, %1546) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1548 = "tosa.reshape"(%1547) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1549 = "tosa.reshape"(%arg170) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1550 = "tosa.add"(%1549, %1548) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1551 = "tosa.reshape"(%1550) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1552 = "tosa.reshape"(%1551) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1553 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1554 = "tosa.transpose"(%1552, %1553) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1555 = "tosa.identity"(%1554) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1556 = "tosa.reshape"(%1529) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1557 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1558 = "tosa.transpose"(%arg171, %1557) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1559 = "tosa.reshape"(%1556) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1560 = "tosa.reshape"(%1558) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1561 = "tosa.matmul"(%1559, %1560) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1562 = "tosa.reshape"(%1561) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1563 = "tosa.reshape"(%arg172) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1564 = "tosa.add"(%1563, %1562) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1565 = "tosa.reshape"(%1564) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1566 = "tosa.reshape"(%1565) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1567 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1568 = "tosa.transpose"(%1566, %1567) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1569 = "tosa.identity"(%1568) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1570 = "tosa.reshape"(%1541) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1571 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1572 = "tosa.transpose"(%1570, %1571) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1573 = "tosa.identity"(%1572) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1574 = "tosa.reshape"(%1573) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1575 = "tosa.reshape"(%1555) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1576 = "tosa.reshape"(%1569) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1577 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1578 = "tosa.transpose"(%1575, %1577) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %1579 = "tosa.matmul"(%1574, %1578) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %1580 = "tosa.reshape"(%1579) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1581 = "tosa.add"(%1580, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1582 = "tosa.reshape"(%1581) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1583 = "tosa.reshape"(%1582) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1584 = "tosa.add"(%1583, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1585 = "tosa.reshape"(%1584) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1586 = "tosa.reduce_max"(%1585) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1587 = "tosa.sub"(%1585, %1586) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1588 = "tosa.exp"(%1587) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1589 = "tosa.reduce_sum"(%1588) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1590 = "tosa.reciprocal"(%1589) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %1591 = "tosa.mul"(%1588, %1590) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1592 = "tosa.identity"(%1591) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1593 = "tosa.matmul"(%1592, %1576) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %1594 = "tosa.reshape"(%1593) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1595 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1596 = "tosa.transpose"(%1594, %1595) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %1597 = "tosa.identity"(%1596) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %1598 = "tosa.reshape"(%1597) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %1599 = "tosa.reshape"(%1598) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1600 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1601 = "tosa.transpose"(%arg173, %1600) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1602 = "tosa.reshape"(%1599) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1603 = "tosa.reshape"(%1601) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1604 = "tosa.matmul"(%1602, %1603) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1605 = "tosa.reshape"(%1604) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1606 = "tosa.reshape"(%arg174) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1607 = "tosa.add"(%1606, %1605) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1608 = "tosa.reshape"(%1607) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1609 = "tosa.add"(%1510, %1608) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1610 = "tosa.reduce_sum"(%1609) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1611 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1612 = "tosa.reciprocal"(%1611) : (tensor<1xf32>) -> tensor<1xf32>
    %1613 = "tosa.mul"(%1612, %1610) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1614 = "tosa.sub"(%1609, %1613) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1615 = "tosa.mul"(%1614, %1614) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1616 = "tosa.reduce_sum"(%1615) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1617 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1618 = "tosa.reciprocal"(%1617) : (tensor<1xf32>) -> tensor<1xf32>
    %1619 = "tosa.mul"(%1618, %1616) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1620 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1621 = "tosa.add"(%1619, %1620) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1622 = "tosa.rsqrt"(%1621) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1623 = "tosa.sub"(%1609, %1613) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1624 = "tosa.mul"(%1623, %1622) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1625 = "tosa.reshape"(%arg175) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1626 = "tosa.mul"(%1624, %1625) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1627 = "tosa.reshape"(%arg176) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1628 = "tosa.add"(%1626, %1627) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1629 = "tosa.reshape"(%1628) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1630 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1631 = "tosa.transpose"(%arg177, %1630) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %1632 = "tosa.reshape"(%1629) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1633 = "tosa.reshape"(%1631) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %1634 = "tosa.matmul"(%1632, %1633) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %1635 = "tosa.reshape"(%1634) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1636 = "tosa.reshape"(%arg178) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1637 = "tosa.add"(%1636, %1635) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %1638 = "tosa.reshape"(%1637) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1639 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1640 = "tosa.mul"(%1638, %1639) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1641 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1642 = "tosa.mul"(%1638, %1641) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1643 = "math.erf"(%1642) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1644 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1645 = "tosa.add"(%1643, %1644) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1646 = "tosa.mul"(%1640, %1645) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1647 = "tosa.reshape"(%1646) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1648 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1649 = "tosa.transpose"(%arg179, %1648) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %1650 = "tosa.reshape"(%1647) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1651 = "tosa.reshape"(%1649) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %1652 = "tosa.matmul"(%1650, %1651) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %1653 = "tosa.reshape"(%1652) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1654 = "tosa.reshape"(%arg180) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1655 = "tosa.add"(%1654, %1653) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1656 = "tosa.reshape"(%1655) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1657 = "tosa.add"(%1609, %1656) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1658 = "tosa.reduce_sum"(%1657) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1659 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1660 = "tosa.reciprocal"(%1659) : (tensor<1xf32>) -> tensor<1xf32>
    %1661 = "tosa.mul"(%1660, %1658) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1662 = "tosa.sub"(%1657, %1661) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1663 = "tosa.mul"(%1662, %1662) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1664 = "tosa.reduce_sum"(%1663) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1665 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1666 = "tosa.reciprocal"(%1665) : (tensor<1xf32>) -> tensor<1xf32>
    %1667 = "tosa.mul"(%1666, %1664) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1668 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1669 = "tosa.add"(%1667, %1668) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1670 = "tosa.rsqrt"(%1669) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1671 = "tosa.sub"(%1657, %1661) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1672 = "tosa.mul"(%1671, %1670) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1673 = "tosa.reshape"(%arg181) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1674 = "tosa.mul"(%1672, %1673) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1675 = "tosa.reshape"(%arg182) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1676 = "tosa.add"(%1674, %1675) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1677 = "tosa.reshape"(%1676) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1678 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1679 = "tosa.transpose"(%arg183, %1678) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1680 = "tosa.reshape"(%1677) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1681 = "tosa.reshape"(%1679) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1682 = "tosa.matmul"(%1680, %1681) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1683 = "tosa.reshape"(%1682) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1684 = "tosa.reshape"(%arg184) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1685 = "tosa.add"(%1684, %1683) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1686 = "tosa.reshape"(%1685) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1687 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %1688 = "tosa.mul"(%1686, %1687) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1689 = "tosa.reshape"(%1676) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1690 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1691 = "tosa.transpose"(%arg185, %1690) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1692 = "tosa.reshape"(%1689) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1693 = "tosa.reshape"(%1691) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1694 = "tosa.matmul"(%1692, %1693) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1695 = "tosa.reshape"(%1694) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1696 = "tosa.reshape"(%arg186) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1697 = "tosa.add"(%1696, %1695) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1698 = "tosa.reshape"(%1697) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1699 = "tosa.reshape"(%1698) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1700 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1701 = "tosa.transpose"(%1699, %1700) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1702 = "tosa.identity"(%1701) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1703 = "tosa.reshape"(%1676) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1704 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1705 = "tosa.transpose"(%arg187, %1704) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1706 = "tosa.reshape"(%1703) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1707 = "tosa.reshape"(%1705) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1708 = "tosa.matmul"(%1706, %1707) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1709 = "tosa.reshape"(%1708) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1710 = "tosa.reshape"(%arg188) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1711 = "tosa.add"(%1710, %1709) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1712 = "tosa.reshape"(%1711) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1713 = "tosa.reshape"(%1712) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1714 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1715 = "tosa.transpose"(%1713, %1714) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1716 = "tosa.identity"(%1715) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1717 = "tosa.reshape"(%1688) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1718 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1719 = "tosa.transpose"(%1717, %1718) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1720 = "tosa.identity"(%1719) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1721 = "tosa.reshape"(%1720) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1722 = "tosa.reshape"(%1702) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1723 = "tosa.reshape"(%1716) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1724 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1725 = "tosa.transpose"(%1722, %1724) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %1726 = "tosa.matmul"(%1721, %1725) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %1727 = "tosa.reshape"(%1726) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1728 = "tosa.add"(%1727, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1729 = "tosa.reshape"(%1728) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1730 = "tosa.reshape"(%1729) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1731 = "tosa.add"(%1730, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1732 = "tosa.reshape"(%1731) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1733 = "tosa.reduce_max"(%1732) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1734 = "tosa.sub"(%1732, %1733) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1735 = "tosa.exp"(%1734) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1736 = "tosa.reduce_sum"(%1735) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1737 = "tosa.reciprocal"(%1736) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %1738 = "tosa.mul"(%1735, %1737) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1739 = "tosa.identity"(%1738) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1740 = "tosa.matmul"(%1739, %1723) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %1741 = "tosa.reshape"(%1740) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1742 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1743 = "tosa.transpose"(%1741, %1742) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %1744 = "tosa.identity"(%1743) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %1745 = "tosa.reshape"(%1744) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %1746 = "tosa.reshape"(%1745) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1747 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1748 = "tosa.transpose"(%arg189, %1747) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1749 = "tosa.reshape"(%1746) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1750 = "tosa.reshape"(%1748) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1751 = "tosa.matmul"(%1749, %1750) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1752 = "tosa.reshape"(%1751) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1753 = "tosa.reshape"(%arg190) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1754 = "tosa.add"(%1753, %1752) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1755 = "tosa.reshape"(%1754) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1756 = "tosa.add"(%1657, %1755) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1757 = "tosa.reduce_sum"(%1756) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1758 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1759 = "tosa.reciprocal"(%1758) : (tensor<1xf32>) -> tensor<1xf32>
    %1760 = "tosa.mul"(%1759, %1757) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1761 = "tosa.sub"(%1756, %1760) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1762 = "tosa.mul"(%1761, %1761) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1763 = "tosa.reduce_sum"(%1762) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1764 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1765 = "tosa.reciprocal"(%1764) : (tensor<1xf32>) -> tensor<1xf32>
    %1766 = "tosa.mul"(%1765, %1763) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1767 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1768 = "tosa.add"(%1766, %1767) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1769 = "tosa.rsqrt"(%1768) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1770 = "tosa.sub"(%1756, %1760) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1771 = "tosa.mul"(%1770, %1769) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1772 = "tosa.reshape"(%arg191) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1773 = "tosa.mul"(%1771, %1772) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1774 = "tosa.reshape"(%arg192) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1775 = "tosa.add"(%1773, %1774) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1776 = "tosa.reshape"(%1775) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1777 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1778 = "tosa.transpose"(%arg193, %1777) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %1779 = "tosa.reshape"(%1776) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1780 = "tosa.reshape"(%1778) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %1781 = "tosa.matmul"(%1779, %1780) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %1782 = "tosa.reshape"(%1781) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1783 = "tosa.reshape"(%arg194) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1784 = "tosa.add"(%1783, %1782) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %1785 = "tosa.reshape"(%1784) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1786 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1787 = "tosa.mul"(%1785, %1786) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1788 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1789 = "tosa.mul"(%1785, %1788) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1790 = "math.erf"(%1789) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1791 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1792 = "tosa.add"(%1790, %1791) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1793 = "tosa.mul"(%1787, %1792) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1794 = "tosa.reshape"(%1793) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1795 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1796 = "tosa.transpose"(%arg195, %1795) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %1797 = "tosa.reshape"(%1794) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1798 = "tosa.reshape"(%1796) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %1799 = "tosa.matmul"(%1797, %1798) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %1800 = "tosa.reshape"(%1799) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1801 = "tosa.reshape"(%arg196) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1802 = "tosa.add"(%1801, %1800) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1803 = "tosa.reshape"(%1802) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1804 = "tosa.add"(%1756, %1803) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1805 = "tosa.reduce_sum"(%1804) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1806 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1807 = "tosa.reciprocal"(%1806) : (tensor<1xf32>) -> tensor<1xf32>
    %1808 = "tosa.mul"(%1807, %1805) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1809 = "tosa.sub"(%1804, %1808) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1810 = "tosa.mul"(%1809, %1809) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1811 = "tosa.reduce_sum"(%1810) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1812 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1813 = "tosa.reciprocal"(%1812) : (tensor<1xf32>) -> tensor<1xf32>
    %1814 = "tosa.mul"(%1813, %1811) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1815 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1816 = "tosa.add"(%1814, %1815) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1817 = "tosa.rsqrt"(%1816) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1818 = "tosa.sub"(%1804, %1808) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1819 = "tosa.mul"(%1818, %1817) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1820 = "tosa.reshape"(%arg197) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1821 = "tosa.mul"(%1819, %1820) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1822 = "tosa.reshape"(%arg198) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1823 = "tosa.add"(%1821, %1822) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1824 = "tosa.reshape"(%1823) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1825 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1826 = "tosa.transpose"(%arg199, %1825) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1827 = "tosa.reshape"(%1824) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1828 = "tosa.reshape"(%1826) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1829 = "tosa.matmul"(%1827, %1828) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1830 = "tosa.reshape"(%1829) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1831 = "tosa.reshape"(%arg200) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1832 = "tosa.add"(%1831, %1830) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1833 = "tosa.reshape"(%1832) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1834 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %1835 = "tosa.mul"(%1833, %1834) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1836 = "tosa.reshape"(%1823) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1837 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1838 = "tosa.transpose"(%arg201, %1837) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1839 = "tosa.reshape"(%1836) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1840 = "tosa.reshape"(%1838) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1841 = "tosa.matmul"(%1839, %1840) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1842 = "tosa.reshape"(%1841) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1843 = "tosa.reshape"(%arg202) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1844 = "tosa.add"(%1843, %1842) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1845 = "tosa.reshape"(%1844) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1846 = "tosa.reshape"(%1845) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1847 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1848 = "tosa.transpose"(%1846, %1847) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1849 = "tosa.identity"(%1848) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1850 = "tosa.reshape"(%1823) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1851 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1852 = "tosa.transpose"(%arg203, %1851) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1853 = "tosa.reshape"(%1850) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1854 = "tosa.reshape"(%1852) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1855 = "tosa.matmul"(%1853, %1854) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1856 = "tosa.reshape"(%1855) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1857 = "tosa.reshape"(%arg204) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1858 = "tosa.add"(%1857, %1856) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1859 = "tosa.reshape"(%1858) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1860 = "tosa.reshape"(%1859) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1861 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1862 = "tosa.transpose"(%1860, %1861) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1863 = "tosa.identity"(%1862) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1864 = "tosa.reshape"(%1835) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1865 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1866 = "tosa.transpose"(%1864, %1865) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1867 = "tosa.identity"(%1866) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1868 = "tosa.reshape"(%1867) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1869 = "tosa.reshape"(%1849) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1870 = "tosa.reshape"(%1863) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %1871 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1872 = "tosa.transpose"(%1869, %1871) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %1873 = "tosa.matmul"(%1868, %1872) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %1874 = "tosa.reshape"(%1873) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1875 = "tosa.add"(%1874, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1876 = "tosa.reshape"(%1875) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1877 = "tosa.reshape"(%1876) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1878 = "tosa.add"(%1877, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %1879 = "tosa.reshape"(%1878) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %1880 = "tosa.reduce_max"(%1879) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1881 = "tosa.sub"(%1879, %1880) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1882 = "tosa.exp"(%1881) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1883 = "tosa.reduce_sum"(%1882) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %1884 = "tosa.reciprocal"(%1883) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %1885 = "tosa.mul"(%1882, %1884) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %1886 = "tosa.identity"(%1885) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %1887 = "tosa.matmul"(%1886, %1870) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %1888 = "tosa.reshape"(%1887) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1889 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1890 = "tosa.transpose"(%1888, %1889) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %1891 = "tosa.identity"(%1890) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %1892 = "tosa.reshape"(%1891) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %1893 = "tosa.reshape"(%1892) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1894 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1895 = "tosa.transpose"(%arg205, %1894) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1896 = "tosa.reshape"(%1893) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1897 = "tosa.reshape"(%1895) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1898 = "tosa.matmul"(%1896, %1897) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1899 = "tosa.reshape"(%1898) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1900 = "tosa.reshape"(%arg206) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1901 = "tosa.add"(%1900, %1899) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1902 = "tosa.reshape"(%1901) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1903 = "tosa.add"(%1804, %1902) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1904 = "tosa.reduce_sum"(%1903) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1905 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1906 = "tosa.reciprocal"(%1905) : (tensor<1xf32>) -> tensor<1xf32>
    %1907 = "tosa.mul"(%1906, %1904) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1908 = "tosa.sub"(%1903, %1907) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1909 = "tosa.mul"(%1908, %1908) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1910 = "tosa.reduce_sum"(%1909) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1911 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1912 = "tosa.reciprocal"(%1911) : (tensor<1xf32>) -> tensor<1xf32>
    %1913 = "tosa.mul"(%1912, %1910) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1914 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1915 = "tosa.add"(%1913, %1914) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1916 = "tosa.rsqrt"(%1915) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1917 = "tosa.sub"(%1903, %1907) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1918 = "tosa.mul"(%1917, %1916) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1919 = "tosa.reshape"(%arg207) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1920 = "tosa.mul"(%1918, %1919) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1921 = "tosa.reshape"(%arg208) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1922 = "tosa.add"(%1920, %1921) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1923 = "tosa.reshape"(%1922) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1924 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1925 = "tosa.transpose"(%arg209, %1924) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %1926 = "tosa.reshape"(%1923) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1927 = "tosa.reshape"(%1925) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %1928 = "tosa.matmul"(%1926, %1927) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %1929 = "tosa.reshape"(%1928) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1930 = "tosa.reshape"(%arg210) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1931 = "tosa.add"(%1930, %1929) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %1932 = "tosa.reshape"(%1931) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1933 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1934 = "tosa.mul"(%1932, %1933) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1935 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1936 = "tosa.mul"(%1932, %1935) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1937 = "math.erf"(%1936) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1938 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %1939 = "tosa.add"(%1937, %1938) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1940 = "tosa.mul"(%1934, %1939) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %1941 = "tosa.reshape"(%1940) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %1942 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1943 = "tosa.transpose"(%arg211, %1942) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %1944 = "tosa.reshape"(%1941) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %1945 = "tosa.reshape"(%1943) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %1946 = "tosa.matmul"(%1944, %1945) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %1947 = "tosa.reshape"(%1946) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1948 = "tosa.reshape"(%arg212) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1949 = "tosa.add"(%1948, %1947) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1950 = "tosa.reshape"(%1949) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1951 = "tosa.add"(%1903, %1950) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1952 = "tosa.reduce_sum"(%1951) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1953 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1954 = "tosa.reciprocal"(%1953) : (tensor<1xf32>) -> tensor<1xf32>
    %1955 = "tosa.mul"(%1954, %1952) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1956 = "tosa.sub"(%1951, %1955) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1957 = "tosa.mul"(%1956, %1956) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1958 = "tosa.reduce_sum"(%1957) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %1959 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1960 = "tosa.reciprocal"(%1959) : (tensor<1xf32>) -> tensor<1xf32>
    %1961 = "tosa.mul"(%1960, %1958) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1962 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %1963 = "tosa.add"(%1961, %1962) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1964 = "tosa.rsqrt"(%1963) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %1965 = "tosa.sub"(%1951, %1955) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1966 = "tosa.mul"(%1965, %1964) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %1967 = "tosa.reshape"(%arg213) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1968 = "tosa.mul"(%1966, %1967) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1969 = "tosa.reshape"(%arg214) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1970 = "tosa.add"(%1968, %1969) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %1971 = "tosa.reshape"(%1970) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1972 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1973 = "tosa.transpose"(%arg215, %1972) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1974 = "tosa.reshape"(%1971) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1975 = "tosa.reshape"(%1973) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1976 = "tosa.matmul"(%1974, %1975) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1977 = "tosa.reshape"(%1976) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1978 = "tosa.reshape"(%arg216) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1979 = "tosa.add"(%1978, %1977) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1980 = "tosa.reshape"(%1979) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1981 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %1982 = "tosa.mul"(%1980, %1981) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %1983 = "tosa.reshape"(%1970) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1984 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1985 = "tosa.transpose"(%arg217, %1984) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %1986 = "tosa.reshape"(%1983) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1987 = "tosa.reshape"(%1985) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1988 = "tosa.matmul"(%1986, %1987) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %1989 = "tosa.reshape"(%1988) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1990 = "tosa.reshape"(%arg218) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %1991 = "tosa.add"(%1990, %1989) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %1992 = "tosa.reshape"(%1991) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %1993 = "tosa.reshape"(%1992) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %1994 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1995 = "tosa.transpose"(%1993, %1994) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %1996 = "tosa.identity"(%1995) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %1997 = "tosa.reshape"(%1970) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1998 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1999 = "tosa.transpose"(%arg219, %1998) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2000 = "tosa.reshape"(%1997) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2001 = "tosa.reshape"(%1999) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2002 = "tosa.matmul"(%2000, %2001) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2003 = "tosa.reshape"(%2002) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2004 = "tosa.reshape"(%arg220) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2005 = "tosa.add"(%2004, %2003) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2006 = "tosa.reshape"(%2005) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2007 = "tosa.reshape"(%2006) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2008 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2009 = "tosa.transpose"(%2007, %2008) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2010 = "tosa.identity"(%2009) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2011 = "tosa.reshape"(%1982) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2012 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2013 = "tosa.transpose"(%2011, %2012) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2014 = "tosa.identity"(%2013) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2015 = "tosa.reshape"(%2014) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2016 = "tosa.reshape"(%1996) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2017 = "tosa.reshape"(%2010) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2018 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %2019 = "tosa.transpose"(%2016, %2018) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %2020 = "tosa.matmul"(%2015, %2019) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %2021 = "tosa.reshape"(%2020) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2022 = "tosa.add"(%2021, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2023 = "tosa.reshape"(%2022) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2024 = "tosa.reshape"(%2023) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2025 = "tosa.add"(%2024, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2026 = "tosa.reshape"(%2025) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2027 = "tosa.reduce_max"(%2026) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2028 = "tosa.sub"(%2026, %2027) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2029 = "tosa.exp"(%2028) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2030 = "tosa.reduce_sum"(%2029) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2031 = "tosa.reciprocal"(%2030) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %2032 = "tosa.mul"(%2029, %2031) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2033 = "tosa.identity"(%2032) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2034 = "tosa.matmul"(%2033, %2017) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %2035 = "tosa.reshape"(%2034) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2036 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2037 = "tosa.transpose"(%2035, %2036) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %2038 = "tosa.identity"(%2037) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %2039 = "tosa.reshape"(%2038) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %2040 = "tosa.reshape"(%2039) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2041 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2042 = "tosa.transpose"(%arg221, %2041) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2043 = "tosa.reshape"(%2040) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2044 = "tosa.reshape"(%2042) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2045 = "tosa.matmul"(%2043, %2044) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2046 = "tosa.reshape"(%2045) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2047 = "tosa.reshape"(%arg222) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2048 = "tosa.add"(%2047, %2046) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2049 = "tosa.reshape"(%2048) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2050 = "tosa.add"(%1951, %2049) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2051 = "tosa.reduce_sum"(%2050) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2052 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2053 = "tosa.reciprocal"(%2052) : (tensor<1xf32>) -> tensor<1xf32>
    %2054 = "tosa.mul"(%2053, %2051) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2055 = "tosa.sub"(%2050, %2054) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2056 = "tosa.mul"(%2055, %2055) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2057 = "tosa.reduce_sum"(%2056) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2058 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2059 = "tosa.reciprocal"(%2058) : (tensor<1xf32>) -> tensor<1xf32>
    %2060 = "tosa.mul"(%2059, %2057) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2061 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2062 = "tosa.add"(%2060, %2061) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2063 = "tosa.rsqrt"(%2062) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2064 = "tosa.sub"(%2050, %2054) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2065 = "tosa.mul"(%2064, %2063) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2066 = "tosa.reshape"(%arg223) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2067 = "tosa.mul"(%2065, %2066) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2068 = "tosa.reshape"(%arg224) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2069 = "tosa.add"(%2067, %2068) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2070 = "tosa.reshape"(%2069) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2071 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2072 = "tosa.transpose"(%arg225, %2071) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %2073 = "tosa.reshape"(%2070) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2074 = "tosa.reshape"(%2072) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %2075 = "tosa.matmul"(%2073, %2074) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %2076 = "tosa.reshape"(%2075) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2077 = "tosa.reshape"(%arg226) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2078 = "tosa.add"(%2077, %2076) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %2079 = "tosa.reshape"(%2078) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2080 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2081 = "tosa.mul"(%2079, %2080) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2082 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2083 = "tosa.mul"(%2079, %2082) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2084 = "math.erf"(%2083) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2085 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2086 = "tosa.add"(%2084, %2085) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2087 = "tosa.mul"(%2081, %2086) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2088 = "tosa.reshape"(%2087) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2089 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2090 = "tosa.transpose"(%arg227, %2089) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %2091 = "tosa.reshape"(%2088) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2092 = "tosa.reshape"(%2090) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %2093 = "tosa.matmul"(%2091, %2092) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %2094 = "tosa.reshape"(%2093) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2095 = "tosa.reshape"(%arg228) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2096 = "tosa.add"(%2095, %2094) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2097 = "tosa.reshape"(%2096) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2098 = "tosa.add"(%2050, %2097) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2099 = "tosa.reduce_sum"(%2098) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2100 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2101 = "tosa.reciprocal"(%2100) : (tensor<1xf32>) -> tensor<1xf32>
    %2102 = "tosa.mul"(%2101, %2099) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2103 = "tosa.sub"(%2098, %2102) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2104 = "tosa.mul"(%2103, %2103) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2105 = "tosa.reduce_sum"(%2104) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2106 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2107 = "tosa.reciprocal"(%2106) : (tensor<1xf32>) -> tensor<1xf32>
    %2108 = "tosa.mul"(%2107, %2105) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2109 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2110 = "tosa.add"(%2108, %2109) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2111 = "tosa.rsqrt"(%2110) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2112 = "tosa.sub"(%2098, %2102) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2113 = "tosa.mul"(%2112, %2111) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2114 = "tosa.reshape"(%arg229) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2115 = "tosa.mul"(%2113, %2114) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2116 = "tosa.reshape"(%arg230) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2117 = "tosa.add"(%2115, %2116) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2118 = "tosa.reshape"(%2117) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2119 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2120 = "tosa.transpose"(%arg231, %2119) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2121 = "tosa.reshape"(%2118) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2122 = "tosa.reshape"(%2120) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2123 = "tosa.matmul"(%2121, %2122) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2124 = "tosa.reshape"(%2123) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2125 = "tosa.reshape"(%arg232) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2126 = "tosa.add"(%2125, %2124) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2127 = "tosa.reshape"(%2126) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2128 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %2129 = "tosa.mul"(%2127, %2128) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2130 = "tosa.reshape"(%2117) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2131 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2132 = "tosa.transpose"(%arg233, %2131) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2133 = "tosa.reshape"(%2130) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2134 = "tosa.reshape"(%2132) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2135 = "tosa.matmul"(%2133, %2134) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2136 = "tosa.reshape"(%2135) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2137 = "tosa.reshape"(%arg234) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2138 = "tosa.add"(%2137, %2136) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2139 = "tosa.reshape"(%2138) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2140 = "tosa.reshape"(%2139) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2141 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2142 = "tosa.transpose"(%2140, %2141) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2143 = "tosa.identity"(%2142) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2144 = "tosa.reshape"(%2117) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2145 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2146 = "tosa.transpose"(%arg235, %2145) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2147 = "tosa.reshape"(%2144) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2148 = "tosa.reshape"(%2146) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2149 = "tosa.matmul"(%2147, %2148) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2150 = "tosa.reshape"(%2149) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2151 = "tosa.reshape"(%arg236) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2152 = "tosa.add"(%2151, %2150) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2153 = "tosa.reshape"(%2152) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2154 = "tosa.reshape"(%2153) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2155 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2156 = "tosa.transpose"(%2154, %2155) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2157 = "tosa.identity"(%2156) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2158 = "tosa.reshape"(%2129) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2159 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2160 = "tosa.transpose"(%2158, %2159) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2161 = "tosa.identity"(%2160) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2162 = "tosa.reshape"(%2161) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2163 = "tosa.reshape"(%2143) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2164 = "tosa.reshape"(%2157) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2165 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %2166 = "tosa.transpose"(%2163, %2165) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %2167 = "tosa.matmul"(%2162, %2166) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %2168 = "tosa.reshape"(%2167) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2169 = "tosa.add"(%2168, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2170 = "tosa.reshape"(%2169) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2171 = "tosa.reshape"(%2170) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2172 = "tosa.add"(%2171, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2173 = "tosa.reshape"(%2172) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2174 = "tosa.reduce_max"(%2173) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2175 = "tosa.sub"(%2173, %2174) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2176 = "tosa.exp"(%2175) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2177 = "tosa.reduce_sum"(%2176) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2178 = "tosa.reciprocal"(%2177) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %2179 = "tosa.mul"(%2176, %2178) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2180 = "tosa.identity"(%2179) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2181 = "tosa.matmul"(%2180, %2164) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %2182 = "tosa.reshape"(%2181) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2183 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2184 = "tosa.transpose"(%2182, %2183) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %2185 = "tosa.identity"(%2184) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %2186 = "tosa.reshape"(%2185) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %2187 = "tosa.reshape"(%2186) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2188 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2189 = "tosa.transpose"(%arg237, %2188) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2190 = "tosa.reshape"(%2187) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2191 = "tosa.reshape"(%2189) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2192 = "tosa.matmul"(%2190, %2191) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2193 = "tosa.reshape"(%2192) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2194 = "tosa.reshape"(%arg238) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2195 = "tosa.add"(%2194, %2193) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2196 = "tosa.reshape"(%2195) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2197 = "tosa.add"(%2098, %2196) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2198 = "tosa.reduce_sum"(%2197) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2199 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2200 = "tosa.reciprocal"(%2199) : (tensor<1xf32>) -> tensor<1xf32>
    %2201 = "tosa.mul"(%2200, %2198) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2202 = "tosa.sub"(%2197, %2201) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2203 = "tosa.mul"(%2202, %2202) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2204 = "tosa.reduce_sum"(%2203) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2205 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2206 = "tosa.reciprocal"(%2205) : (tensor<1xf32>) -> tensor<1xf32>
    %2207 = "tosa.mul"(%2206, %2204) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2208 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2209 = "tosa.add"(%2207, %2208) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2210 = "tosa.rsqrt"(%2209) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2211 = "tosa.sub"(%2197, %2201) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2212 = "tosa.mul"(%2211, %2210) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2213 = "tosa.reshape"(%arg239) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2214 = "tosa.mul"(%2212, %2213) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2215 = "tosa.reshape"(%arg240) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2216 = "tosa.add"(%2214, %2215) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2217 = "tosa.reshape"(%2216) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2218 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2219 = "tosa.transpose"(%arg241, %2218) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %2220 = "tosa.reshape"(%2217) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2221 = "tosa.reshape"(%2219) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %2222 = "tosa.matmul"(%2220, %2221) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %2223 = "tosa.reshape"(%2222) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2224 = "tosa.reshape"(%arg242) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2225 = "tosa.add"(%2224, %2223) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %2226 = "tosa.reshape"(%2225) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2227 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2228 = "tosa.mul"(%2226, %2227) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2229 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2230 = "tosa.mul"(%2226, %2229) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2231 = "math.erf"(%2230) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2232 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2233 = "tosa.add"(%2231, %2232) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2234 = "tosa.mul"(%2228, %2233) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2235 = "tosa.reshape"(%2234) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2236 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2237 = "tosa.transpose"(%arg243, %2236) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %2238 = "tosa.reshape"(%2235) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2239 = "tosa.reshape"(%2237) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %2240 = "tosa.matmul"(%2238, %2239) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %2241 = "tosa.reshape"(%2240) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2242 = "tosa.reshape"(%arg244) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2243 = "tosa.add"(%2242, %2241) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2244 = "tosa.reshape"(%2243) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2245 = "tosa.add"(%2197, %2244) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2246 = "tosa.reduce_sum"(%2245) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2247 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2248 = "tosa.reciprocal"(%2247) : (tensor<1xf32>) -> tensor<1xf32>
    %2249 = "tosa.mul"(%2248, %2246) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2250 = "tosa.sub"(%2245, %2249) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2251 = "tosa.mul"(%2250, %2250) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2252 = "tosa.reduce_sum"(%2251) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2253 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2254 = "tosa.reciprocal"(%2253) : (tensor<1xf32>) -> tensor<1xf32>
    %2255 = "tosa.mul"(%2254, %2252) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2256 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2257 = "tosa.add"(%2255, %2256) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2258 = "tosa.rsqrt"(%2257) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2259 = "tosa.sub"(%2245, %2249) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2260 = "tosa.mul"(%2259, %2258) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2261 = "tosa.reshape"(%arg245) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2262 = "tosa.mul"(%2260, %2261) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2263 = "tosa.reshape"(%arg246) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2264 = "tosa.add"(%2262, %2263) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2265 = "tosa.reshape"(%2264) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2266 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2267 = "tosa.transpose"(%arg247, %2266) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2268 = "tosa.reshape"(%2265) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2269 = "tosa.reshape"(%2267) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2270 = "tosa.matmul"(%2268, %2269) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2271 = "tosa.reshape"(%2270) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2272 = "tosa.reshape"(%arg248) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2273 = "tosa.add"(%2272, %2271) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2274 = "tosa.reshape"(%2273) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2275 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %2276 = "tosa.mul"(%2274, %2275) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2277 = "tosa.reshape"(%2264) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2278 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2279 = "tosa.transpose"(%arg249, %2278) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2280 = "tosa.reshape"(%2277) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2281 = "tosa.reshape"(%2279) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2282 = "tosa.matmul"(%2280, %2281) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2283 = "tosa.reshape"(%2282) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2284 = "tosa.reshape"(%arg250) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2285 = "tosa.add"(%2284, %2283) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2286 = "tosa.reshape"(%2285) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2287 = "tosa.reshape"(%2286) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2288 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2289 = "tosa.transpose"(%2287, %2288) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2290 = "tosa.identity"(%2289) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2291 = "tosa.reshape"(%2264) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2292 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2293 = "tosa.transpose"(%arg251, %2292) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2294 = "tosa.reshape"(%2291) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2295 = "tosa.reshape"(%2293) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2296 = "tosa.matmul"(%2294, %2295) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2297 = "tosa.reshape"(%2296) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2298 = "tosa.reshape"(%arg252) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2299 = "tosa.add"(%2298, %2297) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2300 = "tosa.reshape"(%2299) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2301 = "tosa.reshape"(%2300) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2302 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2303 = "tosa.transpose"(%2301, %2302) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2304 = "tosa.identity"(%2303) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2305 = "tosa.reshape"(%2276) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2306 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2307 = "tosa.transpose"(%2305, %2306) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2308 = "tosa.identity"(%2307) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2309 = "tosa.reshape"(%2308) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2310 = "tosa.reshape"(%2290) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2311 = "tosa.reshape"(%2304) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2312 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %2313 = "tosa.transpose"(%2310, %2312) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %2314 = "tosa.matmul"(%2309, %2313) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %2315 = "tosa.reshape"(%2314) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2316 = "tosa.add"(%2315, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2317 = "tosa.reshape"(%2316) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2318 = "tosa.reshape"(%2317) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2319 = "tosa.add"(%2318, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2320 = "tosa.reshape"(%2319) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2321 = "tosa.reduce_max"(%2320) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2322 = "tosa.sub"(%2320, %2321) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2323 = "tosa.exp"(%2322) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2324 = "tosa.reduce_sum"(%2323) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2325 = "tosa.reciprocal"(%2324) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %2326 = "tosa.mul"(%2323, %2325) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2327 = "tosa.identity"(%2326) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2328 = "tosa.matmul"(%2327, %2311) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %2329 = "tosa.reshape"(%2328) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2330 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2331 = "tosa.transpose"(%2329, %2330) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %2332 = "tosa.identity"(%2331) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %2333 = "tosa.reshape"(%2332) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %2334 = "tosa.reshape"(%2333) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2335 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2336 = "tosa.transpose"(%arg253, %2335) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2337 = "tosa.reshape"(%2334) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2338 = "tosa.reshape"(%2336) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2339 = "tosa.matmul"(%2337, %2338) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2340 = "tosa.reshape"(%2339) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2341 = "tosa.reshape"(%arg254) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2342 = "tosa.add"(%2341, %2340) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2343 = "tosa.reshape"(%2342) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2344 = "tosa.add"(%2245, %2343) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2345 = "tosa.reduce_sum"(%2344) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2346 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2347 = "tosa.reciprocal"(%2346) : (tensor<1xf32>) -> tensor<1xf32>
    %2348 = "tosa.mul"(%2347, %2345) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2349 = "tosa.sub"(%2344, %2348) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2350 = "tosa.mul"(%2349, %2349) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2351 = "tosa.reduce_sum"(%2350) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2352 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2353 = "tosa.reciprocal"(%2352) : (tensor<1xf32>) -> tensor<1xf32>
    %2354 = "tosa.mul"(%2353, %2351) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2355 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2356 = "tosa.add"(%2354, %2355) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2357 = "tosa.rsqrt"(%2356) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2358 = "tosa.sub"(%2344, %2348) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2359 = "tosa.mul"(%2358, %2357) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2360 = "tosa.reshape"(%arg255) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2361 = "tosa.mul"(%2359, %2360) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2362 = "tosa.reshape"(%arg256) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2363 = "tosa.add"(%2361, %2362) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2364 = "tosa.reshape"(%2363) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2365 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2366 = "tosa.transpose"(%arg257, %2365) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %2367 = "tosa.reshape"(%2364) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2368 = "tosa.reshape"(%2366) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %2369 = "tosa.matmul"(%2367, %2368) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %2370 = "tosa.reshape"(%2369) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2371 = "tosa.reshape"(%arg258) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2372 = "tosa.add"(%2371, %2370) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %2373 = "tosa.reshape"(%2372) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2374 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2375 = "tosa.mul"(%2373, %2374) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2376 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2377 = "tosa.mul"(%2373, %2376) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2378 = "math.erf"(%2377) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2379 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2380 = "tosa.add"(%2378, %2379) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2381 = "tosa.mul"(%2375, %2380) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2382 = "tosa.reshape"(%2381) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2383 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2384 = "tosa.transpose"(%arg259, %2383) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %2385 = "tosa.reshape"(%2382) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2386 = "tosa.reshape"(%2384) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %2387 = "tosa.matmul"(%2385, %2386) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %2388 = "tosa.reshape"(%2387) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2389 = "tosa.reshape"(%arg260) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2390 = "tosa.add"(%2389, %2388) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2391 = "tosa.reshape"(%2390) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2392 = "tosa.add"(%2344, %2391) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2393 = "tosa.reduce_sum"(%2392) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2394 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2395 = "tosa.reciprocal"(%2394) : (tensor<1xf32>) -> tensor<1xf32>
    %2396 = "tosa.mul"(%2395, %2393) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2397 = "tosa.sub"(%2392, %2396) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2398 = "tosa.mul"(%2397, %2397) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2399 = "tosa.reduce_sum"(%2398) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2400 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2401 = "tosa.reciprocal"(%2400) : (tensor<1xf32>) -> tensor<1xf32>
    %2402 = "tosa.mul"(%2401, %2399) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2403 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2404 = "tosa.add"(%2402, %2403) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2405 = "tosa.rsqrt"(%2404) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2406 = "tosa.sub"(%2392, %2396) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2407 = "tosa.mul"(%2406, %2405) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2408 = "tosa.reshape"(%arg261) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2409 = "tosa.mul"(%2407, %2408) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2410 = "tosa.reshape"(%arg262) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2411 = "tosa.add"(%2409, %2410) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2412 = "tosa.reshape"(%2411) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2413 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2414 = "tosa.transpose"(%arg263, %2413) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2415 = "tosa.reshape"(%2412) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2416 = "tosa.reshape"(%2414) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2417 = "tosa.matmul"(%2415, %2416) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2418 = "tosa.reshape"(%2417) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2419 = "tosa.reshape"(%arg264) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2420 = "tosa.add"(%2419, %2418) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2421 = "tosa.reshape"(%2420) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2422 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %2423 = "tosa.mul"(%2421, %2422) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2424 = "tosa.reshape"(%2411) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2425 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2426 = "tosa.transpose"(%arg265, %2425) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2427 = "tosa.reshape"(%2424) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2428 = "tosa.reshape"(%2426) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2429 = "tosa.matmul"(%2427, %2428) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2430 = "tosa.reshape"(%2429) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2431 = "tosa.reshape"(%arg266) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2432 = "tosa.add"(%2431, %2430) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2433 = "tosa.reshape"(%2432) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2434 = "tosa.reshape"(%2433) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2435 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2436 = "tosa.transpose"(%2434, %2435) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2437 = "tosa.identity"(%2436) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2438 = "tosa.reshape"(%2411) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2439 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2440 = "tosa.transpose"(%arg267, %2439) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2441 = "tosa.reshape"(%2438) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2442 = "tosa.reshape"(%2440) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2443 = "tosa.matmul"(%2441, %2442) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2444 = "tosa.reshape"(%2443) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2445 = "tosa.reshape"(%arg268) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2446 = "tosa.add"(%2445, %2444) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2447 = "tosa.reshape"(%2446) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2448 = "tosa.reshape"(%2447) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2449 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2450 = "tosa.transpose"(%2448, %2449) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2451 = "tosa.identity"(%2450) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2452 = "tosa.reshape"(%2423) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2453 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2454 = "tosa.transpose"(%2452, %2453) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2455 = "tosa.identity"(%2454) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2456 = "tosa.reshape"(%2455) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2457 = "tosa.reshape"(%2437) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2458 = "tosa.reshape"(%2451) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2459 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %2460 = "tosa.transpose"(%2457, %2459) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %2461 = "tosa.matmul"(%2456, %2460) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %2462 = "tosa.reshape"(%2461) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2463 = "tosa.add"(%2462, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2464 = "tosa.reshape"(%2463) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2465 = "tosa.reshape"(%2464) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2466 = "tosa.add"(%2465, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2467 = "tosa.reshape"(%2466) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2468 = "tosa.reduce_max"(%2467) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2469 = "tosa.sub"(%2467, %2468) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2470 = "tosa.exp"(%2469) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2471 = "tosa.reduce_sum"(%2470) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2472 = "tosa.reciprocal"(%2471) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %2473 = "tosa.mul"(%2470, %2472) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2474 = "tosa.identity"(%2473) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2475 = "tosa.matmul"(%2474, %2458) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %2476 = "tosa.reshape"(%2475) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2477 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2478 = "tosa.transpose"(%2476, %2477) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %2479 = "tosa.identity"(%2478) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %2480 = "tosa.reshape"(%2479) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %2481 = "tosa.reshape"(%2480) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2482 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2483 = "tosa.transpose"(%arg269, %2482) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2484 = "tosa.reshape"(%2481) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2485 = "tosa.reshape"(%2483) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2486 = "tosa.matmul"(%2484, %2485) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2487 = "tosa.reshape"(%2486) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2488 = "tosa.reshape"(%arg270) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2489 = "tosa.add"(%2488, %2487) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2490 = "tosa.reshape"(%2489) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2491 = "tosa.add"(%2392, %2490) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2492 = "tosa.reduce_sum"(%2491) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2493 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2494 = "tosa.reciprocal"(%2493) : (tensor<1xf32>) -> tensor<1xf32>
    %2495 = "tosa.mul"(%2494, %2492) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2496 = "tosa.sub"(%2491, %2495) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2497 = "tosa.mul"(%2496, %2496) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2498 = "tosa.reduce_sum"(%2497) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2499 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2500 = "tosa.reciprocal"(%2499) : (tensor<1xf32>) -> tensor<1xf32>
    %2501 = "tosa.mul"(%2500, %2498) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2502 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2503 = "tosa.add"(%2501, %2502) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2504 = "tosa.rsqrt"(%2503) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2505 = "tosa.sub"(%2491, %2495) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2506 = "tosa.mul"(%2505, %2504) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2507 = "tosa.reshape"(%arg271) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2508 = "tosa.mul"(%2506, %2507) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2509 = "tosa.reshape"(%arg272) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2510 = "tosa.add"(%2508, %2509) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2511 = "tosa.reshape"(%2510) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2512 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2513 = "tosa.transpose"(%arg273, %2512) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %2514 = "tosa.reshape"(%2511) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2515 = "tosa.reshape"(%2513) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %2516 = "tosa.matmul"(%2514, %2515) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %2517 = "tosa.reshape"(%2516) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2518 = "tosa.reshape"(%arg274) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2519 = "tosa.add"(%2518, %2517) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %2520 = "tosa.reshape"(%2519) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2521 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2522 = "tosa.mul"(%2520, %2521) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2523 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2524 = "tosa.mul"(%2520, %2523) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2525 = "math.erf"(%2524) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2526 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2527 = "tosa.add"(%2525, %2526) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2528 = "tosa.mul"(%2522, %2527) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2529 = "tosa.reshape"(%2528) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2530 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2531 = "tosa.transpose"(%arg275, %2530) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %2532 = "tosa.reshape"(%2529) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2533 = "tosa.reshape"(%2531) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %2534 = "tosa.matmul"(%2532, %2533) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %2535 = "tosa.reshape"(%2534) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2536 = "tosa.reshape"(%arg276) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2537 = "tosa.add"(%2536, %2535) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2538 = "tosa.reshape"(%2537) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2539 = "tosa.add"(%2491, %2538) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2540 = "tosa.reduce_sum"(%2539) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2541 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2542 = "tosa.reciprocal"(%2541) : (tensor<1xf32>) -> tensor<1xf32>
    %2543 = "tosa.mul"(%2542, %2540) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2544 = "tosa.sub"(%2539, %2543) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2545 = "tosa.mul"(%2544, %2544) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2546 = "tosa.reduce_sum"(%2545) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2547 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2548 = "tosa.reciprocal"(%2547) : (tensor<1xf32>) -> tensor<1xf32>
    %2549 = "tosa.mul"(%2548, %2546) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2550 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2551 = "tosa.add"(%2549, %2550) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2552 = "tosa.rsqrt"(%2551) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2553 = "tosa.sub"(%2539, %2543) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2554 = "tosa.mul"(%2553, %2552) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2555 = "tosa.reshape"(%arg277) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2556 = "tosa.mul"(%2554, %2555) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2557 = "tosa.reshape"(%arg278) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2558 = "tosa.add"(%2556, %2557) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2559 = "tosa.reshape"(%2558) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2560 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2561 = "tosa.transpose"(%arg279, %2560) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2562 = "tosa.reshape"(%2559) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2563 = "tosa.reshape"(%2561) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2564 = "tosa.matmul"(%2562, %2563) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2565 = "tosa.reshape"(%2564) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2566 = "tosa.reshape"(%arg280) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2567 = "tosa.add"(%2566, %2565) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2568 = "tosa.reshape"(%2567) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2569 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %2570 = "tosa.mul"(%2568, %2569) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2571 = "tosa.reshape"(%2558) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2572 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2573 = "tosa.transpose"(%arg281, %2572) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2574 = "tosa.reshape"(%2571) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2575 = "tosa.reshape"(%2573) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2576 = "tosa.matmul"(%2574, %2575) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2577 = "tosa.reshape"(%2576) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2578 = "tosa.reshape"(%arg282) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2579 = "tosa.add"(%2578, %2577) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2580 = "tosa.reshape"(%2579) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2581 = "tosa.reshape"(%2580) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2582 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2583 = "tosa.transpose"(%2581, %2582) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2584 = "tosa.identity"(%2583) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2585 = "tosa.reshape"(%2558) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2586 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2587 = "tosa.transpose"(%arg283, %2586) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2588 = "tosa.reshape"(%2585) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2589 = "tosa.reshape"(%2587) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2590 = "tosa.matmul"(%2588, %2589) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2591 = "tosa.reshape"(%2590) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2592 = "tosa.reshape"(%arg284) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2593 = "tosa.add"(%2592, %2591) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2594 = "tosa.reshape"(%2593) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2595 = "tosa.reshape"(%2594) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2596 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2597 = "tosa.transpose"(%2595, %2596) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2598 = "tosa.identity"(%2597) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2599 = "tosa.reshape"(%2570) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2600 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2601 = "tosa.transpose"(%2599, %2600) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2602 = "tosa.identity"(%2601) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2603 = "tosa.reshape"(%2602) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2604 = "tosa.reshape"(%2584) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2605 = "tosa.reshape"(%2598) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2606 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %2607 = "tosa.transpose"(%2604, %2606) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %2608 = "tosa.matmul"(%2603, %2607) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %2609 = "tosa.reshape"(%2608) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2610 = "tosa.add"(%2609, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2611 = "tosa.reshape"(%2610) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2612 = "tosa.reshape"(%2611) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2613 = "tosa.add"(%2612, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2614 = "tosa.reshape"(%2613) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2615 = "tosa.reduce_max"(%2614) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2616 = "tosa.sub"(%2614, %2615) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2617 = "tosa.exp"(%2616) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2618 = "tosa.reduce_sum"(%2617) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2619 = "tosa.reciprocal"(%2618) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %2620 = "tosa.mul"(%2617, %2619) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2621 = "tosa.identity"(%2620) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2622 = "tosa.matmul"(%2621, %2605) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %2623 = "tosa.reshape"(%2622) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2624 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2625 = "tosa.transpose"(%2623, %2624) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %2626 = "tosa.identity"(%2625) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %2627 = "tosa.reshape"(%2626) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %2628 = "tosa.reshape"(%2627) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2629 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2630 = "tosa.transpose"(%arg285, %2629) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2631 = "tosa.reshape"(%2628) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2632 = "tosa.reshape"(%2630) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2633 = "tosa.matmul"(%2631, %2632) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2634 = "tosa.reshape"(%2633) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2635 = "tosa.reshape"(%arg286) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2636 = "tosa.add"(%2635, %2634) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2637 = "tosa.reshape"(%2636) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2638 = "tosa.add"(%2539, %2637) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2639 = "tosa.reduce_sum"(%2638) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2640 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2641 = "tosa.reciprocal"(%2640) : (tensor<1xf32>) -> tensor<1xf32>
    %2642 = "tosa.mul"(%2641, %2639) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2643 = "tosa.sub"(%2638, %2642) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2644 = "tosa.mul"(%2643, %2643) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2645 = "tosa.reduce_sum"(%2644) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2646 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2647 = "tosa.reciprocal"(%2646) : (tensor<1xf32>) -> tensor<1xf32>
    %2648 = "tosa.mul"(%2647, %2645) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2649 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2650 = "tosa.add"(%2648, %2649) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2651 = "tosa.rsqrt"(%2650) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2652 = "tosa.sub"(%2638, %2642) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2653 = "tosa.mul"(%2652, %2651) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2654 = "tosa.reshape"(%arg287) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2655 = "tosa.mul"(%2653, %2654) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2656 = "tosa.reshape"(%arg288) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2657 = "tosa.add"(%2655, %2656) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2658 = "tosa.reshape"(%2657) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2659 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2660 = "tosa.transpose"(%arg289, %2659) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %2661 = "tosa.reshape"(%2658) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2662 = "tosa.reshape"(%2660) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %2663 = "tosa.matmul"(%2661, %2662) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %2664 = "tosa.reshape"(%2663) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2665 = "tosa.reshape"(%arg290) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2666 = "tosa.add"(%2665, %2664) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %2667 = "tosa.reshape"(%2666) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2668 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2669 = "tosa.mul"(%2667, %2668) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2670 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2671 = "tosa.mul"(%2667, %2670) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2672 = "math.erf"(%2671) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2673 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2674 = "tosa.add"(%2672, %2673) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2675 = "tosa.mul"(%2669, %2674) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2676 = "tosa.reshape"(%2675) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2677 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2678 = "tosa.transpose"(%arg291, %2677) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %2679 = "tosa.reshape"(%2676) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2680 = "tosa.reshape"(%2678) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %2681 = "tosa.matmul"(%2679, %2680) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %2682 = "tosa.reshape"(%2681) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2683 = "tosa.reshape"(%arg292) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2684 = "tosa.add"(%2683, %2682) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2685 = "tosa.reshape"(%2684) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2686 = "tosa.add"(%2638, %2685) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2687 = "tosa.reduce_sum"(%2686) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2688 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2689 = "tosa.reciprocal"(%2688) : (tensor<1xf32>) -> tensor<1xf32>
    %2690 = "tosa.mul"(%2689, %2687) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2691 = "tosa.sub"(%2686, %2690) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2692 = "tosa.mul"(%2691, %2691) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2693 = "tosa.reduce_sum"(%2692) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2694 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2695 = "tosa.reciprocal"(%2694) : (tensor<1xf32>) -> tensor<1xf32>
    %2696 = "tosa.mul"(%2695, %2693) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2697 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2698 = "tosa.add"(%2696, %2697) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2699 = "tosa.rsqrt"(%2698) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2700 = "tosa.sub"(%2686, %2690) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2701 = "tosa.mul"(%2700, %2699) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2702 = "tosa.reshape"(%arg293) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2703 = "tosa.mul"(%2701, %2702) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2704 = "tosa.reshape"(%arg294) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2705 = "tosa.add"(%2703, %2704) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2706 = "tosa.reshape"(%2705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2707 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2708 = "tosa.transpose"(%arg295, %2707) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2709 = "tosa.reshape"(%2706) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2710 = "tosa.reshape"(%2708) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2711 = "tosa.matmul"(%2709, %2710) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2712 = "tosa.reshape"(%2711) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2713 = "tosa.reshape"(%arg296) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2714 = "tosa.add"(%2713, %2712) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2715 = "tosa.reshape"(%2714) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2716 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %2717 = "tosa.mul"(%2715, %2716) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2718 = "tosa.reshape"(%2705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2719 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2720 = "tosa.transpose"(%arg297, %2719) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2721 = "tosa.reshape"(%2718) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2722 = "tosa.reshape"(%2720) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2723 = "tosa.matmul"(%2721, %2722) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2724 = "tosa.reshape"(%2723) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2725 = "tosa.reshape"(%arg298) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2726 = "tosa.add"(%2725, %2724) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2727 = "tosa.reshape"(%2726) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2728 = "tosa.reshape"(%2727) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2729 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2730 = "tosa.transpose"(%2728, %2729) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2731 = "tosa.identity"(%2730) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2732 = "tosa.reshape"(%2705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2733 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2734 = "tosa.transpose"(%arg299, %2733) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2735 = "tosa.reshape"(%2732) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2736 = "tosa.reshape"(%2734) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2737 = "tosa.matmul"(%2735, %2736) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2738 = "tosa.reshape"(%2737) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2739 = "tosa.reshape"(%arg300) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2740 = "tosa.add"(%2739, %2738) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2741 = "tosa.reshape"(%2740) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2742 = "tosa.reshape"(%2741) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2743 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2744 = "tosa.transpose"(%2742, %2743) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2745 = "tosa.identity"(%2744) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2746 = "tosa.reshape"(%2717) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2747 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2748 = "tosa.transpose"(%2746, %2747) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2749 = "tosa.identity"(%2748) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2750 = "tosa.reshape"(%2749) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2751 = "tosa.reshape"(%2731) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2752 = "tosa.reshape"(%2745) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2753 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %2754 = "tosa.transpose"(%2751, %2753) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %2755 = "tosa.matmul"(%2750, %2754) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %2756 = "tosa.reshape"(%2755) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2757 = "tosa.add"(%2756, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2758 = "tosa.reshape"(%2757) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2759 = "tosa.reshape"(%2758) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2760 = "tosa.add"(%2759, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2761 = "tosa.reshape"(%2760) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2762 = "tosa.reduce_max"(%2761) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2763 = "tosa.sub"(%2761, %2762) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2764 = "tosa.exp"(%2763) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2765 = "tosa.reduce_sum"(%2764) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2766 = "tosa.reciprocal"(%2765) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %2767 = "tosa.mul"(%2764, %2766) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2768 = "tosa.identity"(%2767) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2769 = "tosa.matmul"(%2768, %2752) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %2770 = "tosa.reshape"(%2769) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2771 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2772 = "tosa.transpose"(%2770, %2771) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %2773 = "tosa.identity"(%2772) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %2774 = "tosa.reshape"(%2773) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %2775 = "tosa.reshape"(%2774) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2776 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2777 = "tosa.transpose"(%arg301, %2776) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2778 = "tosa.reshape"(%2775) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2779 = "tosa.reshape"(%2777) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2780 = "tosa.matmul"(%2778, %2779) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2781 = "tosa.reshape"(%2780) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2782 = "tosa.reshape"(%arg302) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2783 = "tosa.add"(%2782, %2781) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2784 = "tosa.reshape"(%2783) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2785 = "tosa.add"(%2686, %2784) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2786 = "tosa.reduce_sum"(%2785) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2787 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2788 = "tosa.reciprocal"(%2787) : (tensor<1xf32>) -> tensor<1xf32>
    %2789 = "tosa.mul"(%2788, %2786) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2790 = "tosa.sub"(%2785, %2789) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2791 = "tosa.mul"(%2790, %2790) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2792 = "tosa.reduce_sum"(%2791) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2793 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2794 = "tosa.reciprocal"(%2793) : (tensor<1xf32>) -> tensor<1xf32>
    %2795 = "tosa.mul"(%2794, %2792) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2796 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2797 = "tosa.add"(%2795, %2796) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2798 = "tosa.rsqrt"(%2797) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2799 = "tosa.sub"(%2785, %2789) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2800 = "tosa.mul"(%2799, %2798) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2801 = "tosa.reshape"(%arg303) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2802 = "tosa.mul"(%2800, %2801) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2803 = "tosa.reshape"(%arg304) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2804 = "tosa.add"(%2802, %2803) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2805 = "tosa.reshape"(%2804) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2806 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2807 = "tosa.transpose"(%arg305, %2806) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %2808 = "tosa.reshape"(%2805) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2809 = "tosa.reshape"(%2807) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %2810 = "tosa.matmul"(%2808, %2809) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %2811 = "tosa.reshape"(%2810) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2812 = "tosa.reshape"(%arg306) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2813 = "tosa.add"(%2812, %2811) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %2814 = "tosa.reshape"(%2813) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2815 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2816 = "tosa.mul"(%2814, %2815) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2817 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2818 = "tosa.mul"(%2814, %2817) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2819 = "math.erf"(%2818) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2820 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2821 = "tosa.add"(%2819, %2820) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2822 = "tosa.mul"(%2816, %2821) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2823 = "tosa.reshape"(%2822) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2824 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2825 = "tosa.transpose"(%arg307, %2824) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %2826 = "tosa.reshape"(%2823) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2827 = "tosa.reshape"(%2825) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %2828 = "tosa.matmul"(%2826, %2827) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %2829 = "tosa.reshape"(%2828) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2830 = "tosa.reshape"(%arg308) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2831 = "tosa.add"(%2830, %2829) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2832 = "tosa.reshape"(%2831) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2833 = "tosa.add"(%2785, %2832) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2834 = "tosa.reduce_sum"(%2833) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2835 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2836 = "tosa.reciprocal"(%2835) : (tensor<1xf32>) -> tensor<1xf32>
    %2837 = "tosa.mul"(%2836, %2834) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2838 = "tosa.sub"(%2833, %2837) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2839 = "tosa.mul"(%2838, %2838) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2840 = "tosa.reduce_sum"(%2839) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2841 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2842 = "tosa.reciprocal"(%2841) : (tensor<1xf32>) -> tensor<1xf32>
    %2843 = "tosa.mul"(%2842, %2840) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2844 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2845 = "tosa.add"(%2843, %2844) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2846 = "tosa.rsqrt"(%2845) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2847 = "tosa.sub"(%2833, %2837) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2848 = "tosa.mul"(%2847, %2846) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2849 = "tosa.reshape"(%arg309) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2850 = "tosa.mul"(%2848, %2849) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2851 = "tosa.reshape"(%arg310) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2852 = "tosa.add"(%2850, %2851) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2853 = "tosa.reshape"(%2852) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2854 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2855 = "tosa.transpose"(%arg311, %2854) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2856 = "tosa.reshape"(%2853) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2857 = "tosa.reshape"(%2855) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2858 = "tosa.matmul"(%2856, %2857) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2859 = "tosa.reshape"(%2858) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2860 = "tosa.reshape"(%arg312) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2861 = "tosa.add"(%2860, %2859) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2862 = "tosa.reshape"(%2861) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2863 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %2864 = "tosa.mul"(%2862, %2863) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2865 = "tosa.reshape"(%2852) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2866 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2867 = "tosa.transpose"(%arg313, %2866) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2868 = "tosa.reshape"(%2865) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2869 = "tosa.reshape"(%2867) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2870 = "tosa.matmul"(%2868, %2869) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2871 = "tosa.reshape"(%2870) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2872 = "tosa.reshape"(%arg314) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2873 = "tosa.add"(%2872, %2871) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2874 = "tosa.reshape"(%2873) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2875 = "tosa.reshape"(%2874) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2876 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2877 = "tosa.transpose"(%2875, %2876) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2878 = "tosa.identity"(%2877) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2879 = "tosa.reshape"(%2852) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2880 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2881 = "tosa.transpose"(%arg315, %2880) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2882 = "tosa.reshape"(%2879) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2883 = "tosa.reshape"(%2881) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2884 = "tosa.matmul"(%2882, %2883) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2885 = "tosa.reshape"(%2884) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2886 = "tosa.reshape"(%arg316) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2887 = "tosa.add"(%2886, %2885) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2888 = "tosa.reshape"(%2887) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2889 = "tosa.reshape"(%2888) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2890 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2891 = "tosa.transpose"(%2889, %2890) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2892 = "tosa.identity"(%2891) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2893 = "tosa.reshape"(%2864) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %2894 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2895 = "tosa.transpose"(%2893, %2894) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %2896 = "tosa.identity"(%2895) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2897 = "tosa.reshape"(%2896) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2898 = "tosa.reshape"(%2878) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2899 = "tosa.reshape"(%2892) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %2900 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %2901 = "tosa.transpose"(%2898, %2900) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %2902 = "tosa.matmul"(%2897, %2901) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %2903 = "tosa.reshape"(%2902) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2904 = "tosa.add"(%2903, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2905 = "tosa.reshape"(%2904) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2906 = "tosa.reshape"(%2905) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2907 = "tosa.add"(%2906, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %2908 = "tosa.reshape"(%2907) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %2909 = "tosa.reduce_max"(%2908) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2910 = "tosa.sub"(%2908, %2909) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2911 = "tosa.exp"(%2910) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2912 = "tosa.reduce_sum"(%2911) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %2913 = "tosa.reciprocal"(%2912) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %2914 = "tosa.mul"(%2911, %2913) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %2915 = "tosa.identity"(%2914) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %2916 = "tosa.matmul"(%2915, %2899) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %2917 = "tosa.reshape"(%2916) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %2918 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2919 = "tosa.transpose"(%2917, %2918) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %2920 = "tosa.identity"(%2919) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %2921 = "tosa.reshape"(%2920) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %2922 = "tosa.reshape"(%2921) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2923 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2924 = "tosa.transpose"(%arg317, %2923) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %2925 = "tosa.reshape"(%2922) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2926 = "tosa.reshape"(%2924) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %2927 = "tosa.matmul"(%2925, %2926) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %2928 = "tosa.reshape"(%2927) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2929 = "tosa.reshape"(%arg318) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2930 = "tosa.add"(%2929, %2928) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2931 = "tosa.reshape"(%2930) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2932 = "tosa.add"(%2833, %2931) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2933 = "tosa.reduce_sum"(%2932) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2934 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2935 = "tosa.reciprocal"(%2934) : (tensor<1xf32>) -> tensor<1xf32>
    %2936 = "tosa.mul"(%2935, %2933) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2937 = "tosa.sub"(%2932, %2936) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2938 = "tosa.mul"(%2937, %2937) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2939 = "tosa.reduce_sum"(%2938) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2940 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2941 = "tosa.reciprocal"(%2940) : (tensor<1xf32>) -> tensor<1xf32>
    %2942 = "tosa.mul"(%2941, %2939) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2943 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2944 = "tosa.add"(%2942, %2943) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2945 = "tosa.rsqrt"(%2944) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2946 = "tosa.sub"(%2932, %2936) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2947 = "tosa.mul"(%2946, %2945) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2948 = "tosa.reshape"(%arg319) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2949 = "tosa.mul"(%2947, %2948) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2950 = "tosa.reshape"(%arg320) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2951 = "tosa.add"(%2949, %2950) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2952 = "tosa.reshape"(%2951) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2953 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2954 = "tosa.transpose"(%arg321, %2953) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %2955 = "tosa.reshape"(%2952) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2956 = "tosa.reshape"(%2954) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %2957 = "tosa.matmul"(%2955, %2956) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %2958 = "tosa.reshape"(%2957) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2959 = "tosa.reshape"(%arg322) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2960 = "tosa.add"(%2959, %2958) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %2961 = "tosa.reshape"(%2960) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2962 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2963 = "tosa.mul"(%2961, %2962) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2964 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2965 = "tosa.mul"(%2961, %2964) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2966 = "math.erf"(%2965) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2967 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %2968 = "tosa.add"(%2966, %2967) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2969 = "tosa.mul"(%2963, %2968) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %2970 = "tosa.reshape"(%2969) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %2971 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2972 = "tosa.transpose"(%arg323, %2971) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %2973 = "tosa.reshape"(%2970) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %2974 = "tosa.reshape"(%2972) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %2975 = "tosa.matmul"(%2973, %2974) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %2976 = "tosa.reshape"(%2975) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2977 = "tosa.reshape"(%arg324) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %2978 = "tosa.add"(%2977, %2976) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %2979 = "tosa.reshape"(%2978) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %2980 = "tosa.add"(%2932, %2979) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2981 = "tosa.reduce_sum"(%2980) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2982 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2983 = "tosa.reciprocal"(%2982) : (tensor<1xf32>) -> tensor<1xf32>
    %2984 = "tosa.mul"(%2983, %2981) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2985 = "tosa.sub"(%2980, %2984) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2986 = "tosa.mul"(%2985, %2985) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %2987 = "tosa.reduce_sum"(%2986) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %2988 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2989 = "tosa.reciprocal"(%2988) : (tensor<1xf32>) -> tensor<1xf32>
    %2990 = "tosa.mul"(%2989, %2987) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2991 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %2992 = "tosa.add"(%2990, %2991) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2993 = "tosa.rsqrt"(%2992) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %2994 = "tosa.sub"(%2980, %2984) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2995 = "tosa.mul"(%2994, %2993) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %2996 = "tosa.reshape"(%arg325) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2997 = "tosa.mul"(%2995, %2996) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %2998 = "tosa.reshape"(%arg326) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %2999 = "tosa.add"(%2997, %2998) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3000 = "tosa.reshape"(%2999) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3001 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3002 = "tosa.transpose"(%arg327, %3001) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3003 = "tosa.reshape"(%3000) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3004 = "tosa.reshape"(%3002) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3005 = "tosa.matmul"(%3003, %3004) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3006 = "tosa.reshape"(%3005) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3007 = "tosa.reshape"(%arg328) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3008 = "tosa.add"(%3007, %3006) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3009 = "tosa.reshape"(%3008) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3010 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %3011 = "tosa.mul"(%3009, %3010) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3012 = "tosa.reshape"(%2999) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3013 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3014 = "tosa.transpose"(%arg329, %3013) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3015 = "tosa.reshape"(%3012) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3016 = "tosa.reshape"(%3014) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3017 = "tosa.matmul"(%3015, %3016) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3018 = "tosa.reshape"(%3017) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3019 = "tosa.reshape"(%arg330) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3020 = "tosa.add"(%3019, %3018) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3021 = "tosa.reshape"(%3020) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3022 = "tosa.reshape"(%3021) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %3023 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3024 = "tosa.transpose"(%3022, %3023) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %3025 = "tosa.identity"(%3024) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3026 = "tosa.reshape"(%2999) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3027 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3028 = "tosa.transpose"(%arg331, %3027) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3029 = "tosa.reshape"(%3026) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3030 = "tosa.reshape"(%3028) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3031 = "tosa.matmul"(%3029, %3030) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3032 = "tosa.reshape"(%3031) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3033 = "tosa.reshape"(%arg332) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3034 = "tosa.add"(%3033, %3032) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3035 = "tosa.reshape"(%3034) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3036 = "tosa.reshape"(%3035) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %3037 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3038 = "tosa.transpose"(%3036, %3037) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %3039 = "tosa.identity"(%3038) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3040 = "tosa.reshape"(%3011) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %3041 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3042 = "tosa.transpose"(%3040, %3041) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %3043 = "tosa.identity"(%3042) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3044 = "tosa.reshape"(%3043) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %3045 = "tosa.reshape"(%3025) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %3046 = "tosa.reshape"(%3039) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %3047 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %3048 = "tosa.transpose"(%3045, %3047) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %3049 = "tosa.matmul"(%3044, %3048) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %3050 = "tosa.reshape"(%3049) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3051 = "tosa.add"(%3050, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3052 = "tosa.reshape"(%3051) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %3053 = "tosa.reshape"(%3052) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3054 = "tosa.add"(%3053, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3055 = "tosa.reshape"(%3054) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %3056 = "tosa.reduce_max"(%3055) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %3057 = "tosa.sub"(%3055, %3056) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %3058 = "tosa.exp"(%3057) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %3059 = "tosa.reduce_sum"(%3058) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %3060 = "tosa.reciprocal"(%3059) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %3061 = "tosa.mul"(%3058, %3060) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %3062 = "tosa.identity"(%3061) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %3063 = "tosa.matmul"(%3062, %3046) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %3064 = "tosa.reshape"(%3063) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3065 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3066 = "tosa.transpose"(%3064, %3065) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %3067 = "tosa.identity"(%3066) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %3068 = "tosa.reshape"(%3067) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %3069 = "tosa.reshape"(%3068) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3070 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3071 = "tosa.transpose"(%arg333, %3070) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3072 = "tosa.reshape"(%3069) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3073 = "tosa.reshape"(%3071) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3074 = "tosa.matmul"(%3072, %3073) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3075 = "tosa.reshape"(%3074) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3076 = "tosa.reshape"(%arg334) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3077 = "tosa.add"(%3076, %3075) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3078 = "tosa.reshape"(%3077) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3079 = "tosa.add"(%2980, %3078) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3080 = "tosa.reduce_sum"(%3079) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3081 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3082 = "tosa.reciprocal"(%3081) : (tensor<1xf32>) -> tensor<1xf32>
    %3083 = "tosa.mul"(%3082, %3080) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3084 = "tosa.sub"(%3079, %3083) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3085 = "tosa.mul"(%3084, %3084) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3086 = "tosa.reduce_sum"(%3085) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3087 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3088 = "tosa.reciprocal"(%3087) : (tensor<1xf32>) -> tensor<1xf32>
    %3089 = "tosa.mul"(%3088, %3086) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3090 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %3091 = "tosa.add"(%3089, %3090) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3092 = "tosa.rsqrt"(%3091) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3093 = "tosa.sub"(%3079, %3083) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3094 = "tosa.mul"(%3093, %3092) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3095 = "tosa.reshape"(%arg335) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3096 = "tosa.mul"(%3094, %3095) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3097 = "tosa.reshape"(%arg336) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3098 = "tosa.add"(%3096, %3097) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3099 = "tosa.reshape"(%3098) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3100 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3101 = "tosa.transpose"(%arg337, %3100) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %3102 = "tosa.reshape"(%3099) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3103 = "tosa.reshape"(%3101) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %3104 = "tosa.matmul"(%3102, %3103) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %3105 = "tosa.reshape"(%3104) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %3106 = "tosa.reshape"(%arg338) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %3107 = "tosa.add"(%3106, %3105) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %3108 = "tosa.reshape"(%3107) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %3109 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %3110 = "tosa.mul"(%3108, %3109) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3111 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %3112 = "tosa.mul"(%3108, %3111) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3113 = "math.erf"(%3112) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3114 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %3115 = "tosa.add"(%3113, %3114) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3116 = "tosa.mul"(%3110, %3115) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3117 = "tosa.reshape"(%3116) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %3118 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3119 = "tosa.transpose"(%arg339, %3118) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %3120 = "tosa.reshape"(%3117) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %3121 = "tosa.reshape"(%3119) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %3122 = "tosa.matmul"(%3120, %3121) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %3123 = "tosa.reshape"(%3122) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3124 = "tosa.reshape"(%arg340) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3125 = "tosa.add"(%3124, %3123) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3126 = "tosa.reshape"(%3125) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3127 = "tosa.add"(%3079, %3126) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3128 = "tosa.reduce_sum"(%3127) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3129 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3130 = "tosa.reciprocal"(%3129) : (tensor<1xf32>) -> tensor<1xf32>
    %3131 = "tosa.mul"(%3130, %3128) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3132 = "tosa.sub"(%3127, %3131) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3133 = "tosa.mul"(%3132, %3132) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3134 = "tosa.reduce_sum"(%3133) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3135 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3136 = "tosa.reciprocal"(%3135) : (tensor<1xf32>) -> tensor<1xf32>
    %3137 = "tosa.mul"(%3136, %3134) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3138 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %3139 = "tosa.add"(%3137, %3138) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3140 = "tosa.rsqrt"(%3139) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3141 = "tosa.sub"(%3127, %3131) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3142 = "tosa.mul"(%3141, %3140) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3143 = "tosa.reshape"(%arg341) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3144 = "tosa.mul"(%3142, %3143) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3145 = "tosa.reshape"(%arg342) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3146 = "tosa.add"(%3144, %3145) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3147 = "tosa.reshape"(%3146) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3148 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3149 = "tosa.transpose"(%arg343, %3148) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3150 = "tosa.reshape"(%3147) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3151 = "tosa.reshape"(%3149) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3152 = "tosa.matmul"(%3150, %3151) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3153 = "tosa.reshape"(%3152) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3154 = "tosa.reshape"(%arg344) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3155 = "tosa.add"(%3154, %3153) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3156 = "tosa.reshape"(%3155) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3157 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %3158 = "tosa.mul"(%3156, %3157) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3159 = "tosa.reshape"(%3146) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3160 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3161 = "tosa.transpose"(%arg345, %3160) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3162 = "tosa.reshape"(%3159) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3163 = "tosa.reshape"(%3161) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3164 = "tosa.matmul"(%3162, %3163) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3165 = "tosa.reshape"(%3164) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3166 = "tosa.reshape"(%arg346) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3167 = "tosa.add"(%3166, %3165) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3168 = "tosa.reshape"(%3167) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3169 = "tosa.reshape"(%3168) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %3170 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3171 = "tosa.transpose"(%3169, %3170) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %3172 = "tosa.identity"(%3171) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3173 = "tosa.reshape"(%3146) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3174 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3175 = "tosa.transpose"(%arg347, %3174) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3176 = "tosa.reshape"(%3173) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3177 = "tosa.reshape"(%3175) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3178 = "tosa.matmul"(%3176, %3177) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3179 = "tosa.reshape"(%3178) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3180 = "tosa.reshape"(%arg348) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3181 = "tosa.add"(%3180, %3179) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3182 = "tosa.reshape"(%3181) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3183 = "tosa.reshape"(%3182) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %3184 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3185 = "tosa.transpose"(%3183, %3184) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %3186 = "tosa.identity"(%3185) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3187 = "tosa.reshape"(%3158) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %3188 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3189 = "tosa.transpose"(%3187, %3188) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %3190 = "tosa.identity"(%3189) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3191 = "tosa.reshape"(%3190) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %3192 = "tosa.reshape"(%3172) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %3193 = "tosa.reshape"(%3186) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %3194 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %3195 = "tosa.transpose"(%3192, %3194) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %3196 = "tosa.matmul"(%3191, %3195) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %3197 = "tosa.reshape"(%3196) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3198 = "tosa.add"(%3197, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3199 = "tosa.reshape"(%3198) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %3200 = "tosa.reshape"(%3199) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3201 = "tosa.add"(%3200, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3202 = "tosa.reshape"(%3201) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %3203 = "tosa.reduce_max"(%3202) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %3204 = "tosa.sub"(%3202, %3203) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %3205 = "tosa.exp"(%3204) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %3206 = "tosa.reduce_sum"(%3205) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %3207 = "tosa.reciprocal"(%3206) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %3208 = "tosa.mul"(%3205, %3207) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %3209 = "tosa.identity"(%3208) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %3210 = "tosa.matmul"(%3209, %3193) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %3211 = "tosa.reshape"(%3210) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3212 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3213 = "tosa.transpose"(%3211, %3212) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %3214 = "tosa.identity"(%3213) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %3215 = "tosa.reshape"(%3214) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %3216 = "tosa.reshape"(%3215) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3217 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3218 = "tosa.transpose"(%arg349, %3217) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3219 = "tosa.reshape"(%3216) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3220 = "tosa.reshape"(%3218) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3221 = "tosa.matmul"(%3219, %3220) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3222 = "tosa.reshape"(%3221) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3223 = "tosa.reshape"(%arg350) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3224 = "tosa.add"(%3223, %3222) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3225 = "tosa.reshape"(%3224) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3226 = "tosa.add"(%3127, %3225) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3227 = "tosa.reduce_sum"(%3226) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3228 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3229 = "tosa.reciprocal"(%3228) : (tensor<1xf32>) -> tensor<1xf32>
    %3230 = "tosa.mul"(%3229, %3227) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3231 = "tosa.sub"(%3226, %3230) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3232 = "tosa.mul"(%3231, %3231) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3233 = "tosa.reduce_sum"(%3232) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3234 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3235 = "tosa.reciprocal"(%3234) : (tensor<1xf32>) -> tensor<1xf32>
    %3236 = "tosa.mul"(%3235, %3233) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3237 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %3238 = "tosa.add"(%3236, %3237) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3239 = "tosa.rsqrt"(%3238) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3240 = "tosa.sub"(%3226, %3230) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3241 = "tosa.mul"(%3240, %3239) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3242 = "tosa.reshape"(%arg351) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3243 = "tosa.mul"(%3241, %3242) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3244 = "tosa.reshape"(%arg352) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3245 = "tosa.add"(%3243, %3244) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3246 = "tosa.reshape"(%3245) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3247 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3248 = "tosa.transpose"(%arg353, %3247) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %3249 = "tosa.reshape"(%3246) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3250 = "tosa.reshape"(%3248) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %3251 = "tosa.matmul"(%3249, %3250) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %3252 = "tosa.reshape"(%3251) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %3253 = "tosa.reshape"(%arg354) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %3254 = "tosa.add"(%3253, %3252) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %3255 = "tosa.reshape"(%3254) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %3256 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %3257 = "tosa.mul"(%3255, %3256) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3258 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %3259 = "tosa.mul"(%3255, %3258) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3260 = "math.erf"(%3259) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3261 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %3262 = "tosa.add"(%3260, %3261) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3263 = "tosa.mul"(%3257, %3262) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3264 = "tosa.reshape"(%3263) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %3265 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3266 = "tosa.transpose"(%arg355, %3265) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %3267 = "tosa.reshape"(%3264) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %3268 = "tosa.reshape"(%3266) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %3269 = "tosa.matmul"(%3267, %3268) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %3270 = "tosa.reshape"(%3269) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3271 = "tosa.reshape"(%arg356) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3272 = "tosa.add"(%3271, %3270) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3273 = "tosa.reshape"(%3272) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3274 = "tosa.add"(%3226, %3273) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3275 = "tosa.reduce_sum"(%3274) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3276 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3277 = "tosa.reciprocal"(%3276) : (tensor<1xf32>) -> tensor<1xf32>
    %3278 = "tosa.mul"(%3277, %3275) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3279 = "tosa.sub"(%3274, %3278) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3280 = "tosa.mul"(%3279, %3279) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3281 = "tosa.reduce_sum"(%3280) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3282 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3283 = "tosa.reciprocal"(%3282) : (tensor<1xf32>) -> tensor<1xf32>
    %3284 = "tosa.mul"(%3283, %3281) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3285 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %3286 = "tosa.add"(%3284, %3285) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3287 = "tosa.rsqrt"(%3286) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3288 = "tosa.sub"(%3274, %3278) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3289 = "tosa.mul"(%3288, %3287) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3290 = "tosa.reshape"(%arg357) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3291 = "tosa.mul"(%3289, %3290) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3292 = "tosa.reshape"(%arg358) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3293 = "tosa.add"(%3291, %3292) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3294 = "tosa.reshape"(%3293) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3295 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3296 = "tosa.transpose"(%arg359, %3295) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3297 = "tosa.reshape"(%3294) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3298 = "tosa.reshape"(%3296) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3299 = "tosa.matmul"(%3297, %3298) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3300 = "tosa.reshape"(%3299) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3301 = "tosa.reshape"(%arg360) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3302 = "tosa.add"(%3301, %3300) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3303 = "tosa.reshape"(%3302) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3304 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x1024xf32>}> : () -> tensor<1x12x1024xf32>
    %3305 = "tosa.mul"(%3303, %3304) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3306 = "tosa.reshape"(%3293) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3307 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3308 = "tosa.transpose"(%arg361, %3307) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3309 = "tosa.reshape"(%3306) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3310 = "tosa.reshape"(%3308) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3311 = "tosa.matmul"(%3309, %3310) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3312 = "tosa.reshape"(%3311) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3313 = "tosa.reshape"(%arg362) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3314 = "tosa.add"(%3313, %3312) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3315 = "tosa.reshape"(%3314) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3316 = "tosa.reshape"(%3315) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %3317 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3318 = "tosa.transpose"(%3316, %3317) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %3319 = "tosa.identity"(%3318) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3320 = "tosa.reshape"(%3293) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3321 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3322 = "tosa.transpose"(%arg363, %3321) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3323 = "tosa.reshape"(%3320) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3324 = "tosa.reshape"(%3322) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3325 = "tosa.matmul"(%3323, %3324) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3326 = "tosa.reshape"(%3325) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3327 = "tosa.reshape"(%arg364) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3328 = "tosa.add"(%3327, %3326) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3329 = "tosa.reshape"(%3328) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3330 = "tosa.reshape"(%3329) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %3331 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3332 = "tosa.transpose"(%3330, %3331) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %3333 = "tosa.identity"(%3332) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3334 = "tosa.reshape"(%3305) <{new_shape = array<i64: 1, 12, 16, 64>}> : (tensor<1x12x1024xf32>) -> tensor<1x12x16x64xf32>
    %3335 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3336 = "tosa.transpose"(%3334, %3335) : (tensor<1x12x16x64xf32>, tensor<4xi32>) -> tensor<1x16x12x64xf32>
    %3337 = "tosa.identity"(%3336) : (tensor<1x16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3338 = "tosa.reshape"(%3337) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %3339 = "tosa.reshape"(%3319) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %3340 = "tosa.reshape"(%3333) <{new_shape = array<i64: 16, 12, 64>}> : (tensor<1x16x12x64xf32>) -> tensor<16x12x64xf32>
    %3341 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %3342 = "tosa.transpose"(%3339, %3341) : (tensor<16x12x64xf32>, tensor<3xi32>) -> tensor<16x64x12xf32>
    %3343 = "tosa.matmul"(%3338, %3342) : (tensor<16x12x64xf32>, tensor<16x64x12xf32>) -> tensor<16x12x12xf32>
    %3344 = "tosa.reshape"(%3343) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3345 = "tosa.add"(%3344, %110) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3346 = "tosa.reshape"(%3345) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %3347 = "tosa.reshape"(%3346) <{new_shape = array<i64: 1, 16, 12, 12>}> : (tensor<16x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3348 = "tosa.add"(%3347, %34) : (tensor<1x16x12x12xf32>, tensor<1x1x12x12xf32>) -> tensor<1x16x12x12xf32>
    %3349 = "tosa.reshape"(%3348) <{new_shape = array<i64: 16, 12, 12>}> : (tensor<1x16x12x12xf32>) -> tensor<16x12x12xf32>
    %3350 = "tosa.reduce_max"(%3349) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %3351 = "tosa.sub"(%3349, %3350) : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %3352 = "tosa.exp"(%3351) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %3353 = "tosa.reduce_sum"(%3352) <{axis = 2 : i32}> : (tensor<16x12x12xf32>) -> tensor<16x12x1xf32>
    %3354 = "tosa.reciprocal"(%3353) : (tensor<16x12x1xf32>) -> tensor<16x12x1xf32>
    %3355 = "tosa.mul"(%3352, %3354) <{shift = 0 : i8}> : (tensor<16x12x12xf32>, tensor<16x12x1xf32>) -> tensor<16x12x12xf32>
    %3356 = "tosa.identity"(%3355) : (tensor<16x12x12xf32>) -> tensor<16x12x12xf32>
    %3357 = "tosa.matmul"(%3356, %3340) : (tensor<16x12x12xf32>, tensor<16x12x64xf32>) -> tensor<16x12x64xf32>
    %3358 = "tosa.reshape"(%3357) <{new_shape = array<i64: 1, 16, 12, 64>}> : (tensor<16x12x64xf32>) -> tensor<1x16x12x64xf32>
    %3359 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3360 = "tosa.transpose"(%3358, %3359) : (tensor<1x16x12x64xf32>, tensor<4xi32>) -> tensor<1x12x16x64xf32>
    %3361 = "tosa.identity"(%3360) : (tensor<1x12x16x64xf32>) -> tensor<1x12x16x64xf32>
    %3362 = "tosa.reshape"(%3361) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<1x12x16x64xf32>) -> tensor<1x12x1024xf32>
    %3363 = "tosa.reshape"(%3362) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3364 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3365 = "tosa.transpose"(%arg365, %3364) : (tensor<1024x1024xf32>, tensor<2xi32>) -> tensor<1024x1024xf32>
    %3366 = "tosa.reshape"(%3363) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3367 = "tosa.reshape"(%3365) <{new_shape = array<i64: 1, 1024, 1024>}> : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %3368 = "tosa.matmul"(%3366, %3367) : (tensor<1x12x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x12x1024xf32>
    %3369 = "tosa.reshape"(%3368) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3370 = "tosa.reshape"(%arg366) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3371 = "tosa.add"(%3370, %3369) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3372 = "tosa.reshape"(%3371) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3373 = "tosa.add"(%3274, %3372) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3374 = "tosa.reduce_sum"(%3373) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3375 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3376 = "tosa.reciprocal"(%3375) : (tensor<1xf32>) -> tensor<1xf32>
    %3377 = "tosa.mul"(%3376, %3374) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3378 = "tosa.sub"(%3373, %3377) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3379 = "tosa.mul"(%3378, %3378) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3380 = "tosa.reduce_sum"(%3379) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3381 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3382 = "tosa.reciprocal"(%3381) : (tensor<1xf32>) -> tensor<1xf32>
    %3383 = "tosa.mul"(%3382, %3380) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3384 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %3385 = "tosa.add"(%3383, %3384) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3386 = "tosa.rsqrt"(%3385) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3387 = "tosa.sub"(%3373, %3377) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3388 = "tosa.mul"(%3387, %3386) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3389 = "tosa.reshape"(%arg367) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3390 = "tosa.mul"(%3388, %3389) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3391 = "tosa.reshape"(%arg368) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3392 = "tosa.add"(%3390, %3391) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3393 = "tosa.reshape"(%3392) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3394 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3395 = "tosa.transpose"(%arg369, %3394) : (tensor<4096x1024xf32>, tensor<2xi32>) -> tensor<1024x4096xf32>
    %3396 = "tosa.reshape"(%3393) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3397 = "tosa.reshape"(%3395) <{new_shape = array<i64: 1, 1024, 4096>}> : (tensor<1024x4096xf32>) -> tensor<1x1024x4096xf32>
    %3398 = "tosa.matmul"(%3396, %3397) : (tensor<1x12x1024xf32>, tensor<1x1024x4096xf32>) -> tensor<1x12x4096xf32>
    %3399 = "tosa.reshape"(%3398) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %3400 = "tosa.reshape"(%arg370) <{new_shape = array<i64: 1, 4096>}> : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %3401 = "tosa.add"(%3400, %3399) : (tensor<1x4096xf32>, tensor<12x4096xf32>) -> tensor<12x4096xf32>
    %3402 = "tosa.reshape"(%3401) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %3403 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %3404 = "tosa.mul"(%3402, %3403) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3405 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %3406 = "tosa.mul"(%3402, %3405) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3407 = "math.erf"(%3406) <{fastmath = #arith.fastmath<none>}> : (tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3408 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x12x4096xf32>}> : () -> tensor<1x12x4096xf32>
    %3409 = "tosa.add"(%3407, %3408) : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3410 = "tosa.mul"(%3404, %3409) <{shift = 0 : i8}> : (tensor<1x12x4096xf32>, tensor<1x12x4096xf32>) -> tensor<1x12x4096xf32>
    %3411 = "tosa.reshape"(%3410) <{new_shape = array<i64: 12, 4096>}> : (tensor<1x12x4096xf32>) -> tensor<12x4096xf32>
    %3412 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3413 = "tosa.transpose"(%arg371, %3412) : (tensor<1024x4096xf32>, tensor<2xi32>) -> tensor<4096x1024xf32>
    %3414 = "tosa.reshape"(%3411) <{new_shape = array<i64: 1, 12, 4096>}> : (tensor<12x4096xf32>) -> tensor<1x12x4096xf32>
    %3415 = "tosa.reshape"(%3413) <{new_shape = array<i64: 1, 4096, 1024>}> : (tensor<4096x1024xf32>) -> tensor<1x4096x1024xf32>
    %3416 = "tosa.matmul"(%3414, %3415) : (tensor<1x12x4096xf32>, tensor<1x4096x1024xf32>) -> tensor<1x12x1024xf32>
    %3417 = "tosa.reshape"(%3416) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3418 = "tosa.reshape"(%arg372) <{new_shape = array<i64: 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3419 = "tosa.add"(%3418, %3417) : (tensor<1x1024xf32>, tensor<12x1024xf32>) -> tensor<12x1024xf32>
    %3420 = "tosa.reshape"(%3419) <{new_shape = array<i64: 1, 12, 1024>}> : (tensor<12x1024xf32>) -> tensor<1x12x1024xf32>
    %3421 = "tosa.add"(%3373, %3420) : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3422 = "tosa.reduce_sum"(%3421) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3423 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3424 = "tosa.reciprocal"(%3423) : (tensor<1xf32>) -> tensor<1xf32>
    %3425 = "tosa.mul"(%3424, %3422) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3426 = "tosa.sub"(%3421, %3425) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3427 = "tosa.mul"(%3426, %3426) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1024xf32>) -> tensor<1x12x1024xf32>
    %3428 = "tosa.reduce_sum"(%3427) <{axis = 2 : i32}> : (tensor<1x12x1024xf32>) -> tensor<1x12x1xf32>
    %3429 = "tosa.const"() <{value = dense<1.024000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3430 = "tosa.reciprocal"(%3429) : (tensor<1xf32>) -> tensor<1xf32>
    %3431 = "tosa.mul"(%3430, %3428) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3432 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x12x1xf32>}> : () -> tensor<1x12x1xf32>
    %3433 = "tosa.add"(%3431, %3432) : (tensor<1x12x1xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3434 = "tosa.rsqrt"(%3433) : (tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %3435 = "tosa.sub"(%3421, %3425) : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3436 = "tosa.mul"(%3435, %3434) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x12x1xf32>) -> tensor<1x12x1024xf32>
    %3437 = "tosa.reshape"(%arg373) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3438 = "tosa.mul"(%3436, %3437) <{shift = 0 : i8}> : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3439 = "tosa.reshape"(%arg374) <{new_shape = array<i64: 1, 1, 1024>}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %3440 = "tosa.add"(%3438, %3439) : (tensor<1x12x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x12x1024xf32>
    %3441 = "tosa.const"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %3442 = "tosa.cast"(%0) : (tensor<1x12xi64>) -> tensor<1x12xi32>
    %3443 = "tosa.argmax"(%3442) <{axis = 1 : i32}> : (tensor<1x12xi32>) -> tensor<1xi32>
    %3444 = "tensor.empty"() : () -> tensor<1x1024xf32>
    %3445 = "linalg.generic"(%3441, %3443, %3444) <{indexing_maps = [#map5, #map5, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg375: i64, %arg376: i32, %arg377: f32):
      %3446 = "arith.index_cast"(%arg375) : (i64) -> index
      %3447 = "arith.index_cast"(%arg376) : (i32) -> index
      %3448 = "tensor.extract"(%3440, %3446, %3447) : (tensor<1x12x1024xf32>, index, index) -> f32
      "linalg.yield"(%3448) : (f32) -> ()
    }) : (tensor<1xi64>, tensor<1xi32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    "func.return"(%3440, %3445) : (tensor<1x12x1024xf32>, tensor<1x1024xf32>) -> ()
  }) : () -> ()
}) : () -> ()

