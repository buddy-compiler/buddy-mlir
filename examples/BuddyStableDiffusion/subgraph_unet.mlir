#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1xf32>, tensor<1280x320xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1x4x64x64xf32>, tensor<320x4x3x3xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320x3x3xf32>, tensor<320xf32>, tensor<320x1280xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320x3x3xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320x1024xf32>, tensor<1x12x1024xf32>, tensor<320x1024xf32>, tensor<1x12x1024xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<2560x320xf32>, tensor<2560xf32>, tensor<320x1280xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320x3x3xf32>, tensor<320xf32>, tensor<320x1280xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320x3x3xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320x1024xf32>, tensor<1x12x1024xf32>, tensor<320x1024xf32>, tensor<1x12x1024xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<2560x320xf32>, tensor<2560xf32>, tensor<320x1280xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320x320x3x3xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<640x320x3x3xf32>, tensor<640xf32>, tensor<640x1280xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640x3x3xf32>, tensor<640xf32>, tensor<640x320x1x1xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640x1024xf32>, tensor<1x12x1024xf32>, tensor<640x1024xf32>, tensor<1x12x1024xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<5120x640xf32>, tensor<5120xf32>, tensor<640x2560xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640x3x3xf32>, tensor<640xf32>, tensor<640x1280xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640x3x3xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640x1024xf32>, tensor<1x12x1024xf32>, tensor<640x1024xf32>, tensor<1x12x1024xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<5120x640xf32>, tensor<5120xf32>, tensor<640x2560xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640x640x3x3xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<1280x640x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x640x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<10240x1280xf32>, tensor<10240xf32>, tensor<1280x5120xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<10240x1280xf32>, tensor<10240xf32>, tensor<1280x5120xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<10240x1280xf32>, tensor<10240xf32>, tensor<1280x5120xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<2560xf32>, tensor<2560xf32>, tensor<1280x2560x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x2560x1x1xf32>, tensor<1280xf32>, tensor<2560xf32>, tensor<2560xf32>, tensor<1280x2560x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x2560x1x1xf32>, tensor<1280xf32>, tensor<2560xf32>, tensor<2560xf32>, tensor<1280x2560x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x2560x1x1xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<2560xf32>, tensor<2560xf32>, tensor<1280x2560x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x2560x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<10240x1280xf32>, tensor<10240xf32>, tensor<1280x5120xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<2560xf32>, tensor<2560xf32>, tensor<1280x2560x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x2560x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<10240x1280xf32>, tensor<10240xf32>, tensor<1280x5120xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1920xf32>, tensor<1920xf32>, tensor<1280x1920x3x3xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1280x1920x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1024xf32>, tensor<1x12x1024xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<10240x1280xf32>, tensor<10240xf32>, tensor<1280x5120xf32>, tensor<1280xf32>, tensor<1280x1280xf32>, tensor<1280xf32>, tensor<1280x1280x3x3xf32>, tensor<1280xf32>, tensor<1920xf32>, tensor<1920xf32>, tensor<640x1920x3x3xf32>, tensor<640xf32>, tensor<640x1280xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640x3x3xf32>, tensor<640xf32>, tensor<640x1920x1x1xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640x1024xf32>, tensor<1x12x1024xf32>, tensor<640x1024xf32>, tensor<1x12x1024xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<5120x640xf32>, tensor<5120xf32>, tensor<640x2560xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<640x1280x3x3xf32>, tensor<640xf32>, tensor<640x1280xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640x3x3xf32>, tensor<640xf32>, tensor<640x1280x1x1xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640x1024xf32>, tensor<1x12x1024xf32>, tensor<640x1024xf32>, tensor<1x12x1024xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<5120x640xf32>, tensor<5120xf32>, tensor<640x2560xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<960xf32>, tensor<960xf32>, tensor<640x960x3x3xf32>, tensor<640xf32>, tensor<640x1280xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640x3x3xf32>, tensor<640xf32>, tensor<640x960x1x1xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640x1024xf32>, tensor<1x12x1024xf32>, tensor<640x1024xf32>, tensor<1x12x1024xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<5120x640xf32>, tensor<5120xf32>, tensor<640x2560xf32>, tensor<640xf32>, tensor<640x640xf32>, tensor<640xf32>, tensor<640x640x3x3xf32>, tensor<640xf32>, tensor<960xf32>, tensor<960xf32>, tensor<320x960x3x3xf32>, tensor<320xf32>, tensor<320x1280xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320x3x3xf32>, tensor<320xf32>, tensor<320x960x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320x1024xf32>, tensor<1x12x1024xf32>, tensor<320x1024xf32>, tensor<1x12x1024xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<2560x320xf32>, tensor<2560xf32>, tensor<320x1280xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<640xf32>, tensor<640xf32>, tensor<320x640x3x3xf32>, tensor<320xf32>, tensor<320x1280xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320x3x3xf32>, tensor<320xf32>, tensor<320x640x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320x1024xf32>, tensor<1x12x1024xf32>, tensor<320x1024xf32>, tensor<1x12x1024xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<2560x320xf32>, tensor<2560xf32>, tensor<320x1280xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<640xf32>, tensor<640xf32>, tensor<320x640x3x3xf32>, tensor<320xf32>, tensor<320x1280xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320x3x3xf32>, tensor<320xf32>, tensor<320x640x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320x1024xf32>, tensor<1x12x1024xf32>, tensor<320x1024xf32>, tensor<1x12x1024xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<2560x320xf32>, tensor<2560xf32>, tensor<320x1280xf32>, tensor<320xf32>, tensor<320x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<4x320x3x3xf32>, tensor<4xf32>) -> tensor<1x4x64x64xf32>, sym_name = "subgraph0"}> ({
  ^bb0(%arg0: tensor<1xf32>, %arg1: tensor<1280x320xf32>, %arg2: tensor<1280xf32>, %arg3: tensor<1280x1280xf32>, %arg4: tensor<1280xf32>, %arg5: tensor<1x4x64x64xf32>, %arg6: tensor<320x4x3x3xf32>, %arg7: tensor<320xf32>, %arg8: tensor<320xf32>, %arg9: tensor<320xf32>, %arg10: tensor<320x320x3x3xf32>, %arg11: tensor<320xf32>, %arg12: tensor<320x1280xf32>, %arg13: tensor<320xf32>, %arg14: tensor<320xf32>, %arg15: tensor<320xf32>, %arg16: tensor<320x320x3x3xf32>, %arg17: tensor<320xf32>, %arg18: tensor<320xf32>, %arg19: tensor<320xf32>, %arg20: tensor<320x320xf32>, %arg21: tensor<320xf32>, %arg22: tensor<320xf32>, %arg23: tensor<320xf32>, %arg24: tensor<320x320xf32>, %arg25: tensor<320x320xf32>, %arg26: tensor<320x320xf32>, %arg27: tensor<320x320xf32>, %arg28: tensor<320xf32>, %arg29: tensor<320xf32>, %arg30: tensor<320xf32>, %arg31: tensor<320x320xf32>, %arg32: tensor<320x1024xf32>, %arg33: tensor<1x12x1024xf32>, %arg34: tensor<320x1024xf32>, %arg35: tensor<1x12x1024xf32>, %arg36: tensor<320x320xf32>, %arg37: tensor<320xf32>, %arg38: tensor<320xf32>, %arg39: tensor<320xf32>, %arg40: tensor<2560x320xf32>, %arg41: tensor<2560xf32>, %arg42: tensor<320x1280xf32>, %arg43: tensor<320xf32>, %arg44: tensor<320x320xf32>, %arg45: tensor<320xf32>, %arg46: tensor<320xf32>, %arg47: tensor<320xf32>, %arg48: tensor<320x320x3x3xf32>, %arg49: tensor<320xf32>, %arg50: tensor<320x1280xf32>, %arg51: tensor<320xf32>, %arg52: tensor<320xf32>, %arg53: tensor<320xf32>, %arg54: tensor<320x320x3x3xf32>, %arg55: tensor<320xf32>, %arg56: tensor<320xf32>, %arg57: tensor<320xf32>, %arg58: tensor<320x320xf32>, %arg59: tensor<320xf32>, %arg60: tensor<320xf32>, %arg61: tensor<320xf32>, %arg62: tensor<320x320xf32>, %arg63: tensor<320x320xf32>, %arg64: tensor<320x320xf32>, %arg65: tensor<320x320xf32>, %arg66: tensor<320xf32>, %arg67: tensor<320xf32>, %arg68: tensor<320xf32>, %arg69: tensor<320x320xf32>, %arg70: tensor<320x1024xf32>, %arg71: tensor<1x12x1024xf32>, %arg72: tensor<320x1024xf32>, %arg73: tensor<1x12x1024xf32>, %arg74: tensor<320x320xf32>, %arg75: tensor<320xf32>, %arg76: tensor<320xf32>, %arg77: tensor<320xf32>, %arg78: tensor<2560x320xf32>, %arg79: tensor<2560xf32>, %arg80: tensor<320x1280xf32>, %arg81: tensor<320xf32>, %arg82: tensor<320x320xf32>, %arg83: tensor<320xf32>, %arg84: tensor<320x320x3x3xf32>, %arg85: tensor<320xf32>, %arg86: tensor<320xf32>, %arg87: tensor<320xf32>, %arg88: tensor<640x320x3x3xf32>, %arg89: tensor<640xf32>, %arg90: tensor<640x1280xf32>, %arg91: tensor<640xf32>, %arg92: tensor<640xf32>, %arg93: tensor<640xf32>, %arg94: tensor<640x640x3x3xf32>, %arg95: tensor<640xf32>, %arg96: tensor<640x320x1x1xf32>, %arg97: tensor<640xf32>, %arg98: tensor<640xf32>, %arg99: tensor<640xf32>, %arg100: tensor<640x640xf32>, %arg101: tensor<640xf32>, %arg102: tensor<640xf32>, %arg103: tensor<640xf32>, %arg104: tensor<640x640xf32>, %arg105: tensor<640x640xf32>, %arg106: tensor<640x640xf32>, %arg107: tensor<640x640xf32>, %arg108: tensor<640xf32>, %arg109: tensor<640xf32>, %arg110: tensor<640xf32>, %arg111: tensor<640x640xf32>, %arg112: tensor<640x1024xf32>, %arg113: tensor<1x12x1024xf32>, %arg114: tensor<640x1024xf32>, %arg115: tensor<1x12x1024xf32>, %arg116: tensor<640x640xf32>, %arg117: tensor<640xf32>, %arg118: tensor<640xf32>, %arg119: tensor<640xf32>, %arg120: tensor<5120x640xf32>, %arg121: tensor<5120xf32>, %arg122: tensor<640x2560xf32>, %arg123: tensor<640xf32>, %arg124: tensor<640x640xf32>, %arg125: tensor<640xf32>, %arg126: tensor<640xf32>, %arg127: tensor<640xf32>, %arg128: tensor<640x640x3x3xf32>, %arg129: tensor<640xf32>, %arg130: tensor<640x1280xf32>, %arg131: tensor<640xf32>, %arg132: tensor<640xf32>, %arg133: tensor<640xf32>, %arg134: tensor<640x640x3x3xf32>, %arg135: tensor<640xf32>, %arg136: tensor<640xf32>, %arg137: tensor<640xf32>, %arg138: tensor<640x640xf32>, %arg139: tensor<640xf32>, %arg140: tensor<640xf32>, %arg141: tensor<640xf32>, %arg142: tensor<640x640xf32>, %arg143: tensor<640x640xf32>, %arg144: tensor<640x640xf32>, %arg145: tensor<640x640xf32>, %arg146: tensor<640xf32>, %arg147: tensor<640xf32>, %arg148: tensor<640xf32>, %arg149: tensor<640x640xf32>, %arg150: tensor<640x1024xf32>, %arg151: tensor<1x12x1024xf32>, %arg152: tensor<640x1024xf32>, %arg153: tensor<1x12x1024xf32>, %arg154: tensor<640x640xf32>, %arg155: tensor<640xf32>, %arg156: tensor<640xf32>, %arg157: tensor<640xf32>, %arg158: tensor<5120x640xf32>, %arg159: tensor<5120xf32>, %arg160: tensor<640x2560xf32>, %arg161: tensor<640xf32>, %arg162: tensor<640x640xf32>, %arg163: tensor<640xf32>, %arg164: tensor<640x640x3x3xf32>, %arg165: tensor<640xf32>, %arg166: tensor<640xf32>, %arg167: tensor<640xf32>, %arg168: tensor<1280x640x3x3xf32>, %arg169: tensor<1280xf32>, %arg170: tensor<1280x1280xf32>, %arg171: tensor<1280xf32>, %arg172: tensor<1280xf32>, %arg173: tensor<1280xf32>, %arg174: tensor<1280x1280x3x3xf32>, %arg175: tensor<1280xf32>, %arg176: tensor<1280x640x1x1xf32>, %arg177: tensor<1280xf32>, %arg178: tensor<1280xf32>, %arg179: tensor<1280xf32>, %arg180: tensor<1280x1280xf32>, %arg181: tensor<1280xf32>, %arg182: tensor<1280xf32>, %arg183: tensor<1280xf32>, %arg184: tensor<1280x1280xf32>, %arg185: tensor<1280x1280xf32>, %arg186: tensor<1280x1280xf32>, %arg187: tensor<1280x1280xf32>, %arg188: tensor<1280xf32>, %arg189: tensor<1280xf32>, %arg190: tensor<1280xf32>, %arg191: tensor<1280x1280xf32>, %arg192: tensor<1280x1024xf32>, %arg193: tensor<1x12x1024xf32>, %arg194: tensor<1280x1024xf32>, %arg195: tensor<1x12x1024xf32>, %arg196: tensor<1280x1280xf32>, %arg197: tensor<1280xf32>, %arg198: tensor<1280xf32>, %arg199: tensor<1280xf32>, %arg200: tensor<10240x1280xf32>, %arg201: tensor<10240xf32>, %arg202: tensor<1280x5120xf32>, %arg203: tensor<1280xf32>, %arg204: tensor<1280x1280xf32>, %arg205: tensor<1280xf32>, %arg206: tensor<1280xf32>, %arg207: tensor<1280xf32>, %arg208: tensor<1280x1280x3x3xf32>, %arg209: tensor<1280xf32>, %arg210: tensor<1280x1280xf32>, %arg211: tensor<1280xf32>, %arg212: tensor<1280xf32>, %arg213: tensor<1280xf32>, %arg214: tensor<1280x1280x3x3xf32>, %arg215: tensor<1280xf32>, %arg216: tensor<1280xf32>, %arg217: tensor<1280xf32>, %arg218: tensor<1280x1280xf32>, %arg219: tensor<1280xf32>, %arg220: tensor<1280xf32>, %arg221: tensor<1280xf32>, %arg222: tensor<1280x1280xf32>, %arg223: tensor<1280x1280xf32>, %arg224: tensor<1280x1280xf32>, %arg225: tensor<1280x1280xf32>, %arg226: tensor<1280xf32>, %arg227: tensor<1280xf32>, %arg228: tensor<1280xf32>, %arg229: tensor<1280x1280xf32>, %arg230: tensor<1280x1024xf32>, %arg231: tensor<1x12x1024xf32>, %arg232: tensor<1280x1024xf32>, %arg233: tensor<1x12x1024xf32>, %arg234: tensor<1280x1280xf32>, %arg235: tensor<1280xf32>, %arg236: tensor<1280xf32>, %arg237: tensor<1280xf32>, %arg238: tensor<10240x1280xf32>, %arg239: tensor<10240xf32>, %arg240: tensor<1280x5120xf32>, %arg241: tensor<1280xf32>, %arg242: tensor<1280x1280xf32>, %arg243: tensor<1280xf32>, %arg244: tensor<1280x1280x3x3xf32>, %arg245: tensor<1280xf32>, %arg246: tensor<1280xf32>, %arg247: tensor<1280xf32>, %arg248: tensor<1280x1280x3x3xf32>, %arg249: tensor<1280xf32>, %arg250: tensor<1280x1280xf32>, %arg251: tensor<1280xf32>, %arg252: tensor<1280xf32>, %arg253: tensor<1280xf32>, %arg254: tensor<1280x1280x3x3xf32>, %arg255: tensor<1280xf32>, %arg256: tensor<1280xf32>, %arg257: tensor<1280xf32>, %arg258: tensor<1280x1280x3x3xf32>, %arg259: tensor<1280xf32>, %arg260: tensor<1280x1280xf32>, %arg261: tensor<1280xf32>, %arg262: tensor<1280xf32>, %arg263: tensor<1280xf32>, %arg264: tensor<1280x1280x3x3xf32>, %arg265: tensor<1280xf32>, %arg266: tensor<1280xf32>, %arg267: tensor<1280xf32>, %arg268: tensor<1280x1280x3x3xf32>, %arg269: tensor<1280xf32>, %arg270: tensor<1280x1280xf32>, %arg271: tensor<1280xf32>, %arg272: tensor<1280xf32>, %arg273: tensor<1280xf32>, %arg274: tensor<1280x1280x3x3xf32>, %arg275: tensor<1280xf32>, %arg276: tensor<1280xf32>, %arg277: tensor<1280xf32>, %arg278: tensor<1280x1280xf32>, %arg279: tensor<1280xf32>, %arg280: tensor<1280xf32>, %arg281: tensor<1280xf32>, %arg282: tensor<1280x1280xf32>, %arg283: tensor<1280x1280xf32>, %arg284: tensor<1280x1280xf32>, %arg285: tensor<1280x1280xf32>, %arg286: tensor<1280xf32>, %arg287: tensor<1280xf32>, %arg288: tensor<1280xf32>, %arg289: tensor<1280x1280xf32>, %arg290: tensor<1280x1024xf32>, %arg291: tensor<1x12x1024xf32>, %arg292: tensor<1280x1024xf32>, %arg293: tensor<1x12x1024xf32>, %arg294: tensor<1280x1280xf32>, %arg295: tensor<1280xf32>, %arg296: tensor<1280xf32>, %arg297: tensor<1280xf32>, %arg298: tensor<10240x1280xf32>, %arg299: tensor<10240xf32>, %arg300: tensor<1280x5120xf32>, %arg301: tensor<1280xf32>, %arg302: tensor<1280x1280xf32>, %arg303: tensor<1280xf32>, %arg304: tensor<1280xf32>, %arg305: tensor<1280xf32>, %arg306: tensor<1280x1280x3x3xf32>, %arg307: tensor<1280xf32>, %arg308: tensor<1280x1280xf32>, %arg309: tensor<1280xf32>, %arg310: tensor<1280xf32>, %arg311: tensor<1280xf32>, %arg312: tensor<1280x1280x3x3xf32>, %arg313: tensor<1280xf32>, %arg314: tensor<2560xf32>, %arg315: tensor<2560xf32>, %arg316: tensor<1280x2560x3x3xf32>, %arg317: tensor<1280xf32>, %arg318: tensor<1280x1280xf32>, %arg319: tensor<1280xf32>, %arg320: tensor<1280xf32>, %arg321: tensor<1280xf32>, %arg322: tensor<1280x1280x3x3xf32>, %arg323: tensor<1280xf32>, %arg324: tensor<1280x2560x1x1xf32>, %arg325: tensor<1280xf32>, %arg326: tensor<2560xf32>, %arg327: tensor<2560xf32>, %arg328: tensor<1280x2560x3x3xf32>, %arg329: tensor<1280xf32>, %arg330: tensor<1280x1280xf32>, %arg331: tensor<1280xf32>, %arg332: tensor<1280xf32>, %arg333: tensor<1280xf32>, %arg334: tensor<1280x1280x3x3xf32>, %arg335: tensor<1280xf32>, %arg336: tensor<1280x2560x1x1xf32>, %arg337: tensor<1280xf32>, %arg338: tensor<2560xf32>, %arg339: tensor<2560xf32>, %arg340: tensor<1280x2560x3x3xf32>, %arg341: tensor<1280xf32>, %arg342: tensor<1280x1280xf32>, %arg343: tensor<1280xf32>, %arg344: tensor<1280xf32>, %arg345: tensor<1280xf32>, %arg346: tensor<1280x1280x3x3xf32>, %arg347: tensor<1280xf32>, %arg348: tensor<1280x2560x1x1xf32>, %arg349: tensor<1280xf32>, %arg350: tensor<1280x1280x3x3xf32>, %arg351: tensor<1280xf32>, %arg352: tensor<2560xf32>, %arg353: tensor<2560xf32>, %arg354: tensor<1280x2560x3x3xf32>, %arg355: tensor<1280xf32>, %arg356: tensor<1280x1280xf32>, %arg357: tensor<1280xf32>, %arg358: tensor<1280xf32>, %arg359: tensor<1280xf32>, %arg360: tensor<1280x1280x3x3xf32>, %arg361: tensor<1280xf32>, %arg362: tensor<1280x2560x1x1xf32>, %arg363: tensor<1280xf32>, %arg364: tensor<1280xf32>, %arg365: tensor<1280xf32>, %arg366: tensor<1280x1280xf32>, %arg367: tensor<1280xf32>, %arg368: tensor<1280xf32>, %arg369: tensor<1280xf32>, %arg370: tensor<1280x1280xf32>, %arg371: tensor<1280x1280xf32>, %arg372: tensor<1280x1280xf32>, %arg373: tensor<1280x1280xf32>, %arg374: tensor<1280xf32>, %arg375: tensor<1280xf32>, %arg376: tensor<1280xf32>, %arg377: tensor<1280x1280xf32>, %arg378: tensor<1280x1024xf32>, %arg379: tensor<1x12x1024xf32>, %arg380: tensor<1280x1024xf32>, %arg381: tensor<1x12x1024xf32>, %arg382: tensor<1280x1280xf32>, %arg383: tensor<1280xf32>, %arg384: tensor<1280xf32>, %arg385: tensor<1280xf32>, %arg386: tensor<10240x1280xf32>, %arg387: tensor<10240xf32>, %arg388: tensor<1280x5120xf32>, %arg389: tensor<1280xf32>, %arg390: tensor<1280x1280xf32>, %arg391: tensor<1280xf32>, %arg392: tensor<2560xf32>, %arg393: tensor<2560xf32>, %arg394: tensor<1280x2560x3x3xf32>, %arg395: tensor<1280xf32>, %arg396: tensor<1280x1280xf32>, %arg397: tensor<1280xf32>, %arg398: tensor<1280xf32>, %arg399: tensor<1280xf32>, %arg400: tensor<1280x1280x3x3xf32>, %arg401: tensor<1280xf32>, %arg402: tensor<1280x2560x1x1xf32>, %arg403: tensor<1280xf32>, %arg404: tensor<1280xf32>, %arg405: tensor<1280xf32>, %arg406: tensor<1280x1280xf32>, %arg407: tensor<1280xf32>, %arg408: tensor<1280xf32>, %arg409: tensor<1280xf32>, %arg410: tensor<1280x1280xf32>, %arg411: tensor<1280x1280xf32>, %arg412: tensor<1280x1280xf32>, %arg413: tensor<1280x1280xf32>, %arg414: tensor<1280xf32>, %arg415: tensor<1280xf32>, %arg416: tensor<1280xf32>, %arg417: tensor<1280x1280xf32>, %arg418: tensor<1280x1024xf32>, %arg419: tensor<1x12x1024xf32>, %arg420: tensor<1280x1024xf32>, %arg421: tensor<1x12x1024xf32>, %arg422: tensor<1280x1280xf32>, %arg423: tensor<1280xf32>, %arg424: tensor<1280xf32>, %arg425: tensor<1280xf32>, %arg426: tensor<10240x1280xf32>, %arg427: tensor<10240xf32>, %arg428: tensor<1280x5120xf32>, %arg429: tensor<1280xf32>, %arg430: tensor<1280x1280xf32>, %arg431: tensor<1280xf32>, %arg432: tensor<1920xf32>, %arg433: tensor<1920xf32>, %arg434: tensor<1280x1920x3x3xf32>, %arg435: tensor<1280xf32>, %arg436: tensor<1280x1280xf32>, %arg437: tensor<1280xf32>, %arg438: tensor<1280xf32>, %arg439: tensor<1280xf32>, %arg440: tensor<1280x1280x3x3xf32>, %arg441: tensor<1280xf32>, %arg442: tensor<1280x1920x1x1xf32>, %arg443: tensor<1280xf32>, %arg444: tensor<1280xf32>, %arg445: tensor<1280xf32>, %arg446: tensor<1280x1280xf32>, %arg447: tensor<1280xf32>, %arg448: tensor<1280xf32>, %arg449: tensor<1280xf32>, %arg450: tensor<1280x1280xf32>, %arg451: tensor<1280x1280xf32>, %arg452: tensor<1280x1280xf32>, %arg453: tensor<1280x1280xf32>, %arg454: tensor<1280xf32>, %arg455: tensor<1280xf32>, %arg456: tensor<1280xf32>, %arg457: tensor<1280x1280xf32>, %arg458: tensor<1280x1024xf32>, %arg459: tensor<1x12x1024xf32>, %arg460: tensor<1280x1024xf32>, %arg461: tensor<1x12x1024xf32>, %arg462: tensor<1280x1280xf32>, %arg463: tensor<1280xf32>, %arg464: tensor<1280xf32>, %arg465: tensor<1280xf32>, %arg466: tensor<10240x1280xf32>, %arg467: tensor<10240xf32>, %arg468: tensor<1280x5120xf32>, %arg469: tensor<1280xf32>, %arg470: tensor<1280x1280xf32>, %arg471: tensor<1280xf32>, %arg472: tensor<1280x1280x3x3xf32>, %arg473: tensor<1280xf32>, %arg474: tensor<1920xf32>, %arg475: tensor<1920xf32>, %arg476: tensor<640x1920x3x3xf32>, %arg477: tensor<640xf32>, %arg478: tensor<640x1280xf32>, %arg479: tensor<640xf32>, %arg480: tensor<640xf32>, %arg481: tensor<640xf32>, %arg482: tensor<640x640x3x3xf32>, %arg483: tensor<640xf32>, %arg484: tensor<640x1920x1x1xf32>, %arg485: tensor<640xf32>, %arg486: tensor<640xf32>, %arg487: tensor<640xf32>, %arg488: tensor<640x640xf32>, %arg489: tensor<640xf32>, %arg490: tensor<640xf32>, %arg491: tensor<640xf32>, %arg492: tensor<640x640xf32>, %arg493: tensor<640x640xf32>, %arg494: tensor<640x640xf32>, %arg495: tensor<640x640xf32>, %arg496: tensor<640xf32>, %arg497: tensor<640xf32>, %arg498: tensor<640xf32>, %arg499: tensor<640x640xf32>, %arg500: tensor<640x1024xf32>, %arg501: tensor<1x12x1024xf32>, %arg502: tensor<640x1024xf32>, %arg503: tensor<1x12x1024xf32>, %arg504: tensor<640x640xf32>, %arg505: tensor<640xf32>, %arg506: tensor<640xf32>, %arg507: tensor<640xf32>, %arg508: tensor<5120x640xf32>, %arg509: tensor<5120xf32>, %arg510: tensor<640x2560xf32>, %arg511: tensor<640xf32>, %arg512: tensor<640x640xf32>, %arg513: tensor<640xf32>, %arg514: tensor<1280xf32>, %arg515: tensor<1280xf32>, %arg516: tensor<640x1280x3x3xf32>, %arg517: tensor<640xf32>, %arg518: tensor<640x1280xf32>, %arg519: tensor<640xf32>, %arg520: tensor<640xf32>, %arg521: tensor<640xf32>, %arg522: tensor<640x640x3x3xf32>, %arg523: tensor<640xf32>, %arg524: tensor<640x1280x1x1xf32>, %arg525: tensor<640xf32>, %arg526: tensor<640xf32>, %arg527: tensor<640xf32>, %arg528: tensor<640x640xf32>, %arg529: tensor<640xf32>, %arg530: tensor<640xf32>, %arg531: tensor<640xf32>, %arg532: tensor<640x640xf32>, %arg533: tensor<640x640xf32>, %arg534: tensor<640x640xf32>, %arg535: tensor<640x640xf32>, %arg536: tensor<640xf32>, %arg537: tensor<640xf32>, %arg538: tensor<640xf32>, %arg539: tensor<640x640xf32>, %arg540: tensor<640x1024xf32>, %arg541: tensor<1x12x1024xf32>, %arg542: tensor<640x1024xf32>, %arg543: tensor<1x12x1024xf32>, %arg544: tensor<640x640xf32>, %arg545: tensor<640xf32>, %arg546: tensor<640xf32>, %arg547: tensor<640xf32>, %arg548: tensor<5120x640xf32>, %arg549: tensor<5120xf32>, %arg550: tensor<640x2560xf32>, %arg551: tensor<640xf32>, %arg552: tensor<640x640xf32>, %arg553: tensor<640xf32>, %arg554: tensor<960xf32>, %arg555: tensor<960xf32>, %arg556: tensor<640x960x3x3xf32>, %arg557: tensor<640xf32>, %arg558: tensor<640x1280xf32>, %arg559: tensor<640xf32>, %arg560: tensor<640xf32>, %arg561: tensor<640xf32>, %arg562: tensor<640x640x3x3xf32>, %arg563: tensor<640xf32>, %arg564: tensor<640x960x1x1xf32>, %arg565: tensor<640xf32>, %arg566: tensor<640xf32>, %arg567: tensor<640xf32>, %arg568: tensor<640x640xf32>, %arg569: tensor<640xf32>, %arg570: tensor<640xf32>, %arg571: tensor<640xf32>, %arg572: tensor<640x640xf32>, %arg573: tensor<640x640xf32>, %arg574: tensor<640x640xf32>, %arg575: tensor<640x640xf32>, %arg576: tensor<640xf32>, %arg577: tensor<640xf32>, %arg578: tensor<640xf32>, %arg579: tensor<640x640xf32>, %arg580: tensor<640x1024xf32>, %arg581: tensor<1x12x1024xf32>, %arg582: tensor<640x1024xf32>, %arg583: tensor<1x12x1024xf32>, %arg584: tensor<640x640xf32>, %arg585: tensor<640xf32>, %arg586: tensor<640xf32>, %arg587: tensor<640xf32>, %arg588: tensor<5120x640xf32>, %arg589: tensor<5120xf32>, %arg590: tensor<640x2560xf32>, %arg591: tensor<640xf32>, %arg592: tensor<640x640xf32>, %arg593: tensor<640xf32>, %arg594: tensor<640x640x3x3xf32>, %arg595: tensor<640xf32>, %arg596: tensor<960xf32>, %arg597: tensor<960xf32>, %arg598: tensor<320x960x3x3xf32>, %arg599: tensor<320xf32>, %arg600: tensor<320x1280xf32>, %arg601: tensor<320xf32>, %arg602: tensor<320xf32>, %arg603: tensor<320xf32>, %arg604: tensor<320x320x3x3xf32>, %arg605: tensor<320xf32>, %arg606: tensor<320x960x1x1xf32>, %arg607: tensor<320xf32>, %arg608: tensor<320xf32>, %arg609: tensor<320xf32>, %arg610: tensor<320x320xf32>, %arg611: tensor<320xf32>, %arg612: tensor<320xf32>, %arg613: tensor<320xf32>, %arg614: tensor<320x320xf32>, %arg615: tensor<320x320xf32>, %arg616: tensor<320x320xf32>, %arg617: tensor<320x320xf32>, %arg618: tensor<320xf32>, %arg619: tensor<320xf32>, %arg620: tensor<320xf32>, %arg621: tensor<320x320xf32>, %arg622: tensor<320x1024xf32>, %arg623: tensor<1x12x1024xf32>, %arg624: tensor<320x1024xf32>, %arg625: tensor<1x12x1024xf32>, %arg626: tensor<320x320xf32>, %arg627: tensor<320xf32>, %arg628: tensor<320xf32>, %arg629: tensor<320xf32>, %arg630: tensor<2560x320xf32>, %arg631: tensor<2560xf32>, %arg632: tensor<320x1280xf32>, %arg633: tensor<320xf32>, %arg634: tensor<320x320xf32>, %arg635: tensor<320xf32>, %arg636: tensor<640xf32>, %arg637: tensor<640xf32>, %arg638: tensor<320x640x3x3xf32>, %arg639: tensor<320xf32>, %arg640: tensor<320x1280xf32>, %arg641: tensor<320xf32>, %arg642: tensor<320xf32>, %arg643: tensor<320xf32>, %arg644: tensor<320x320x3x3xf32>, %arg645: tensor<320xf32>, %arg646: tensor<320x640x1x1xf32>, %arg647: tensor<320xf32>, %arg648: tensor<320xf32>, %arg649: tensor<320xf32>, %arg650: tensor<320x320xf32>, %arg651: tensor<320xf32>, %arg652: tensor<320xf32>, %arg653: tensor<320xf32>, %arg654: tensor<320x320xf32>, %arg655: tensor<320x320xf32>, %arg656: tensor<320x320xf32>, %arg657: tensor<320x320xf32>, %arg658: tensor<320xf32>, %arg659: tensor<320xf32>, %arg660: tensor<320xf32>, %arg661: tensor<320x320xf32>, %arg662: tensor<320x1024xf32>, %arg663: tensor<1x12x1024xf32>, %arg664: tensor<320x1024xf32>, %arg665: tensor<1x12x1024xf32>, %arg666: tensor<320x320xf32>, %arg667: tensor<320xf32>, %arg668: tensor<320xf32>, %arg669: tensor<320xf32>, %arg670: tensor<2560x320xf32>, %arg671: tensor<2560xf32>, %arg672: tensor<320x1280xf32>, %arg673: tensor<320xf32>, %arg674: tensor<320x320xf32>, %arg675: tensor<320xf32>, %arg676: tensor<640xf32>, %arg677: tensor<640xf32>, %arg678: tensor<320x640x3x3xf32>, %arg679: tensor<320xf32>, %arg680: tensor<320x1280xf32>, %arg681: tensor<320xf32>, %arg682: tensor<320xf32>, %arg683: tensor<320xf32>, %arg684: tensor<320x320x3x3xf32>, %arg685: tensor<320xf32>, %arg686: tensor<320x640x1x1xf32>, %arg687: tensor<320xf32>, %arg688: tensor<320xf32>, %arg689: tensor<320xf32>, %arg690: tensor<320x320xf32>, %arg691: tensor<320xf32>, %arg692: tensor<320xf32>, %arg693: tensor<320xf32>, %arg694: tensor<320x320xf32>, %arg695: tensor<320x320xf32>, %arg696: tensor<320x320xf32>, %arg697: tensor<320x320xf32>, %arg698: tensor<320xf32>, %arg699: tensor<320xf32>, %arg700: tensor<320xf32>, %arg701: tensor<320x320xf32>, %arg702: tensor<320x1024xf32>, %arg703: tensor<1x12x1024xf32>, %arg704: tensor<320x1024xf32>, %arg705: tensor<1x12x1024xf32>, %arg706: tensor<320x320xf32>, %arg707: tensor<320xf32>, %arg708: tensor<320xf32>, %arg709: tensor<320xf32>, %arg710: tensor<2560x320xf32>, %arg711: tensor<2560xf32>, %arg712: tensor<320x1280xf32>, %arg713: tensor<320xf32>, %arg714: tensor<320x320xf32>, %arg715: tensor<320xf32>, %arg716: tensor<320xf32>, %arg717: tensor<320xf32>, %arg718: tensor<4x320x3x3xf32>, %arg719: tensor<4xf32>):
    %0 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = "tosa.add"(%arg0, %0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %2 = "tosa.const"() <{value = dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F0000000000000080000000000000008100000000000000820000000000000083000000000000008400000000000000850000000000000086000000000000008700000000000000880000000000000089000000000000008A000000000000008B000000000000008C000000000000008D000000000000008E000000000000008F0000000000000090000000000000009100000000000000920000000000000093000000000000009400000000000000950000000000000096000000000000009700000000000000980000000000000099000000000000009A000000000000009B000000000000009C000000000000009D000000000000009E000000000000009F00000000000000"> : tensor<160xi64>}> : () -> tensor<160xi64>
    %3 = "tosa.cast"(%2) : (tensor<160xi64>) -> tensor<160xf32>
    %4 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<160xf32>}> : () -> tensor<160xf32>
    %5 = "tosa.mul"(%3, %4) <{shift = 0 : i8}> : (tensor<160xf32>, tensor<160xf32>) -> tensor<160xf32>
    %6 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<160xf32>}> : () -> tensor<160xf32>
    %7 = "tosa.add"(%5, %6) : (tensor<160xf32>, tensor<160xf32>) -> tensor<160xf32>
    %8 = "tosa.const"() <{value = dense<-9.21034049> : tensor<160xf32>}> : () -> tensor<160xf32>
    %9 = "tosa.mul"(%7, %8) <{shift = 0 : i8}> : (tensor<160xf32>, tensor<160xf32>) -> tensor<160xf32>
    %10 = "tosa.const"() <{value = dense<1.600000e+02> : tensor<160xf32>}> : () -> tensor<160xf32>
    %11 = "tosa.reciprocal"(%10) : (tensor<160xf32>) -> tensor<160xf32>
    %12 = "tosa.mul"(%9, %11) <{shift = 0 : i8}> : (tensor<160xf32>, tensor<160xf32>) -> tensor<160xf32>
    %13 = "tosa.exp"(%12) : (tensor<160xf32>) -> tensor<160xf32>
    %14 = "tensor.extract_slice"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (tensor<1xf32>) -> tensor<1xf32>
    %15 = "tosa.reshape"(%14) <{new_shape = array<i64: 1, 1>}> : (tensor<1xf32>) -> tensor<1x1xf32>
    %16 = "tosa.reshape"(%13) <{new_shape = array<i64: 1, 160>}> : (tensor<160xf32>) -> tensor<1x160xf32>
    %17 = "tensor.extract_slice"(%16) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 160>, static_strides = array<i64: 1, 1>}> : (tensor<1x160xf32>) -> tensor<1x160xf32>
    %18 = "tosa.mul"(%15, %17) <{shift = 0 : i8}> : (tensor<1x1xf32>, tensor<1x160xf32>) -> tensor<1x160xf32>
    %19 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x160xf32>}> : () -> tensor<1x160xf32>
    %20 = "tosa.mul"(%18, %19) <{shift = 0 : i8}> : (tensor<1x160xf32>, tensor<1x160xf32>) -> tensor<1x160xf32>
    %21 = "math.sin"(%20) <{fastmath = #arith.fastmath<none>}> : (tensor<1x160xf32>) -> tensor<1x160xf32>
    %22 = "math.cos"(%20) <{fastmath = #arith.fastmath<none>}> : (tensor<1x160xf32>) -> tensor<1x160xf32>
    %23 = "tensor.empty"() : () -> tensor<1x320xf32>
    %24 = "tensor.insert_slice"(%21, %23) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 160>, static_strides = array<i64: 1, 1>}> : (tensor<1x160xf32>, tensor<1x320xf32>) -> tensor<1x320xf32>
    %25 = "tensor.insert_slice"(%22, %24) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 160>, static_sizes = array<i64: 1, 160>, static_strides = array<i64: 1, 1>}> : (tensor<1x160xf32>, tensor<1x320xf32>) -> tensor<1x320xf32>
    %26 = "tensor.extract_slice"(%25) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %27 = "tensor.extract_slice"(%26) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 160>, static_sizes = array<i64: 1, 160>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x160xf32>
    %28 = "tensor.extract_slice"(%25) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %29 = "tensor.extract_slice"(%28) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 160>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x160xf32>
    %30 = "tensor.empty"() : () -> tensor<1x320xf32>
    %31 = "tensor.insert_slice"(%27, %30) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 160>, static_strides = array<i64: 1, 1>}> : (tensor<1x160xf32>, tensor<1x320xf32>) -> tensor<1x320xf32>
    %32 = "tensor.insert_slice"(%29, %31) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 160>, static_sizes = array<i64: 1, 160>, static_strides = array<i64: 1, 1>}> : (tensor<1x160xf32>, tensor<1x320xf32>) -> tensor<1x320xf32>
    %33 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %34 = "tosa.transpose"(%arg1, %33) : (tensor<1280x320xf32>, tensor<2xi32>) -> tensor<320x1280xf32>
    %35 = "tosa.reshape"(%32) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<1x320xf32>) -> tensor<1x1x320xf32>
    %36 = "tosa.reshape"(%34) <{new_shape = array<i64: 1, 320, 1280>}> : (tensor<320x1280xf32>) -> tensor<1x320x1280xf32>
    %37 = "tosa.matmul"(%35, %36) : (tensor<1x1x320xf32>, tensor<1x320x1280xf32>) -> tensor<1x1x1280xf32>
    %38 = "tosa.reshape"(%37) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %39 = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %40 = "tosa.add"(%39, %38) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %41 = "tosa.sigmoid"(%40) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %42 = "tosa.mul"(%40, %41) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %43 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %44 = "tosa.transpose"(%arg3, %43) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %45 = "tosa.reshape"(%42) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %46 = "tosa.reshape"(%44) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %47 = "tosa.matmul"(%45, %46) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %48 = "tosa.reshape"(%47) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %49 = "tosa.reshape"(%arg4) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %50 = "tosa.add"(%49, %48) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %51 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %52 = "tosa.transpose"(%arg5, %51) : (tensor<1x4x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x4xf32>
    %53 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %54 = "tosa.transpose"(%arg6, %53) : (tensor<320x4x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x4xf32>
    %55 = "tosa.conv2d"(%52, %54, %arg7) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x4xf32>, tensor<320x3x3x4xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %56 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %57 = "tosa.transpose"(%55, %56) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %58 = "tosa.reshape"(%57) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %59 = "tosa.reduce_sum"(%58) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %60 = "tosa.reduce_sum"(%59) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %61 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %62 = "tosa.reciprocal"(%61) : (tensor<1xf32>) -> tensor<1xf32>
    %63 = "tosa.mul"(%62, %60) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %64 = "tosa.sub"(%58, %63) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %65 = "tosa.mul"(%64, %64) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %66 = "tosa.reduce_sum"(%65) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %67 = "tosa.reduce_sum"(%66) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %68 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %69 = "tosa.reciprocal"(%68) : (tensor<1xf32>) -> tensor<1xf32>
    %70 = "tosa.mul"(%69, %67) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %71 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %72 = "tosa.add"(%70, %71) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %73 = "tosa.rsqrt"(%72) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %74 = "tosa.sub"(%58, %63) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %75 = "tosa.mul"(%74, %73) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %76 = "tosa.reshape"(%75) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %77 = "tosa.reshape"(%arg8) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %78 = "tosa.reshape"(%77) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %79 = "tosa.reshape"(%78) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %80 = "tosa.reshape"(%arg9) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %81 = "tosa.reshape"(%80) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %82 = "tosa.reshape"(%81) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %83 = "tosa.mul"(%76, %82) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %84 = "tosa.add"(%83, %79) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %85 = "tosa.sigmoid"(%84) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %86 = "tosa.mul"(%84, %85) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %87 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %88 = "tosa.transpose"(%86, %87) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %89 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %90 = "tosa.transpose"(%arg10, %89) : (tensor<320x320x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x320xf32>
    %91 = "tosa.conv2d"(%88, %90, %arg11) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x320xf32>, tensor<320x3x3x320xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %92 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %93 = "tosa.transpose"(%91, %92) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %94 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %95 = "tosa.mul"(%50, %94) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %96 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %97 = "tosa.transpose"(%arg12, %96) : (tensor<320x1280xf32>, tensor<2xi32>) -> tensor<1280x320xf32>
    %98 = "tosa.reshape"(%95) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %99 = "tosa.reshape"(%97) <{new_shape = array<i64: 1, 1280, 320>}> : (tensor<1280x320xf32>) -> tensor<1x1280x320xf32>
    %100 = "tosa.matmul"(%98, %99) : (tensor<1x1x1280xf32>, tensor<1x1280x320xf32>) -> tensor<1x1x320xf32>
    %101 = "tosa.reshape"(%100) <{new_shape = array<i64: 1, 320>}> : (tensor<1x1x320xf32>) -> tensor<1x320xf32>
    %102 = "tosa.reshape"(%arg13) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %103 = "tosa.add"(%102, %101) : (tensor<1x320xf32>, tensor<1x320xf32>) -> tensor<1x320xf32>
    %104 = "tensor.extract_slice"(%103) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %105 = "tensor.extract_slice"(%104) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %106 = "tosa.reshape"(%105) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %107 = "tosa.reshape"(%106) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %108 = "tosa.add"(%93, %107) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %109 = "tosa.reshape"(%108) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %110 = "tosa.reduce_sum"(%109) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %111 = "tosa.reduce_sum"(%110) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %112 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %113 = "tosa.reciprocal"(%112) : (tensor<1xf32>) -> tensor<1xf32>
    %114 = "tosa.mul"(%113, %111) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %115 = "tosa.sub"(%109, %114) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %116 = "tosa.mul"(%115, %115) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %117 = "tosa.reduce_sum"(%116) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %118 = "tosa.reduce_sum"(%117) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %119 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %120 = "tosa.reciprocal"(%119) : (tensor<1xf32>) -> tensor<1xf32>
    %121 = "tosa.mul"(%120, %118) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %122 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %123 = "tosa.add"(%121, %122) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %124 = "tosa.rsqrt"(%123) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %125 = "tosa.sub"(%109, %114) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %126 = "tosa.mul"(%125, %124) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %127 = "tosa.reshape"(%126) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %128 = "tosa.reshape"(%arg14) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %129 = "tosa.reshape"(%128) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %130 = "tosa.reshape"(%129) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %131 = "tosa.reshape"(%arg15) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %132 = "tosa.reshape"(%131) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %133 = "tosa.reshape"(%132) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %134 = "tosa.mul"(%127, %133) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %135 = "tosa.add"(%134, %130) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %136 = "tosa.sigmoid"(%135) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %137 = "tosa.mul"(%135, %136) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %138 = "tosa.identity"(%137) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %139 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %140 = "tosa.transpose"(%138, %139) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %141 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %142 = "tosa.transpose"(%arg16, %141) : (tensor<320x320x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x320xf32>
    %143 = "tosa.conv2d"(%140, %142, %arg17) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x320xf32>, tensor<320x3x3x320xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %144 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %145 = "tosa.transpose"(%143, %144) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %146 = "tosa.add"(%57, %145) : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %147 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x320x64x64xf32>}> : () -> tensor<1x320x64x64xf32>
    %148 = "tosa.reciprocal"(%147) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %149 = "tosa.mul"(%146, %148) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %150 = "tosa.reshape"(%149) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %151 = "tosa.reduce_sum"(%150) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %152 = "tosa.reduce_sum"(%151) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %153 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %154 = "tosa.reciprocal"(%153) : (tensor<1xf32>) -> tensor<1xf32>
    %155 = "tosa.mul"(%154, %152) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %156 = "tosa.sub"(%150, %155) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %157 = "tosa.mul"(%156, %156) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %158 = "tosa.reduce_sum"(%157) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %159 = "tosa.reduce_sum"(%158) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %160 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %161 = "tosa.reciprocal"(%160) : (tensor<1xf32>) -> tensor<1xf32>
    %162 = "tosa.mul"(%161, %159) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %163 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %164 = "tosa.add"(%162, %163) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %165 = "tosa.rsqrt"(%164) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %166 = "tosa.sub"(%150, %155) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %167 = "tosa.mul"(%166, %165) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %168 = "tosa.reshape"(%167) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %169 = "tosa.reshape"(%arg18) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %170 = "tosa.reshape"(%169) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %171 = "tosa.reshape"(%170) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %172 = "tosa.reshape"(%arg19) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %173 = "tosa.reshape"(%172) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %174 = "tosa.reshape"(%173) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %175 = "tosa.mul"(%168, %174) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %176 = "tosa.add"(%175, %171) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %177 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %178 = "tosa.transpose"(%176, %177) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %179 = "tosa.reshape"(%178) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x64x64x320xf32>) -> tensor<1x4096x320xf32>
    %180 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %181 = "tosa.transpose"(%arg20, %180) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %182 = "tosa.reshape"(%179) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %183 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %184 = "linalg.matmul"(%182, %181, %183) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %185 = "tosa.reshape"(%184) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %186 = "tosa.reshape"(%arg21) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %187 = "tosa.add"(%185, %186) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %188 = "tosa.reduce_sum"(%187) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %189 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %190 = "tosa.reciprocal"(%189) : (tensor<1xf32>) -> tensor<1xf32>
    %191 = "tosa.mul"(%190, %188) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %192 = "tosa.sub"(%187, %191) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %193 = "tosa.mul"(%192, %192) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %194 = "tosa.reduce_sum"(%193) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %195 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %196 = "tosa.reciprocal"(%195) : (tensor<1xf32>) -> tensor<1xf32>
    %197 = "tosa.mul"(%196, %194) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %198 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %199 = "tosa.add"(%197, %198) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %200 = "tosa.rsqrt"(%199) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %201 = "tosa.sub"(%187, %191) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %202 = "tosa.mul"(%201, %200) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %203 = "tosa.reshape"(%arg22) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %204 = "tosa.mul"(%202, %203) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %205 = "tosa.reshape"(%arg23) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %206 = "tosa.add"(%204, %205) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %207 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %208 = "tosa.transpose"(%arg24, %207) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %209 = "tosa.reshape"(%206) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %210 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %211 = "linalg.matmul"(%209, %208, %210) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %212 = "tosa.reshape"(%211) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %213 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %214 = "tosa.transpose"(%arg25, %213) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %215 = "tosa.reshape"(%206) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %216 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %217 = "linalg.matmul"(%215, %214, %216) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %218 = "tosa.reshape"(%217) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %219 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %220 = "tosa.transpose"(%arg26, %219) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %221 = "tosa.reshape"(%206) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %222 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %223 = "linalg.matmul"(%221, %220, %222) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %224 = "tosa.reshape"(%223) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %225 = "tosa.reshape"(%212) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %226 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %227 = "tosa.transpose"(%225, %226) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %228 = "tosa.reshape"(%218) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %229 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %230 = "tosa.transpose"(%228, %229) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %231 = "tosa.reshape"(%224) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %232 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %233 = "tosa.transpose"(%231, %232) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %234 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x4096xf32>}> : () -> tensor<4096x4096xf32>
    %235 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %236 = "tosa.transpose"(%230, %235) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x5x64x4096xf32>
    %237 = "tosa.reshape"(%227) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %238 = "tosa.reshape"(%236) <{new_shape = array<i64: 5, 64, 4096>}> : (tensor<1x5x64x4096xf32>) -> tensor<5x64x4096xf32>
    %239 = "tosa.matmul"(%237, %238) : (tensor<5x4096x64xf32>, tensor<5x64x4096xf32>) -> tensor<5x4096x4096xf32>
    %240 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<5x4096x4096xf32>}> : () -> tensor<5x4096x4096xf32>
    %241 = "tosa.mul"(%239, %240) <{shift = 0 : i8}> : (tensor<5x4096x4096xf32>, tensor<5x4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %242 = "tosa.add"(%241, %234) : (tensor<5x4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %243 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<5x4096x4096xf32>}> : () -> tensor<5x4096x4096xf32>
    %244 = "linalg.softmax"(%242, %243) <{dimension = 3 : i64}> : (tensor<5x4096x4096xf32>, tensor<5x4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %245 = "tosa.reshape"(%233) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %246 = "tosa.matmul"(%244, %245) : (tensor<5x4096x4096xf32>, tensor<5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %247 = "tosa.reshape"(%246) <{new_shape = array<i64: 1, 5, 4096, 64>}> : (tensor<5x4096x64xf32>) -> tensor<1x5x4096x64xf32>
    %248 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %249 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %250 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %251 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %252 = "tosa.transpose"(%247, %251) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x4096x5x64xf32>
    %253 = "tosa.reshape"(%252) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x4096x5x64xf32>) -> tensor<1x4096x320xf32>
    %254 = "tosa.reshape"(%253) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %255 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %256 = "tosa.transpose"(%arg27, %255) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %257 = "tosa.reshape"(%254) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %258 = "tosa.reshape"(%256) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %259 = "tosa.matmul"(%257, %258) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %260 = "tosa.reshape"(%259) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %261 = "tosa.reshape"(%arg28) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %262 = "tosa.add"(%261, %260) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %263 = "tosa.reshape"(%262) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %264 = "tosa.identity"(%263) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %265 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x320xf32>}> : () -> tensor<1x4096x320xf32>
    %266 = "tosa.reciprocal"(%265) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %267 = "tosa.mul"(%264, %266) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %268 = "tosa.add"(%267, %187) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %269 = "tosa.reduce_sum"(%268) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %270 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %271 = "tosa.reciprocal"(%270) : (tensor<1xf32>) -> tensor<1xf32>
    %272 = "tosa.mul"(%271, %269) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %273 = "tosa.sub"(%268, %272) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %274 = "tosa.mul"(%273, %273) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %275 = "tosa.reduce_sum"(%274) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %276 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %277 = "tosa.reciprocal"(%276) : (tensor<1xf32>) -> tensor<1xf32>
    %278 = "tosa.mul"(%277, %275) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %279 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %280 = "tosa.add"(%278, %279) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %281 = "tosa.rsqrt"(%280) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %282 = "tosa.sub"(%268, %272) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %283 = "tosa.mul"(%282, %281) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %284 = "tosa.reshape"(%arg29) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %285 = "tosa.mul"(%283, %284) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %286 = "tosa.reshape"(%arg30) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %287 = "tosa.add"(%285, %286) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %288 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %289 = "tosa.transpose"(%arg31, %288) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %290 = "tosa.reshape"(%287) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %291 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %292 = "linalg.matmul"(%290, %289, %291) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %293 = "tosa.reshape"(%292) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %294 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %295 = "tosa.transpose"(%arg32, %294) : (tensor<320x1024xf32>, tensor<2xi32>) -> tensor<1024x320xf32>
    %296 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %297 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x320xf32>}> : () -> tensor<12x320xf32>
    %298 = "linalg.matmul"(%296, %295, %297) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x320xf32>, tensor<12x320xf32>) -> tensor<12x320xf32>
    %299 = "tosa.reshape"(%298) <{new_shape = array<i64: 1, 12, 320>}> : (tensor<12x320xf32>) -> tensor<1x12x320xf32>
    %300 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %301 = "tosa.transpose"(%arg34, %300) : (tensor<320x1024xf32>, tensor<2xi32>) -> tensor<1024x320xf32>
    %302 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %303 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x320xf32>}> : () -> tensor<12x320xf32>
    %304 = "linalg.matmul"(%302, %301, %303) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x320xf32>, tensor<12x320xf32>) -> tensor<12x320xf32>
    %305 = "tosa.reshape"(%304) <{new_shape = array<i64: 1, 12, 320>}> : (tensor<12x320xf32>) -> tensor<1x12x320xf32>
    %306 = "tosa.reshape"(%293) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %307 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %308 = "tosa.transpose"(%306, %307) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %309 = "tosa.reshape"(%299) <{new_shape = array<i64: 1, 12, 5, 64>}> : (tensor<1x12x320xf32>) -> tensor<1x12x5x64xf32>
    %310 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %311 = "tosa.transpose"(%309, %310) : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %312 = "tosa.reshape"(%305) <{new_shape = array<i64: 1, 12, 5, 64>}> : (tensor<1x12x320xf32>) -> tensor<1x12x5x64xf32>
    %313 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %314 = "tosa.transpose"(%312, %313) : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %315 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x12xf32>}> : () -> tensor<4096x12xf32>
    %316 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %317 = "tosa.transpose"(%311, %316) : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x5x64x12xf32>
    %318 = "tosa.reshape"(%308) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %319 = "tosa.reshape"(%317) <{new_shape = array<i64: 5, 64, 12>}> : (tensor<1x5x64x12xf32>) -> tensor<5x64x12xf32>
    %320 = "tosa.matmul"(%318, %319) : (tensor<5x4096x64xf32>, tensor<5x64x12xf32>) -> tensor<5x4096x12xf32>
    %321 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<5x4096x12xf32>}> : () -> tensor<5x4096x12xf32>
    %322 = "tosa.mul"(%320, %321) <{shift = 0 : i8}> : (tensor<5x4096x12xf32>, tensor<5x4096x12xf32>) -> tensor<5x4096x12xf32>
    %323 = "tosa.add"(%322, %315) : (tensor<5x4096x12xf32>, tensor<4096x12xf32>) -> tensor<5x4096x12xf32>
    %324 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<5x4096x12xf32>}> : () -> tensor<5x4096x12xf32>
    %325 = "linalg.softmax"(%323, %324) <{dimension = 3 : i64}> : (tensor<5x4096x12xf32>, tensor<5x4096x12xf32>) -> tensor<5x4096x12xf32>
    %326 = "tosa.reshape"(%314) <{new_shape = array<i64: 5, 12, 64>}> : (tensor<1x5x12x64xf32>) -> tensor<5x12x64xf32>
    %327 = "tosa.matmul"(%325, %326) : (tensor<5x4096x12xf32>, tensor<5x12x64xf32>) -> tensor<5x4096x64xf32>
    %328 = "tosa.reshape"(%327) <{new_shape = array<i64: 1, 5, 4096, 64>}> : (tensor<5x4096x64xf32>) -> tensor<1x5x4096x64xf32>
    %329 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %330 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %331 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %332 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %333 = "tosa.transpose"(%328, %332) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x4096x5x64xf32>
    %334 = "tosa.reshape"(%333) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x4096x5x64xf32>) -> tensor<1x4096x320xf32>
    %335 = "tosa.reshape"(%334) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %336 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %337 = "tosa.transpose"(%arg36, %336) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %338 = "tosa.reshape"(%335) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %339 = "tosa.reshape"(%337) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %340 = "tosa.matmul"(%338, %339) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %341 = "tosa.reshape"(%340) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %342 = "tosa.reshape"(%arg37) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %343 = "tosa.add"(%342, %341) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %344 = "tosa.reshape"(%343) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %345 = "tosa.identity"(%344) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %346 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x320xf32>}> : () -> tensor<1x4096x320xf32>
    %347 = "tosa.reciprocal"(%346) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %348 = "tosa.mul"(%345, %347) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %349 = "tosa.add"(%348, %268) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %350 = "tosa.reduce_sum"(%349) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %351 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %352 = "tosa.reciprocal"(%351) : (tensor<1xf32>) -> tensor<1xf32>
    %353 = "tosa.mul"(%352, %350) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %354 = "tosa.sub"(%349, %353) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %355 = "tosa.mul"(%354, %354) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %356 = "tosa.reduce_sum"(%355) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %357 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %358 = "tosa.reciprocal"(%357) : (tensor<1xf32>) -> tensor<1xf32>
    %359 = "tosa.mul"(%358, %356) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %360 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %361 = "tosa.add"(%359, %360) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %362 = "tosa.rsqrt"(%361) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %363 = "tosa.sub"(%349, %353) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %364 = "tosa.mul"(%363, %362) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %365 = "tosa.reshape"(%arg38) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %366 = "tosa.mul"(%364, %365) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %367 = "tosa.reshape"(%arg39) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %368 = "tosa.add"(%366, %367) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %369 = "tosa.reshape"(%368) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %370 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %371 = "tosa.transpose"(%arg40, %370) : (tensor<2560x320xf32>, tensor<2xi32>) -> tensor<320x2560xf32>
    %372 = "tosa.reshape"(%369) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %373 = "tosa.reshape"(%371) <{new_shape = array<i64: 1, 320, 2560>}> : (tensor<320x2560xf32>) -> tensor<1x320x2560xf32>
    %374 = "tosa.matmul"(%372, %373) : (tensor<1x4096x320xf32>, tensor<1x320x2560xf32>) -> tensor<1x4096x2560xf32>
    %375 = "tosa.reshape"(%374) <{new_shape = array<i64: 4096, 2560>}> : (tensor<1x4096x2560xf32>) -> tensor<4096x2560xf32>
    %376 = "tosa.reshape"(%arg41) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %377 = "tosa.add"(%376, %375) : (tensor<1x2560xf32>, tensor<4096x2560xf32>) -> tensor<4096x2560xf32>
    %378 = "tosa.reshape"(%377) <{new_shape = array<i64: 1, 4096, 2560>}> : (tensor<4096x2560xf32>) -> tensor<1x4096x2560xf32>
    %379 = "tosa.slice"(%378) <{size = array<i64: 0, 0, 1280>, start = array<i64: 0, 0, 0>}> : (tensor<1x4096x2560xf32>) -> tensor<1x4096x1280xf32>
    %380 = "tosa.slice"(%378) <{size = array<i64: 0, 0, 2560>, start = array<i64: 0, 0, 1280>}> : (tensor<1x4096x2560xf32>) -> tensor<1x4096x1280xf32>
    %381 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %382 = "tosa.mul"(%380, %381) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %383 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %384 = "tosa.mul"(%380, %383) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %385 = "math.erf"(%384) <{fastmath = #arith.fastmath<none>}> : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %386 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %387 = "tosa.add"(%385, %386) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %388 = "tosa.mul"(%382, %387) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %389 = "tosa.mul"(%379, %388) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %390 = "tosa.identity"(%389) : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %391 = "tosa.reshape"(%390) <{new_shape = array<i64: 4096, 1280>}> : (tensor<1x4096x1280xf32>) -> tensor<4096x1280xf32>
    %392 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %393 = "tosa.transpose"(%arg42, %392) : (tensor<320x1280xf32>, tensor<2xi32>) -> tensor<1280x320xf32>
    %394 = "tosa.reshape"(%391) <{new_shape = array<i64: 1, 4096, 1280>}> : (tensor<4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %395 = "tosa.reshape"(%393) <{new_shape = array<i64: 1, 1280, 320>}> : (tensor<1280x320xf32>) -> tensor<1x1280x320xf32>
    %396 = "tosa.matmul"(%394, %395) : (tensor<1x4096x1280xf32>, tensor<1x1280x320xf32>) -> tensor<1x4096x320xf32>
    %397 = "tosa.reshape"(%396) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %398 = "tosa.reshape"(%arg43) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %399 = "tosa.add"(%398, %397) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %400 = "tosa.reshape"(%399) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %401 = "tosa.add"(%400, %349) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %402 = "tosa.reshape"(%401) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %403 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %404 = "tosa.transpose"(%arg44, %403) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %405 = "tosa.reshape"(%402) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %406 = "tosa.reshape"(%404) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %407 = "tosa.matmul"(%405, %406) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %408 = "tosa.reshape"(%407) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %409 = "tosa.reshape"(%arg45) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %410 = "tosa.add"(%409, %408) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %411 = "tosa.reshape"(%410) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %412 = "tosa.reshape"(%411) <{new_shape = array<i64: 1, 64, 64, 320>}> : (tensor<1x4096x320xf32>) -> tensor<1x64x64x320xf32>
    %413 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %414 = "tosa.transpose"(%412, %413) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %415 = "tosa.identity"(%414) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %416 = "tosa.add"(%415, %149) : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %417 = "tosa.reshape"(%416) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %418 = "tosa.reduce_sum"(%417) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %419 = "tosa.reduce_sum"(%418) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %420 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %421 = "tosa.reciprocal"(%420) : (tensor<1xf32>) -> tensor<1xf32>
    %422 = "tosa.mul"(%421, %419) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %423 = "tosa.sub"(%417, %422) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %424 = "tosa.mul"(%423, %423) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %425 = "tosa.reduce_sum"(%424) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %426 = "tosa.reduce_sum"(%425) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %427 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %428 = "tosa.reciprocal"(%427) : (tensor<1xf32>) -> tensor<1xf32>
    %429 = "tosa.mul"(%428, %426) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %430 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %431 = "tosa.add"(%429, %430) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %432 = "tosa.rsqrt"(%431) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %433 = "tosa.sub"(%417, %422) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %434 = "tosa.mul"(%433, %432) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %435 = "tosa.reshape"(%434) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %436 = "tosa.reshape"(%arg46) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %437 = "tosa.reshape"(%436) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %438 = "tosa.reshape"(%437) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %439 = "tosa.reshape"(%arg47) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %440 = "tosa.reshape"(%439) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %441 = "tosa.reshape"(%440) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %442 = "tosa.mul"(%435, %441) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %443 = "tosa.add"(%442, %438) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %444 = "tosa.sigmoid"(%443) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %445 = "tosa.mul"(%443, %444) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %446 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %447 = "tosa.transpose"(%445, %446) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %448 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %449 = "tosa.transpose"(%arg48, %448) : (tensor<320x320x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x320xf32>
    %450 = "tosa.conv2d"(%447, %449, %arg49) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x320xf32>, tensor<320x3x3x320xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %451 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %452 = "tosa.transpose"(%450, %451) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %453 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %454 = "tosa.mul"(%50, %453) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %455 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %456 = "tosa.transpose"(%arg50, %455) : (tensor<320x1280xf32>, tensor<2xi32>) -> tensor<1280x320xf32>
    %457 = "tosa.reshape"(%454) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %458 = "tosa.reshape"(%456) <{new_shape = array<i64: 1, 1280, 320>}> : (tensor<1280x320xf32>) -> tensor<1x1280x320xf32>
    %459 = "tosa.matmul"(%457, %458) : (tensor<1x1x1280xf32>, tensor<1x1280x320xf32>) -> tensor<1x1x320xf32>
    %460 = "tosa.reshape"(%459) <{new_shape = array<i64: 1, 320>}> : (tensor<1x1x320xf32>) -> tensor<1x320xf32>
    %461 = "tosa.reshape"(%arg51) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %462 = "tosa.add"(%461, %460) : (tensor<1x320xf32>, tensor<1x320xf32>) -> tensor<1x320xf32>
    %463 = "tensor.extract_slice"(%462) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %464 = "tensor.extract_slice"(%463) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %465 = "tosa.reshape"(%464) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %466 = "tosa.reshape"(%465) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %467 = "tosa.add"(%452, %466) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %468 = "tosa.reshape"(%467) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %469 = "tosa.reduce_sum"(%468) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %470 = "tosa.reduce_sum"(%469) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %471 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %472 = "tosa.reciprocal"(%471) : (tensor<1xf32>) -> tensor<1xf32>
    %473 = "tosa.mul"(%472, %470) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %474 = "tosa.sub"(%468, %473) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %475 = "tosa.mul"(%474, %474) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %476 = "tosa.reduce_sum"(%475) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %477 = "tosa.reduce_sum"(%476) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %478 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %479 = "tosa.reciprocal"(%478) : (tensor<1xf32>) -> tensor<1xf32>
    %480 = "tosa.mul"(%479, %477) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %481 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %482 = "tosa.add"(%480, %481) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %483 = "tosa.rsqrt"(%482) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %484 = "tosa.sub"(%468, %473) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %485 = "tosa.mul"(%484, %483) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %486 = "tosa.reshape"(%485) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %487 = "tosa.reshape"(%arg52) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %488 = "tosa.reshape"(%487) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %489 = "tosa.reshape"(%488) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %490 = "tosa.reshape"(%arg53) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %491 = "tosa.reshape"(%490) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %492 = "tosa.reshape"(%491) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %493 = "tosa.mul"(%486, %492) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %494 = "tosa.add"(%493, %489) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %495 = "tosa.sigmoid"(%494) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %496 = "tosa.mul"(%494, %495) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %497 = "tosa.identity"(%496) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %498 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %499 = "tosa.transpose"(%497, %498) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %500 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %501 = "tosa.transpose"(%arg54, %500) : (tensor<320x320x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x320xf32>
    %502 = "tosa.conv2d"(%499, %501, %arg55) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x320xf32>, tensor<320x3x3x320xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %503 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %504 = "tosa.transpose"(%502, %503) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %505 = "tosa.add"(%416, %504) : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %506 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x320x64x64xf32>}> : () -> tensor<1x320x64x64xf32>
    %507 = "tosa.reciprocal"(%506) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %508 = "tosa.mul"(%505, %507) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %509 = "tosa.reshape"(%508) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %510 = "tosa.reduce_sum"(%509) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %511 = "tosa.reduce_sum"(%510) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %512 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %513 = "tosa.reciprocal"(%512) : (tensor<1xf32>) -> tensor<1xf32>
    %514 = "tosa.mul"(%513, %511) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %515 = "tosa.sub"(%509, %514) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %516 = "tosa.mul"(%515, %515) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %517 = "tosa.reduce_sum"(%516) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %518 = "tosa.reduce_sum"(%517) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %519 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %520 = "tosa.reciprocal"(%519) : (tensor<1xf32>) -> tensor<1xf32>
    %521 = "tosa.mul"(%520, %518) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %522 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %523 = "tosa.add"(%521, %522) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %524 = "tosa.rsqrt"(%523) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %525 = "tosa.sub"(%509, %514) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %526 = "tosa.mul"(%525, %524) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %527 = "tosa.reshape"(%526) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %528 = "tosa.reshape"(%arg56) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %529 = "tosa.reshape"(%528) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %530 = "tosa.reshape"(%529) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %531 = "tosa.reshape"(%arg57) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %532 = "tosa.reshape"(%531) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %533 = "tosa.reshape"(%532) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %534 = "tosa.mul"(%527, %533) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %535 = "tosa.add"(%534, %530) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %536 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %537 = "tosa.transpose"(%535, %536) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %538 = "tosa.reshape"(%537) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x64x64x320xf32>) -> tensor<1x4096x320xf32>
    %539 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %540 = "tosa.transpose"(%arg58, %539) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %541 = "tosa.reshape"(%538) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %542 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %543 = "linalg.matmul"(%541, %540, %542) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %544 = "tosa.reshape"(%543) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %545 = "tosa.reshape"(%arg59) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %546 = "tosa.add"(%544, %545) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %547 = "tosa.reduce_sum"(%546) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %548 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %549 = "tosa.reciprocal"(%548) : (tensor<1xf32>) -> tensor<1xf32>
    %550 = "tosa.mul"(%549, %547) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %551 = "tosa.sub"(%546, %550) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %552 = "tosa.mul"(%551, %551) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %553 = "tosa.reduce_sum"(%552) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %554 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %555 = "tosa.reciprocal"(%554) : (tensor<1xf32>) -> tensor<1xf32>
    %556 = "tosa.mul"(%555, %553) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %557 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %558 = "tosa.add"(%556, %557) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %559 = "tosa.rsqrt"(%558) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %560 = "tosa.sub"(%546, %550) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %561 = "tosa.mul"(%560, %559) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %562 = "tosa.reshape"(%arg60) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %563 = "tosa.mul"(%561, %562) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %564 = "tosa.reshape"(%arg61) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %565 = "tosa.add"(%563, %564) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %566 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %567 = "tosa.transpose"(%arg62, %566) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %568 = "tosa.reshape"(%565) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %569 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %570 = "linalg.matmul"(%568, %567, %569) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %571 = "tosa.reshape"(%570) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %572 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %573 = "tosa.transpose"(%arg63, %572) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %574 = "tosa.reshape"(%565) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %575 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %576 = "linalg.matmul"(%574, %573, %575) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %577 = "tosa.reshape"(%576) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %578 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %579 = "tosa.transpose"(%arg64, %578) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %580 = "tosa.reshape"(%565) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %581 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %582 = "linalg.matmul"(%580, %579, %581) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %583 = "tosa.reshape"(%582) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %584 = "tosa.reshape"(%571) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %585 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %586 = "tosa.transpose"(%584, %585) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %587 = "tosa.reshape"(%577) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %588 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %589 = "tosa.transpose"(%587, %588) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %590 = "tosa.reshape"(%583) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %591 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %592 = "tosa.transpose"(%590, %591) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %593 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x4096xf32>}> : () -> tensor<4096x4096xf32>
    %594 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %595 = "tosa.transpose"(%589, %594) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x5x64x4096xf32>
    %596 = "tosa.reshape"(%586) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %597 = "tosa.reshape"(%595) <{new_shape = array<i64: 5, 64, 4096>}> : (tensor<1x5x64x4096xf32>) -> tensor<5x64x4096xf32>
    %598 = "tosa.matmul"(%596, %597) : (tensor<5x4096x64xf32>, tensor<5x64x4096xf32>) -> tensor<5x4096x4096xf32>
    %599 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<5x4096x4096xf32>}> : () -> tensor<5x4096x4096xf32>
    %600 = "tosa.mul"(%598, %599) <{shift = 0 : i8}> : (tensor<5x4096x4096xf32>, tensor<5x4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %601 = "tosa.add"(%600, %593) : (tensor<5x4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %602 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<5x4096x4096xf32>}> : () -> tensor<5x4096x4096xf32>
    %603 = "linalg.softmax"(%601, %602) <{dimension = 3 : i64}> : (tensor<5x4096x4096xf32>, tensor<5x4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %604 = "tosa.reshape"(%592) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %605 = "tosa.matmul"(%603, %604) : (tensor<5x4096x4096xf32>, tensor<5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %606 = "tosa.reshape"(%605) <{new_shape = array<i64: 1, 5, 4096, 64>}> : (tensor<5x4096x64xf32>) -> tensor<1x5x4096x64xf32>
    %607 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %608 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %609 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %610 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %611 = "tosa.transpose"(%606, %610) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x4096x5x64xf32>
    %612 = "tosa.reshape"(%611) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x4096x5x64xf32>) -> tensor<1x4096x320xf32>
    %613 = "tosa.reshape"(%612) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %614 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %615 = "tosa.transpose"(%arg65, %614) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %616 = "tosa.reshape"(%613) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %617 = "tosa.reshape"(%615) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %618 = "tosa.matmul"(%616, %617) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %619 = "tosa.reshape"(%618) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %620 = "tosa.reshape"(%arg66) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %621 = "tosa.add"(%620, %619) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %622 = "tosa.reshape"(%621) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %623 = "tosa.identity"(%622) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %624 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x320xf32>}> : () -> tensor<1x4096x320xf32>
    %625 = "tosa.reciprocal"(%624) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %626 = "tosa.mul"(%623, %625) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %627 = "tosa.add"(%626, %546) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %628 = "tosa.reduce_sum"(%627) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %629 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %630 = "tosa.reciprocal"(%629) : (tensor<1xf32>) -> tensor<1xf32>
    %631 = "tosa.mul"(%630, %628) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %632 = "tosa.sub"(%627, %631) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %633 = "tosa.mul"(%632, %632) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %634 = "tosa.reduce_sum"(%633) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %635 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %636 = "tosa.reciprocal"(%635) : (tensor<1xf32>) -> tensor<1xf32>
    %637 = "tosa.mul"(%636, %634) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %638 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %639 = "tosa.add"(%637, %638) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %640 = "tosa.rsqrt"(%639) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %641 = "tosa.sub"(%627, %631) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %642 = "tosa.mul"(%641, %640) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %643 = "tosa.reshape"(%arg67) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %644 = "tosa.mul"(%642, %643) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %645 = "tosa.reshape"(%arg68) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %646 = "tosa.add"(%644, %645) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %647 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %648 = "tosa.transpose"(%arg69, %647) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %649 = "tosa.reshape"(%646) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %650 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %651 = "linalg.matmul"(%649, %648, %650) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %652 = "tosa.reshape"(%651) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %653 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %654 = "tosa.transpose"(%arg70, %653) : (tensor<320x1024xf32>, tensor<2xi32>) -> tensor<1024x320xf32>
    %655 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %656 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x320xf32>}> : () -> tensor<12x320xf32>
    %657 = "linalg.matmul"(%655, %654, %656) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x320xf32>, tensor<12x320xf32>) -> tensor<12x320xf32>
    %658 = "tosa.reshape"(%657) <{new_shape = array<i64: 1, 12, 320>}> : (tensor<12x320xf32>) -> tensor<1x12x320xf32>
    %659 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %660 = "tosa.transpose"(%arg72, %659) : (tensor<320x1024xf32>, tensor<2xi32>) -> tensor<1024x320xf32>
    %661 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %662 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x320xf32>}> : () -> tensor<12x320xf32>
    %663 = "linalg.matmul"(%661, %660, %662) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x320xf32>, tensor<12x320xf32>) -> tensor<12x320xf32>
    %664 = "tosa.reshape"(%663) <{new_shape = array<i64: 1, 12, 320>}> : (tensor<12x320xf32>) -> tensor<1x12x320xf32>
    %665 = "tosa.reshape"(%652) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %666 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %667 = "tosa.transpose"(%665, %666) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %668 = "tosa.reshape"(%658) <{new_shape = array<i64: 1, 12, 5, 64>}> : (tensor<1x12x320xf32>) -> tensor<1x12x5x64xf32>
    %669 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %670 = "tosa.transpose"(%668, %669) : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %671 = "tosa.reshape"(%664) <{new_shape = array<i64: 1, 12, 5, 64>}> : (tensor<1x12x320xf32>) -> tensor<1x12x5x64xf32>
    %672 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %673 = "tosa.transpose"(%671, %672) : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %674 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x12xf32>}> : () -> tensor<4096x12xf32>
    %675 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %676 = "tosa.transpose"(%670, %675) : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x5x64x12xf32>
    %677 = "tosa.reshape"(%667) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %678 = "tosa.reshape"(%676) <{new_shape = array<i64: 5, 64, 12>}> : (tensor<1x5x64x12xf32>) -> tensor<5x64x12xf32>
    %679 = "tosa.matmul"(%677, %678) : (tensor<5x4096x64xf32>, tensor<5x64x12xf32>) -> tensor<5x4096x12xf32>
    %680 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<5x4096x12xf32>}> : () -> tensor<5x4096x12xf32>
    %681 = "tosa.mul"(%679, %680) <{shift = 0 : i8}> : (tensor<5x4096x12xf32>, tensor<5x4096x12xf32>) -> tensor<5x4096x12xf32>
    %682 = "tosa.add"(%681, %674) : (tensor<5x4096x12xf32>, tensor<4096x12xf32>) -> tensor<5x4096x12xf32>
    %683 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<5x4096x12xf32>}> : () -> tensor<5x4096x12xf32>
    %684 = "linalg.softmax"(%682, %683) <{dimension = 3 : i64}> : (tensor<5x4096x12xf32>, tensor<5x4096x12xf32>) -> tensor<5x4096x12xf32>
    %685 = "tosa.reshape"(%673) <{new_shape = array<i64: 5, 12, 64>}> : (tensor<1x5x12x64xf32>) -> tensor<5x12x64xf32>
    %686 = "tosa.matmul"(%684, %685) : (tensor<5x4096x12xf32>, tensor<5x12x64xf32>) -> tensor<5x4096x64xf32>
    %687 = "tosa.reshape"(%686) <{new_shape = array<i64: 1, 5, 4096, 64>}> : (tensor<5x4096x64xf32>) -> tensor<1x5x4096x64xf32>
    %688 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %689 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %690 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %691 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %692 = "tosa.transpose"(%687, %691) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x4096x5x64xf32>
    %693 = "tosa.reshape"(%692) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x4096x5x64xf32>) -> tensor<1x4096x320xf32>
    %694 = "tosa.reshape"(%693) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %695 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %696 = "tosa.transpose"(%arg74, %695) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %697 = "tosa.reshape"(%694) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %698 = "tosa.reshape"(%696) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %699 = "tosa.matmul"(%697, %698) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %700 = "tosa.reshape"(%699) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %701 = "tosa.reshape"(%arg75) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %702 = "tosa.add"(%701, %700) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %703 = "tosa.reshape"(%702) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %704 = "tosa.identity"(%703) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %705 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x320xf32>}> : () -> tensor<1x4096x320xf32>
    %706 = "tosa.reciprocal"(%705) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %707 = "tosa.mul"(%704, %706) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %708 = "tosa.add"(%707, %627) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %709 = "tosa.reduce_sum"(%708) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %710 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %711 = "tosa.reciprocal"(%710) : (tensor<1xf32>) -> tensor<1xf32>
    %712 = "tosa.mul"(%711, %709) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %713 = "tosa.sub"(%708, %712) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %714 = "tosa.mul"(%713, %713) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %715 = "tosa.reduce_sum"(%714) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %716 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %717 = "tosa.reciprocal"(%716) : (tensor<1xf32>) -> tensor<1xf32>
    %718 = "tosa.mul"(%717, %715) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %719 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %720 = "tosa.add"(%718, %719) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %721 = "tosa.rsqrt"(%720) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %722 = "tosa.sub"(%708, %712) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %723 = "tosa.mul"(%722, %721) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %724 = "tosa.reshape"(%arg76) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %725 = "tosa.mul"(%723, %724) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %726 = "tosa.reshape"(%arg77) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %727 = "tosa.add"(%725, %726) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %728 = "tosa.reshape"(%727) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %729 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %730 = "tosa.transpose"(%arg78, %729) : (tensor<2560x320xf32>, tensor<2xi32>) -> tensor<320x2560xf32>
    %731 = "tosa.reshape"(%728) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %732 = "tosa.reshape"(%730) <{new_shape = array<i64: 1, 320, 2560>}> : (tensor<320x2560xf32>) -> tensor<1x320x2560xf32>
    %733 = "tosa.matmul"(%731, %732) : (tensor<1x4096x320xf32>, tensor<1x320x2560xf32>) -> tensor<1x4096x2560xf32>
    %734 = "tosa.reshape"(%733) <{new_shape = array<i64: 4096, 2560>}> : (tensor<1x4096x2560xf32>) -> tensor<4096x2560xf32>
    %735 = "tosa.reshape"(%arg79) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %736 = "tosa.add"(%735, %734) : (tensor<1x2560xf32>, tensor<4096x2560xf32>) -> tensor<4096x2560xf32>
    %737 = "tosa.reshape"(%736) <{new_shape = array<i64: 1, 4096, 2560>}> : (tensor<4096x2560xf32>) -> tensor<1x4096x2560xf32>
    %738 = "tosa.slice"(%737) <{size = array<i64: 0, 0, 1280>, start = array<i64: 0, 0, 0>}> : (tensor<1x4096x2560xf32>) -> tensor<1x4096x1280xf32>
    %739 = "tosa.slice"(%737) <{size = array<i64: 0, 0, 2560>, start = array<i64: 0, 0, 1280>}> : (tensor<1x4096x2560xf32>) -> tensor<1x4096x1280xf32>
    %740 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %741 = "tosa.mul"(%739, %740) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %742 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %743 = "tosa.mul"(%739, %742) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %744 = "math.erf"(%743) <{fastmath = #arith.fastmath<none>}> : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %745 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %746 = "tosa.add"(%744, %745) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %747 = "tosa.mul"(%741, %746) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %748 = "tosa.mul"(%738, %747) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %749 = "tosa.identity"(%748) : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %750 = "tosa.reshape"(%749) <{new_shape = array<i64: 4096, 1280>}> : (tensor<1x4096x1280xf32>) -> tensor<4096x1280xf32>
    %751 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %752 = "tosa.transpose"(%arg80, %751) : (tensor<320x1280xf32>, tensor<2xi32>) -> tensor<1280x320xf32>
    %753 = "tosa.reshape"(%750) <{new_shape = array<i64: 1, 4096, 1280>}> : (tensor<4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %754 = "tosa.reshape"(%752) <{new_shape = array<i64: 1, 1280, 320>}> : (tensor<1280x320xf32>) -> tensor<1x1280x320xf32>
    %755 = "tosa.matmul"(%753, %754) : (tensor<1x4096x1280xf32>, tensor<1x1280x320xf32>) -> tensor<1x4096x320xf32>
    %756 = "tosa.reshape"(%755) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %757 = "tosa.reshape"(%arg81) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %758 = "tosa.add"(%757, %756) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %759 = "tosa.reshape"(%758) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %760 = "tosa.add"(%759, %708) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %761 = "tosa.reshape"(%760) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %762 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %763 = "tosa.transpose"(%arg82, %762) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %764 = "tosa.reshape"(%761) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %765 = "tosa.reshape"(%763) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %766 = "tosa.matmul"(%764, %765) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %767 = "tosa.reshape"(%766) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %768 = "tosa.reshape"(%arg83) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %769 = "tosa.add"(%768, %767) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %770 = "tosa.reshape"(%769) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %771 = "tosa.reshape"(%770) <{new_shape = array<i64: 1, 64, 64, 320>}> : (tensor<1x4096x320xf32>) -> tensor<1x64x64x320xf32>
    %772 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %773 = "tosa.transpose"(%771, %772) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %774 = "tosa.identity"(%773) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %775 = "tosa.add"(%774, %508) : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %776 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %777 = "tosa.transpose"(%775, %776) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %778 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %779 = "tosa.transpose"(%arg84, %778) : (tensor<320x320x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x320xf32>
    %780 = "tosa.conv2d"(%777, %779, %arg85) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<1x64x64x320xf32>, tensor<320x3x3x320xf32>, tensor<320xf32>) -> tensor<1x32x32x320xf32>
    %781 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %782 = "tosa.transpose"(%780, %781) : (tensor<1x32x32x320xf32>, tensor<4xi32>) -> tensor<1x320x32x32xf32>
    %783 = "tosa.reshape"(%782) <{new_shape = array<i64: 1, 32, 10, 1024>}> : (tensor<1x320x32x32xf32>) -> tensor<1x32x10x1024xf32>
    %784 = "tosa.reduce_sum"(%783) <{axis = 2 : i32}> : (tensor<1x32x10x1024xf32>) -> tensor<1x32x1x1024xf32>
    %785 = "tosa.reduce_sum"(%784) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %786 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %787 = "tosa.reciprocal"(%786) : (tensor<1xf32>) -> tensor<1xf32>
    %788 = "tosa.mul"(%787, %785) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %789 = "tosa.sub"(%783, %788) : (tensor<1x32x10x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x1024xf32>
    %790 = "tosa.mul"(%789, %789) <{shift = 0 : i8}> : (tensor<1x32x10x1024xf32>, tensor<1x32x10x1024xf32>) -> tensor<1x32x10x1024xf32>
    %791 = "tosa.reduce_sum"(%790) <{axis = 2 : i32}> : (tensor<1x32x10x1024xf32>) -> tensor<1x32x1x1024xf32>
    %792 = "tosa.reduce_sum"(%791) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %793 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %794 = "tosa.reciprocal"(%793) : (tensor<1xf32>) -> tensor<1xf32>
    %795 = "tosa.mul"(%794, %792) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %796 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %797 = "tosa.add"(%795, %796) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %798 = "tosa.rsqrt"(%797) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %799 = "tosa.sub"(%783, %788) : (tensor<1x32x10x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x1024xf32>
    %800 = "tosa.mul"(%799, %798) <{shift = 0 : i8}> : (tensor<1x32x10x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x1024xf32>
    %801 = "tosa.reshape"(%800) <{new_shape = array<i64: 1, 320, 32, 32>}> : (tensor<1x32x10x1024xf32>) -> tensor<1x320x32x32xf32>
    %802 = "tosa.reshape"(%arg86) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %803 = "tosa.reshape"(%802) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %804 = "tosa.reshape"(%803) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %805 = "tosa.reshape"(%arg87) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %806 = "tosa.reshape"(%805) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %807 = "tosa.reshape"(%806) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %808 = "tosa.mul"(%801, %807) <{shift = 0 : i8}> : (tensor<1x320x32x32xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x32x32xf32>
    %809 = "tosa.add"(%808, %804) : (tensor<1x320x32x32xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x32x32xf32>
    %810 = "tosa.sigmoid"(%809) : (tensor<1x320x32x32xf32>) -> tensor<1x320x32x32xf32>
    %811 = "tosa.mul"(%809, %810) <{shift = 0 : i8}> : (tensor<1x320x32x32xf32>, tensor<1x320x32x32xf32>) -> tensor<1x320x32x32xf32>
    %812 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %813 = "tosa.transpose"(%811, %812) : (tensor<1x320x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x320xf32>
    %814 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %815 = "tosa.transpose"(%arg88, %814) : (tensor<640x320x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x320xf32>
    %816 = "tosa.conv2d"(%813, %815, %arg89) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x320xf32>, tensor<640x3x3x320xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %817 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %818 = "tosa.transpose"(%816, %817) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %819 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %820 = "tosa.mul"(%50, %819) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %821 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %822 = "tosa.transpose"(%arg90, %821) : (tensor<640x1280xf32>, tensor<2xi32>) -> tensor<1280x640xf32>
    %823 = "tosa.reshape"(%820) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %824 = "tosa.reshape"(%822) <{new_shape = array<i64: 1, 1280, 640>}> : (tensor<1280x640xf32>) -> tensor<1x1280x640xf32>
    %825 = "tosa.matmul"(%823, %824) : (tensor<1x1x1280xf32>, tensor<1x1280x640xf32>) -> tensor<1x1x640xf32>
    %826 = "tosa.reshape"(%825) <{new_shape = array<i64: 1, 640>}> : (tensor<1x1x640xf32>) -> tensor<1x640xf32>
    %827 = "tosa.reshape"(%arg91) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %828 = "tosa.add"(%827, %826) : (tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %829 = "tensor.extract_slice"(%828) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 640>, static_strides = array<i64: 1, 1>}> : (tensor<1x640xf32>) -> tensor<1x640xf32>
    %830 = "tensor.extract_slice"(%829) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 640>, static_strides = array<i64: 1, 1>}> : (tensor<1x640xf32>) -> tensor<1x640xf32>
    %831 = "tosa.reshape"(%830) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %832 = "tosa.reshape"(%831) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %833 = "tosa.add"(%818, %832) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %834 = "tosa.reshape"(%833) <{new_shape = array<i64: 1, 32, 20, 1024>}> : (tensor<1x640x32x32xf32>) -> tensor<1x32x20x1024xf32>
    %835 = "tosa.reduce_sum"(%834) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %836 = "tosa.reduce_sum"(%835) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %837 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %838 = "tosa.reciprocal"(%837) : (tensor<1xf32>) -> tensor<1xf32>
    %839 = "tosa.mul"(%838, %836) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %840 = "tosa.sub"(%834, %839) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %841 = "tosa.mul"(%840, %840) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x20x1024xf32>) -> tensor<1x32x20x1024xf32>
    %842 = "tosa.reduce_sum"(%841) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %843 = "tosa.reduce_sum"(%842) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %844 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %845 = "tosa.reciprocal"(%844) : (tensor<1xf32>) -> tensor<1xf32>
    %846 = "tosa.mul"(%845, %843) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %847 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %848 = "tosa.add"(%846, %847) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %849 = "tosa.rsqrt"(%848) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %850 = "tosa.sub"(%834, %839) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %851 = "tosa.mul"(%850, %849) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %852 = "tosa.reshape"(%851) <{new_shape = array<i64: 1, 640, 32, 32>}> : (tensor<1x32x20x1024xf32>) -> tensor<1x640x32x32xf32>
    %853 = "tosa.reshape"(%arg92) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %854 = "tosa.reshape"(%853) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %855 = "tosa.reshape"(%854) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %856 = "tosa.reshape"(%arg93) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %857 = "tosa.reshape"(%856) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %858 = "tosa.reshape"(%857) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %859 = "tosa.mul"(%852, %858) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %860 = "tosa.add"(%859, %855) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %861 = "tosa.sigmoid"(%860) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %862 = "tosa.mul"(%860, %861) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %863 = "tosa.identity"(%862) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %864 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %865 = "tosa.transpose"(%863, %864) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %866 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %867 = "tosa.transpose"(%arg94, %866) : (tensor<640x640x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x640xf32>
    %868 = "tosa.conv2d"(%865, %867, %arg95) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x640xf32>, tensor<640x3x3x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %869 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %870 = "tosa.transpose"(%868, %869) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %871 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %872 = "tosa.transpose"(%782, %871) : (tensor<1x320x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x320xf32>
    %873 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %874 = "tosa.transpose"(%arg96, %873) : (tensor<640x320x1x1xf32>, tensor<4xi32>) -> tensor<640x1x1x320xf32>
    %875 = "tosa.conv2d"(%872, %874, %arg97) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x320xf32>, tensor<640x1x1x320xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %876 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %877 = "tosa.transpose"(%875, %876) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %878 = "tosa.add"(%877, %870) : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %879 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x640x32x32xf32>}> : () -> tensor<1x640x32x32xf32>
    %880 = "tosa.reciprocal"(%879) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %881 = "tosa.mul"(%878, %880) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %882 = "tosa.reshape"(%881) <{new_shape = array<i64: 1, 32, 20, 1024>}> : (tensor<1x640x32x32xf32>) -> tensor<1x32x20x1024xf32>
    %883 = "tosa.reduce_sum"(%882) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %884 = "tosa.reduce_sum"(%883) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %885 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %886 = "tosa.reciprocal"(%885) : (tensor<1xf32>) -> tensor<1xf32>
    %887 = "tosa.mul"(%886, %884) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %888 = "tosa.sub"(%882, %887) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %889 = "tosa.mul"(%888, %888) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x20x1024xf32>) -> tensor<1x32x20x1024xf32>
    %890 = "tosa.reduce_sum"(%889) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %891 = "tosa.reduce_sum"(%890) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %892 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %893 = "tosa.reciprocal"(%892) : (tensor<1xf32>) -> tensor<1xf32>
    %894 = "tosa.mul"(%893, %891) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %895 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %896 = "tosa.add"(%894, %895) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %897 = "tosa.rsqrt"(%896) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %898 = "tosa.sub"(%882, %887) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %899 = "tosa.mul"(%898, %897) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %900 = "tosa.reshape"(%899) <{new_shape = array<i64: 1, 640, 32, 32>}> : (tensor<1x32x20x1024xf32>) -> tensor<1x640x32x32xf32>
    %901 = "tosa.reshape"(%arg98) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %902 = "tosa.reshape"(%901) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %903 = "tosa.reshape"(%902) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %904 = "tosa.reshape"(%arg99) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %905 = "tosa.reshape"(%904) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %906 = "tosa.reshape"(%905) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %907 = "tosa.mul"(%900, %906) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %908 = "tosa.add"(%907, %903) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %909 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %910 = "tosa.transpose"(%908, %909) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %911 = "tosa.reshape"(%910) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x32x32x640xf32>) -> tensor<1x1024x640xf32>
    %912 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %913 = "tosa.transpose"(%arg100, %912) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %914 = "tosa.reshape"(%911) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %915 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %916 = "linalg.matmul"(%914, %913, %915) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %917 = "tosa.reshape"(%916) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %918 = "tosa.reshape"(%arg101) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %919 = "tosa.add"(%917, %918) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %920 = "tosa.reduce_sum"(%919) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %921 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %922 = "tosa.reciprocal"(%921) : (tensor<1xf32>) -> tensor<1xf32>
    %923 = "tosa.mul"(%922, %920) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %924 = "tosa.sub"(%919, %923) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %925 = "tosa.mul"(%924, %924) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %926 = "tosa.reduce_sum"(%925) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %927 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %928 = "tosa.reciprocal"(%927) : (tensor<1xf32>) -> tensor<1xf32>
    %929 = "tosa.mul"(%928, %926) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %930 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %931 = "tosa.add"(%929, %930) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %932 = "tosa.rsqrt"(%931) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %933 = "tosa.sub"(%919, %923) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %934 = "tosa.mul"(%933, %932) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %935 = "tosa.reshape"(%arg102) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %936 = "tosa.mul"(%934, %935) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %937 = "tosa.reshape"(%arg103) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %938 = "tosa.add"(%936, %937) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %939 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %940 = "tosa.transpose"(%arg104, %939) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %941 = "tosa.reshape"(%938) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %942 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %943 = "linalg.matmul"(%941, %940, %942) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %944 = "tosa.reshape"(%943) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %945 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %946 = "tosa.transpose"(%arg105, %945) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %947 = "tosa.reshape"(%938) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %948 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %949 = "linalg.matmul"(%947, %946, %948) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %950 = "tosa.reshape"(%949) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %951 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %952 = "tosa.transpose"(%arg106, %951) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %953 = "tosa.reshape"(%938) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %954 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %955 = "linalg.matmul"(%953, %952, %954) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %956 = "tosa.reshape"(%955) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %957 = "tosa.reshape"(%944) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %958 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %959 = "tosa.transpose"(%957, %958) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %960 = "tosa.reshape"(%950) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %961 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %962 = "tosa.transpose"(%960, %961) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %963 = "tosa.reshape"(%956) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %964 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %965 = "tosa.transpose"(%963, %964) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %966 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x1024xf32>}> : () -> tensor<1024x1024xf32>
    %967 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %968 = "tosa.transpose"(%962, %967) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x10x64x1024xf32>
    %969 = "tosa.reshape"(%959) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %970 = "tosa.reshape"(%968) <{new_shape = array<i64: 10, 64, 1024>}> : (tensor<1x10x64x1024xf32>) -> tensor<10x64x1024xf32>
    %971 = "tosa.matmul"(%969, %970) : (tensor<10x1024x64xf32>, tensor<10x64x1024xf32>) -> tensor<10x1024x1024xf32>
    %972 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<10x1024x1024xf32>}> : () -> tensor<10x1024x1024xf32>
    %973 = "tosa.mul"(%971, %972) <{shift = 0 : i8}> : (tensor<10x1024x1024xf32>, tensor<10x1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %974 = "tosa.add"(%973, %966) : (tensor<10x1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %975 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<10x1024x1024xf32>}> : () -> tensor<10x1024x1024xf32>
    %976 = "linalg.softmax"(%974, %975) <{dimension = 3 : i64}> : (tensor<10x1024x1024xf32>, tensor<10x1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %977 = "tosa.reshape"(%965) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %978 = "tosa.matmul"(%976, %977) : (tensor<10x1024x1024xf32>, tensor<10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %979 = "tosa.reshape"(%978) <{new_shape = array<i64: 1, 10, 1024, 64>}> : (tensor<10x1024x64xf32>) -> tensor<1x10x1024x64xf32>
    %980 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %981 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %982 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %983 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %984 = "tosa.transpose"(%979, %983) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x1024x10x64xf32>
    %985 = "tosa.reshape"(%984) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x1024x10x64xf32>) -> tensor<1x1024x640xf32>
    %986 = "tosa.reshape"(%985) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %987 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %988 = "tosa.transpose"(%arg107, %987) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %989 = "tosa.reshape"(%986) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %990 = "tosa.reshape"(%988) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %991 = "tosa.matmul"(%989, %990) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %992 = "tosa.reshape"(%991) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %993 = "tosa.reshape"(%arg108) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %994 = "tosa.add"(%993, %992) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %995 = "tosa.reshape"(%994) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %996 = "tosa.identity"(%995) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %997 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x640xf32>}> : () -> tensor<1x1024x640xf32>
    %998 = "tosa.reciprocal"(%997) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %999 = "tosa.mul"(%996, %998) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1000 = "tosa.add"(%999, %919) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1001 = "tosa.reduce_sum"(%1000) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %1002 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1003 = "tosa.reciprocal"(%1002) : (tensor<1xf32>) -> tensor<1xf32>
    %1004 = "tosa.mul"(%1003, %1001) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1005 = "tosa.sub"(%1000, %1004) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1006 = "tosa.mul"(%1005, %1005) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1007 = "tosa.reduce_sum"(%1006) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %1008 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1009 = "tosa.reciprocal"(%1008) : (tensor<1xf32>) -> tensor<1xf32>
    %1010 = "tosa.mul"(%1009, %1007) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1011 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %1012 = "tosa.add"(%1010, %1011) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1013 = "tosa.rsqrt"(%1012) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1014 = "tosa.sub"(%1000, %1004) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1015 = "tosa.mul"(%1014, %1013) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1016 = "tosa.reshape"(%arg109) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %1017 = "tosa.mul"(%1015, %1016) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %1018 = "tosa.reshape"(%arg110) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %1019 = "tosa.add"(%1017, %1018) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %1020 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1021 = "tosa.transpose"(%arg111, %1020) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %1022 = "tosa.reshape"(%1019) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1023 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %1024 = "linalg.matmul"(%1022, %1021, %1023) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1025 = "tosa.reshape"(%1024) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1026 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1027 = "tosa.transpose"(%arg112, %1026) : (tensor<640x1024xf32>, tensor<2xi32>) -> tensor<1024x640xf32>
    %1028 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1029 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x640xf32>}> : () -> tensor<12x640xf32>
    %1030 = "linalg.matmul"(%1028, %1027, %1029) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x640xf32>, tensor<12x640xf32>) -> tensor<12x640xf32>
    %1031 = "tosa.reshape"(%1030) <{new_shape = array<i64: 1, 12, 640>}> : (tensor<12x640xf32>) -> tensor<1x12x640xf32>
    %1032 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1033 = "tosa.transpose"(%arg114, %1032) : (tensor<640x1024xf32>, tensor<2xi32>) -> tensor<1024x640xf32>
    %1034 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1035 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x640xf32>}> : () -> tensor<12x640xf32>
    %1036 = "linalg.matmul"(%1034, %1033, %1035) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x640xf32>, tensor<12x640xf32>) -> tensor<12x640xf32>
    %1037 = "tosa.reshape"(%1036) <{new_shape = array<i64: 1, 12, 640>}> : (tensor<12x640xf32>) -> tensor<1x12x640xf32>
    %1038 = "tosa.reshape"(%1025) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %1039 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1040 = "tosa.transpose"(%1038, %1039) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %1041 = "tosa.reshape"(%1031) <{new_shape = array<i64: 1, 12, 10, 64>}> : (tensor<1x12x640xf32>) -> tensor<1x12x10x64xf32>
    %1042 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1043 = "tosa.transpose"(%1041, %1042) : (tensor<1x12x10x64xf32>, tensor<4xi32>) -> tensor<1x10x12x64xf32>
    %1044 = "tosa.reshape"(%1037) <{new_shape = array<i64: 1, 12, 10, 64>}> : (tensor<1x12x640xf32>) -> tensor<1x12x10x64xf32>
    %1045 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1046 = "tosa.transpose"(%1044, %1045) : (tensor<1x12x10x64xf32>, tensor<4xi32>) -> tensor<1x10x12x64xf32>
    %1047 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x12xf32>}> : () -> tensor<1024x12xf32>
    %1048 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1049 = "tosa.transpose"(%1043, %1048) : (tensor<1x10x12x64xf32>, tensor<4xi32>) -> tensor<1x10x64x12xf32>
    %1050 = "tosa.reshape"(%1040) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %1051 = "tosa.reshape"(%1049) <{new_shape = array<i64: 10, 64, 12>}> : (tensor<1x10x64x12xf32>) -> tensor<10x64x12xf32>
    %1052 = "tosa.matmul"(%1050, %1051) : (tensor<10x1024x64xf32>, tensor<10x64x12xf32>) -> tensor<10x1024x12xf32>
    %1053 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<10x1024x12xf32>}> : () -> tensor<10x1024x12xf32>
    %1054 = "tosa.mul"(%1052, %1053) <{shift = 0 : i8}> : (tensor<10x1024x12xf32>, tensor<10x1024x12xf32>) -> tensor<10x1024x12xf32>
    %1055 = "tosa.add"(%1054, %1047) : (tensor<10x1024x12xf32>, tensor<1024x12xf32>) -> tensor<10x1024x12xf32>
    %1056 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<10x1024x12xf32>}> : () -> tensor<10x1024x12xf32>
    %1057 = "linalg.softmax"(%1055, %1056) <{dimension = 3 : i64}> : (tensor<10x1024x12xf32>, tensor<10x1024x12xf32>) -> tensor<10x1024x12xf32>
    %1058 = "tosa.reshape"(%1046) <{new_shape = array<i64: 10, 12, 64>}> : (tensor<1x10x12x64xf32>) -> tensor<10x12x64xf32>
    %1059 = "tosa.matmul"(%1057, %1058) : (tensor<10x1024x12xf32>, tensor<10x12x64xf32>) -> tensor<10x1024x64xf32>
    %1060 = "tosa.reshape"(%1059) <{new_shape = array<i64: 1, 10, 1024, 64>}> : (tensor<10x1024x64xf32>) -> tensor<1x10x1024x64xf32>
    %1061 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1062 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1063 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1064 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1065 = "tosa.transpose"(%1060, %1064) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x1024x10x64xf32>
    %1066 = "tosa.reshape"(%1065) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x1024x10x64xf32>) -> tensor<1x1024x640xf32>
    %1067 = "tosa.reshape"(%1066) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1068 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1069 = "tosa.transpose"(%arg116, %1068) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %1070 = "tosa.reshape"(%1067) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1071 = "tosa.reshape"(%1069) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %1072 = "tosa.matmul"(%1070, %1071) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %1073 = "tosa.reshape"(%1072) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1074 = "tosa.reshape"(%arg117) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1075 = "tosa.add"(%1074, %1073) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1076 = "tosa.reshape"(%1075) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1077 = "tosa.identity"(%1076) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1078 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x640xf32>}> : () -> tensor<1x1024x640xf32>
    %1079 = "tosa.reciprocal"(%1078) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1080 = "tosa.mul"(%1077, %1079) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1081 = "tosa.add"(%1080, %1000) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1082 = "tosa.reduce_sum"(%1081) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %1083 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1084 = "tosa.reciprocal"(%1083) : (tensor<1xf32>) -> tensor<1xf32>
    %1085 = "tosa.mul"(%1084, %1082) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1086 = "tosa.sub"(%1081, %1085) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1087 = "tosa.mul"(%1086, %1086) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1088 = "tosa.reduce_sum"(%1087) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %1089 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1090 = "tosa.reciprocal"(%1089) : (tensor<1xf32>) -> tensor<1xf32>
    %1091 = "tosa.mul"(%1090, %1088) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1092 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %1093 = "tosa.add"(%1091, %1092) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1094 = "tosa.rsqrt"(%1093) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1095 = "tosa.sub"(%1081, %1085) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1096 = "tosa.mul"(%1095, %1094) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1097 = "tosa.reshape"(%arg118) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %1098 = "tosa.mul"(%1096, %1097) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %1099 = "tosa.reshape"(%arg119) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %1100 = "tosa.add"(%1098, %1099) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %1101 = "tosa.reshape"(%1100) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1102 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1103 = "tosa.transpose"(%arg120, %1102) : (tensor<5120x640xf32>, tensor<2xi32>) -> tensor<640x5120xf32>
    %1104 = "tosa.reshape"(%1101) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1105 = "tosa.reshape"(%1103) <{new_shape = array<i64: 1, 640, 5120>}> : (tensor<640x5120xf32>) -> tensor<1x640x5120xf32>
    %1106 = "tosa.matmul"(%1104, %1105) : (tensor<1x1024x640xf32>, tensor<1x640x5120xf32>) -> tensor<1x1024x5120xf32>
    %1107 = "tosa.reshape"(%1106) <{new_shape = array<i64: 1024, 5120>}> : (tensor<1x1024x5120xf32>) -> tensor<1024x5120xf32>
    %1108 = "tosa.reshape"(%arg121) <{new_shape = array<i64: 1, 5120>}> : (tensor<5120xf32>) -> tensor<1x5120xf32>
    %1109 = "tosa.add"(%1108, %1107) : (tensor<1x5120xf32>, tensor<1024x5120xf32>) -> tensor<1024x5120xf32>
    %1110 = "tosa.reshape"(%1109) <{new_shape = array<i64: 1, 1024, 5120>}> : (tensor<1024x5120xf32>) -> tensor<1x1024x5120xf32>
    %1111 = "tosa.slice"(%1110) <{size = array<i64: 0, 0, 2560>, start = array<i64: 0, 0, 0>}> : (tensor<1x1024x5120xf32>) -> tensor<1x1024x2560xf32>
    %1112 = "tosa.slice"(%1110) <{size = array<i64: 0, 0, 5120>, start = array<i64: 0, 0, 2560>}> : (tensor<1x1024x5120xf32>) -> tensor<1x1024x2560xf32>
    %1113 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %1114 = "tosa.mul"(%1112, %1113) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1115 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %1116 = "tosa.mul"(%1112, %1115) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1117 = "math.erf"(%1116) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1118 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %1119 = "tosa.add"(%1117, %1118) : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1120 = "tosa.mul"(%1114, %1119) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1121 = "tosa.mul"(%1111, %1120) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1122 = "tosa.identity"(%1121) : (tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1123 = "tosa.reshape"(%1122) <{new_shape = array<i64: 1024, 2560>}> : (tensor<1x1024x2560xf32>) -> tensor<1024x2560xf32>
    %1124 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1125 = "tosa.transpose"(%arg122, %1124) : (tensor<640x2560xf32>, tensor<2xi32>) -> tensor<2560x640xf32>
    %1126 = "tosa.reshape"(%1123) <{new_shape = array<i64: 1, 1024, 2560>}> : (tensor<1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1127 = "tosa.reshape"(%1125) <{new_shape = array<i64: 1, 2560, 640>}> : (tensor<2560x640xf32>) -> tensor<1x2560x640xf32>
    %1128 = "tosa.matmul"(%1126, %1127) : (tensor<1x1024x2560xf32>, tensor<1x2560x640xf32>) -> tensor<1x1024x640xf32>
    %1129 = "tosa.reshape"(%1128) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1130 = "tosa.reshape"(%arg123) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1131 = "tosa.add"(%1130, %1129) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1132 = "tosa.reshape"(%1131) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1133 = "tosa.add"(%1132, %1081) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1134 = "tosa.reshape"(%1133) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1135 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1136 = "tosa.transpose"(%arg124, %1135) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %1137 = "tosa.reshape"(%1134) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1138 = "tosa.reshape"(%1136) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %1139 = "tosa.matmul"(%1137, %1138) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %1140 = "tosa.reshape"(%1139) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1141 = "tosa.reshape"(%arg125) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1142 = "tosa.add"(%1141, %1140) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1143 = "tosa.reshape"(%1142) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1144 = "tosa.reshape"(%1143) <{new_shape = array<i64: 1, 32, 32, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1x32x32x640xf32>
    %1145 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1146 = "tosa.transpose"(%1144, %1145) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %1147 = "tosa.identity"(%1146) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1148 = "tosa.add"(%1147, %881) : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1149 = "tosa.reshape"(%1148) <{new_shape = array<i64: 1, 32, 20, 1024>}> : (tensor<1x640x32x32xf32>) -> tensor<1x32x20x1024xf32>
    %1150 = "tosa.reduce_sum"(%1149) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %1151 = "tosa.reduce_sum"(%1150) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %1152 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1153 = "tosa.reciprocal"(%1152) : (tensor<1xf32>) -> tensor<1xf32>
    %1154 = "tosa.mul"(%1153, %1151) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1155 = "tosa.sub"(%1149, %1154) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %1156 = "tosa.mul"(%1155, %1155) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x20x1024xf32>) -> tensor<1x32x20x1024xf32>
    %1157 = "tosa.reduce_sum"(%1156) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %1158 = "tosa.reduce_sum"(%1157) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %1159 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1160 = "tosa.reciprocal"(%1159) : (tensor<1xf32>) -> tensor<1xf32>
    %1161 = "tosa.mul"(%1160, %1158) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1162 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1163 = "tosa.add"(%1161, %1162) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1164 = "tosa.rsqrt"(%1163) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1165 = "tosa.sub"(%1149, %1154) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %1166 = "tosa.mul"(%1165, %1164) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %1167 = "tosa.reshape"(%1166) <{new_shape = array<i64: 1, 640, 32, 32>}> : (tensor<1x32x20x1024xf32>) -> tensor<1x640x32x32xf32>
    %1168 = "tosa.reshape"(%arg126) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1169 = "tosa.reshape"(%1168) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %1170 = "tosa.reshape"(%1169) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %1171 = "tosa.reshape"(%arg127) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1172 = "tosa.reshape"(%1171) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %1173 = "tosa.reshape"(%1172) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %1174 = "tosa.mul"(%1167, %1173) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %1175 = "tosa.add"(%1174, %1170) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %1176 = "tosa.sigmoid"(%1175) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1177 = "tosa.mul"(%1175, %1176) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1178 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1179 = "tosa.transpose"(%1177, %1178) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %1180 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1181 = "tosa.transpose"(%arg128, %1180) : (tensor<640x640x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x640xf32>
    %1182 = "tosa.conv2d"(%1179, %1181, %arg129) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x640xf32>, tensor<640x3x3x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %1183 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1184 = "tosa.transpose"(%1182, %1183) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %1185 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1186 = "tosa.mul"(%50, %1185) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1187 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1188 = "tosa.transpose"(%arg130, %1187) : (tensor<640x1280xf32>, tensor<2xi32>) -> tensor<1280x640xf32>
    %1189 = "tosa.reshape"(%1186) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %1190 = "tosa.reshape"(%1188) <{new_shape = array<i64: 1, 1280, 640>}> : (tensor<1280x640xf32>) -> tensor<1x1280x640xf32>
    %1191 = "tosa.matmul"(%1189, %1190) : (tensor<1x1x1280xf32>, tensor<1x1280x640xf32>) -> tensor<1x1x640xf32>
    %1192 = "tosa.reshape"(%1191) <{new_shape = array<i64: 1, 640>}> : (tensor<1x1x640xf32>) -> tensor<1x640xf32>
    %1193 = "tosa.reshape"(%arg131) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1194 = "tosa.add"(%1193, %1192) : (tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %1195 = "tensor.extract_slice"(%1194) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 640>, static_strides = array<i64: 1, 1>}> : (tensor<1x640xf32>) -> tensor<1x640xf32>
    %1196 = "tensor.extract_slice"(%1195) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 640>, static_strides = array<i64: 1, 1>}> : (tensor<1x640xf32>) -> tensor<1x640xf32>
    %1197 = "tosa.reshape"(%1196) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %1198 = "tosa.reshape"(%1197) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %1199 = "tosa.add"(%1184, %1198) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %1200 = "tosa.reshape"(%1199) <{new_shape = array<i64: 1, 32, 20, 1024>}> : (tensor<1x640x32x32xf32>) -> tensor<1x32x20x1024xf32>
    %1201 = "tosa.reduce_sum"(%1200) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %1202 = "tosa.reduce_sum"(%1201) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %1203 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1204 = "tosa.reciprocal"(%1203) : (tensor<1xf32>) -> tensor<1xf32>
    %1205 = "tosa.mul"(%1204, %1202) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1206 = "tosa.sub"(%1200, %1205) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %1207 = "tosa.mul"(%1206, %1206) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x20x1024xf32>) -> tensor<1x32x20x1024xf32>
    %1208 = "tosa.reduce_sum"(%1207) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %1209 = "tosa.reduce_sum"(%1208) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %1210 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1211 = "tosa.reciprocal"(%1210) : (tensor<1xf32>) -> tensor<1xf32>
    %1212 = "tosa.mul"(%1211, %1209) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1213 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1214 = "tosa.add"(%1212, %1213) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1215 = "tosa.rsqrt"(%1214) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1216 = "tosa.sub"(%1200, %1205) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %1217 = "tosa.mul"(%1216, %1215) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %1218 = "tosa.reshape"(%1217) <{new_shape = array<i64: 1, 640, 32, 32>}> : (tensor<1x32x20x1024xf32>) -> tensor<1x640x32x32xf32>
    %1219 = "tosa.reshape"(%arg132) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1220 = "tosa.reshape"(%1219) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %1221 = "tosa.reshape"(%1220) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %1222 = "tosa.reshape"(%arg133) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1223 = "tosa.reshape"(%1222) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %1224 = "tosa.reshape"(%1223) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %1225 = "tosa.mul"(%1218, %1224) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %1226 = "tosa.add"(%1225, %1221) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %1227 = "tosa.sigmoid"(%1226) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1228 = "tosa.mul"(%1226, %1227) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1229 = "tosa.identity"(%1228) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1230 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1231 = "tosa.transpose"(%1229, %1230) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %1232 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1233 = "tosa.transpose"(%arg134, %1232) : (tensor<640x640x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x640xf32>
    %1234 = "tosa.conv2d"(%1231, %1233, %arg135) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x640xf32>, tensor<640x3x3x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %1235 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1236 = "tosa.transpose"(%1234, %1235) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %1237 = "tosa.add"(%1148, %1236) : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1238 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x640x32x32xf32>}> : () -> tensor<1x640x32x32xf32>
    %1239 = "tosa.reciprocal"(%1238) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1240 = "tosa.mul"(%1237, %1239) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1241 = "tosa.reshape"(%1240) <{new_shape = array<i64: 1, 32, 20, 1024>}> : (tensor<1x640x32x32xf32>) -> tensor<1x32x20x1024xf32>
    %1242 = "tosa.reduce_sum"(%1241) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %1243 = "tosa.reduce_sum"(%1242) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %1244 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1245 = "tosa.reciprocal"(%1244) : (tensor<1xf32>) -> tensor<1xf32>
    %1246 = "tosa.mul"(%1245, %1243) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1247 = "tosa.sub"(%1241, %1246) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %1248 = "tosa.mul"(%1247, %1247) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x20x1024xf32>) -> tensor<1x32x20x1024xf32>
    %1249 = "tosa.reduce_sum"(%1248) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %1250 = "tosa.reduce_sum"(%1249) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %1251 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1252 = "tosa.reciprocal"(%1251) : (tensor<1xf32>) -> tensor<1xf32>
    %1253 = "tosa.mul"(%1252, %1250) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1254 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1255 = "tosa.add"(%1253, %1254) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1256 = "tosa.rsqrt"(%1255) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1257 = "tosa.sub"(%1241, %1246) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %1258 = "tosa.mul"(%1257, %1256) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %1259 = "tosa.reshape"(%1258) <{new_shape = array<i64: 1, 640, 32, 32>}> : (tensor<1x32x20x1024xf32>) -> tensor<1x640x32x32xf32>
    %1260 = "tosa.reshape"(%arg136) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1261 = "tosa.reshape"(%1260) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %1262 = "tosa.reshape"(%1261) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %1263 = "tosa.reshape"(%arg137) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1264 = "tosa.reshape"(%1263) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %1265 = "tosa.reshape"(%1264) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %1266 = "tosa.mul"(%1259, %1265) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %1267 = "tosa.add"(%1266, %1262) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %1268 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1269 = "tosa.transpose"(%1267, %1268) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %1270 = "tosa.reshape"(%1269) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x32x32x640xf32>) -> tensor<1x1024x640xf32>
    %1271 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1272 = "tosa.transpose"(%arg138, %1271) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %1273 = "tosa.reshape"(%1270) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1274 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %1275 = "linalg.matmul"(%1273, %1272, %1274) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1276 = "tosa.reshape"(%1275) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1277 = "tosa.reshape"(%arg139) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %1278 = "tosa.add"(%1276, %1277) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %1279 = "tosa.reduce_sum"(%1278) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %1280 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1281 = "tosa.reciprocal"(%1280) : (tensor<1xf32>) -> tensor<1xf32>
    %1282 = "tosa.mul"(%1281, %1279) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1283 = "tosa.sub"(%1278, %1282) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1284 = "tosa.mul"(%1283, %1283) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1285 = "tosa.reduce_sum"(%1284) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %1286 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1287 = "tosa.reciprocal"(%1286) : (tensor<1xf32>) -> tensor<1xf32>
    %1288 = "tosa.mul"(%1287, %1285) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1289 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %1290 = "tosa.add"(%1288, %1289) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1291 = "tosa.rsqrt"(%1290) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1292 = "tosa.sub"(%1278, %1282) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1293 = "tosa.mul"(%1292, %1291) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1294 = "tosa.reshape"(%arg140) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %1295 = "tosa.mul"(%1293, %1294) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %1296 = "tosa.reshape"(%arg141) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %1297 = "tosa.add"(%1295, %1296) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %1298 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1299 = "tosa.transpose"(%arg142, %1298) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %1300 = "tosa.reshape"(%1297) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1301 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %1302 = "linalg.matmul"(%1300, %1299, %1301) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1303 = "tosa.reshape"(%1302) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1304 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1305 = "tosa.transpose"(%arg143, %1304) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %1306 = "tosa.reshape"(%1297) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1307 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %1308 = "linalg.matmul"(%1306, %1305, %1307) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1309 = "tosa.reshape"(%1308) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1310 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1311 = "tosa.transpose"(%arg144, %1310) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %1312 = "tosa.reshape"(%1297) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1313 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %1314 = "linalg.matmul"(%1312, %1311, %1313) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1315 = "tosa.reshape"(%1314) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1316 = "tosa.reshape"(%1303) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %1317 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1318 = "tosa.transpose"(%1316, %1317) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %1319 = "tosa.reshape"(%1309) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %1320 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1321 = "tosa.transpose"(%1319, %1320) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %1322 = "tosa.reshape"(%1315) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %1323 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1324 = "tosa.transpose"(%1322, %1323) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %1325 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x1024xf32>}> : () -> tensor<1024x1024xf32>
    %1326 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1327 = "tosa.transpose"(%1321, %1326) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x10x64x1024xf32>
    %1328 = "tosa.reshape"(%1318) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %1329 = "tosa.reshape"(%1327) <{new_shape = array<i64: 10, 64, 1024>}> : (tensor<1x10x64x1024xf32>) -> tensor<10x64x1024xf32>
    %1330 = "tosa.matmul"(%1328, %1329) : (tensor<10x1024x64xf32>, tensor<10x64x1024xf32>) -> tensor<10x1024x1024xf32>
    %1331 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<10x1024x1024xf32>}> : () -> tensor<10x1024x1024xf32>
    %1332 = "tosa.mul"(%1330, %1331) <{shift = 0 : i8}> : (tensor<10x1024x1024xf32>, tensor<10x1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %1333 = "tosa.add"(%1332, %1325) : (tensor<10x1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %1334 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<10x1024x1024xf32>}> : () -> tensor<10x1024x1024xf32>
    %1335 = "linalg.softmax"(%1333, %1334) <{dimension = 3 : i64}> : (tensor<10x1024x1024xf32>, tensor<10x1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %1336 = "tosa.reshape"(%1324) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %1337 = "tosa.matmul"(%1335, %1336) : (tensor<10x1024x1024xf32>, tensor<10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %1338 = "tosa.reshape"(%1337) <{new_shape = array<i64: 1, 10, 1024, 64>}> : (tensor<10x1024x64xf32>) -> tensor<1x10x1024x64xf32>
    %1339 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1340 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1341 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1342 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1343 = "tosa.transpose"(%1338, %1342) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x1024x10x64xf32>
    %1344 = "tosa.reshape"(%1343) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x1024x10x64xf32>) -> tensor<1x1024x640xf32>
    %1345 = "tosa.reshape"(%1344) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1346 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1347 = "tosa.transpose"(%arg145, %1346) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %1348 = "tosa.reshape"(%1345) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1349 = "tosa.reshape"(%1347) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %1350 = "tosa.matmul"(%1348, %1349) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %1351 = "tosa.reshape"(%1350) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1352 = "tosa.reshape"(%arg146) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1353 = "tosa.add"(%1352, %1351) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1354 = "tosa.reshape"(%1353) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1355 = "tosa.identity"(%1354) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1356 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x640xf32>}> : () -> tensor<1x1024x640xf32>
    %1357 = "tosa.reciprocal"(%1356) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1358 = "tosa.mul"(%1355, %1357) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1359 = "tosa.add"(%1358, %1278) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1360 = "tosa.reduce_sum"(%1359) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %1361 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1362 = "tosa.reciprocal"(%1361) : (tensor<1xf32>) -> tensor<1xf32>
    %1363 = "tosa.mul"(%1362, %1360) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1364 = "tosa.sub"(%1359, %1363) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1365 = "tosa.mul"(%1364, %1364) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1366 = "tosa.reduce_sum"(%1365) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %1367 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1368 = "tosa.reciprocal"(%1367) : (tensor<1xf32>) -> tensor<1xf32>
    %1369 = "tosa.mul"(%1368, %1366) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1370 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %1371 = "tosa.add"(%1369, %1370) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1372 = "tosa.rsqrt"(%1371) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1373 = "tosa.sub"(%1359, %1363) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1374 = "tosa.mul"(%1373, %1372) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1375 = "tosa.reshape"(%arg147) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %1376 = "tosa.mul"(%1374, %1375) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %1377 = "tosa.reshape"(%arg148) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %1378 = "tosa.add"(%1376, %1377) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %1379 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1380 = "tosa.transpose"(%arg149, %1379) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %1381 = "tosa.reshape"(%1378) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1382 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %1383 = "linalg.matmul"(%1381, %1380, %1382) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1384 = "tosa.reshape"(%1383) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1385 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1386 = "tosa.transpose"(%arg150, %1385) : (tensor<640x1024xf32>, tensor<2xi32>) -> tensor<1024x640xf32>
    %1387 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1388 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x640xf32>}> : () -> tensor<12x640xf32>
    %1389 = "linalg.matmul"(%1387, %1386, %1388) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x640xf32>, tensor<12x640xf32>) -> tensor<12x640xf32>
    %1390 = "tosa.reshape"(%1389) <{new_shape = array<i64: 1, 12, 640>}> : (tensor<12x640xf32>) -> tensor<1x12x640xf32>
    %1391 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1392 = "tosa.transpose"(%arg152, %1391) : (tensor<640x1024xf32>, tensor<2xi32>) -> tensor<1024x640xf32>
    %1393 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1394 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x640xf32>}> : () -> tensor<12x640xf32>
    %1395 = "linalg.matmul"(%1393, %1392, %1394) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x640xf32>, tensor<12x640xf32>) -> tensor<12x640xf32>
    %1396 = "tosa.reshape"(%1395) <{new_shape = array<i64: 1, 12, 640>}> : (tensor<12x640xf32>) -> tensor<1x12x640xf32>
    %1397 = "tosa.reshape"(%1384) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %1398 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1399 = "tosa.transpose"(%1397, %1398) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %1400 = "tosa.reshape"(%1390) <{new_shape = array<i64: 1, 12, 10, 64>}> : (tensor<1x12x640xf32>) -> tensor<1x12x10x64xf32>
    %1401 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1402 = "tosa.transpose"(%1400, %1401) : (tensor<1x12x10x64xf32>, tensor<4xi32>) -> tensor<1x10x12x64xf32>
    %1403 = "tosa.reshape"(%1396) <{new_shape = array<i64: 1, 12, 10, 64>}> : (tensor<1x12x640xf32>) -> tensor<1x12x10x64xf32>
    %1404 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1405 = "tosa.transpose"(%1403, %1404) : (tensor<1x12x10x64xf32>, tensor<4xi32>) -> tensor<1x10x12x64xf32>
    %1406 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x12xf32>}> : () -> tensor<1024x12xf32>
    %1407 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1408 = "tosa.transpose"(%1402, %1407) : (tensor<1x10x12x64xf32>, tensor<4xi32>) -> tensor<1x10x64x12xf32>
    %1409 = "tosa.reshape"(%1399) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %1410 = "tosa.reshape"(%1408) <{new_shape = array<i64: 10, 64, 12>}> : (tensor<1x10x64x12xf32>) -> tensor<10x64x12xf32>
    %1411 = "tosa.matmul"(%1409, %1410) : (tensor<10x1024x64xf32>, tensor<10x64x12xf32>) -> tensor<10x1024x12xf32>
    %1412 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<10x1024x12xf32>}> : () -> tensor<10x1024x12xf32>
    %1413 = "tosa.mul"(%1411, %1412) <{shift = 0 : i8}> : (tensor<10x1024x12xf32>, tensor<10x1024x12xf32>) -> tensor<10x1024x12xf32>
    %1414 = "tosa.add"(%1413, %1406) : (tensor<10x1024x12xf32>, tensor<1024x12xf32>) -> tensor<10x1024x12xf32>
    %1415 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<10x1024x12xf32>}> : () -> tensor<10x1024x12xf32>
    %1416 = "linalg.softmax"(%1414, %1415) <{dimension = 3 : i64}> : (tensor<10x1024x12xf32>, tensor<10x1024x12xf32>) -> tensor<10x1024x12xf32>
    %1417 = "tosa.reshape"(%1405) <{new_shape = array<i64: 10, 12, 64>}> : (tensor<1x10x12x64xf32>) -> tensor<10x12x64xf32>
    %1418 = "tosa.matmul"(%1416, %1417) : (tensor<10x1024x12xf32>, tensor<10x12x64xf32>) -> tensor<10x1024x64xf32>
    %1419 = "tosa.reshape"(%1418) <{new_shape = array<i64: 1, 10, 1024, 64>}> : (tensor<10x1024x64xf32>) -> tensor<1x10x1024x64xf32>
    %1420 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1421 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1422 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1423 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1424 = "tosa.transpose"(%1419, %1423) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x1024x10x64xf32>
    %1425 = "tosa.reshape"(%1424) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x1024x10x64xf32>) -> tensor<1x1024x640xf32>
    %1426 = "tosa.reshape"(%1425) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1427 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1428 = "tosa.transpose"(%arg154, %1427) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %1429 = "tosa.reshape"(%1426) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1430 = "tosa.reshape"(%1428) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %1431 = "tosa.matmul"(%1429, %1430) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %1432 = "tosa.reshape"(%1431) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1433 = "tosa.reshape"(%arg155) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1434 = "tosa.add"(%1433, %1432) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1435 = "tosa.reshape"(%1434) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1436 = "tosa.identity"(%1435) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1437 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x640xf32>}> : () -> tensor<1x1024x640xf32>
    %1438 = "tosa.reciprocal"(%1437) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1439 = "tosa.mul"(%1436, %1438) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1440 = "tosa.add"(%1439, %1359) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1441 = "tosa.reduce_sum"(%1440) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %1442 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1443 = "tosa.reciprocal"(%1442) : (tensor<1xf32>) -> tensor<1xf32>
    %1444 = "tosa.mul"(%1443, %1441) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1445 = "tosa.sub"(%1440, %1444) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1446 = "tosa.mul"(%1445, %1445) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1447 = "tosa.reduce_sum"(%1446) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %1448 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1449 = "tosa.reciprocal"(%1448) : (tensor<1xf32>) -> tensor<1xf32>
    %1450 = "tosa.mul"(%1449, %1447) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1451 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %1452 = "tosa.add"(%1450, %1451) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1453 = "tosa.rsqrt"(%1452) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %1454 = "tosa.sub"(%1440, %1444) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1455 = "tosa.mul"(%1454, %1453) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %1456 = "tosa.reshape"(%arg156) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %1457 = "tosa.mul"(%1455, %1456) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %1458 = "tosa.reshape"(%arg157) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %1459 = "tosa.add"(%1457, %1458) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %1460 = "tosa.reshape"(%1459) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1461 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1462 = "tosa.transpose"(%arg158, %1461) : (tensor<5120x640xf32>, tensor<2xi32>) -> tensor<640x5120xf32>
    %1463 = "tosa.reshape"(%1460) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1464 = "tosa.reshape"(%1462) <{new_shape = array<i64: 1, 640, 5120>}> : (tensor<640x5120xf32>) -> tensor<1x640x5120xf32>
    %1465 = "tosa.matmul"(%1463, %1464) : (tensor<1x1024x640xf32>, tensor<1x640x5120xf32>) -> tensor<1x1024x5120xf32>
    %1466 = "tosa.reshape"(%1465) <{new_shape = array<i64: 1024, 5120>}> : (tensor<1x1024x5120xf32>) -> tensor<1024x5120xf32>
    %1467 = "tosa.reshape"(%arg159) <{new_shape = array<i64: 1, 5120>}> : (tensor<5120xf32>) -> tensor<1x5120xf32>
    %1468 = "tosa.add"(%1467, %1466) : (tensor<1x5120xf32>, tensor<1024x5120xf32>) -> tensor<1024x5120xf32>
    %1469 = "tosa.reshape"(%1468) <{new_shape = array<i64: 1, 1024, 5120>}> : (tensor<1024x5120xf32>) -> tensor<1x1024x5120xf32>
    %1470 = "tosa.slice"(%1469) <{size = array<i64: 0, 0, 2560>, start = array<i64: 0, 0, 0>}> : (tensor<1x1024x5120xf32>) -> tensor<1x1024x2560xf32>
    %1471 = "tosa.slice"(%1469) <{size = array<i64: 0, 0, 5120>, start = array<i64: 0, 0, 2560>}> : (tensor<1x1024x5120xf32>) -> tensor<1x1024x2560xf32>
    %1472 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %1473 = "tosa.mul"(%1471, %1472) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1474 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %1475 = "tosa.mul"(%1471, %1474) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1476 = "math.erf"(%1475) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1477 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %1478 = "tosa.add"(%1476, %1477) : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1479 = "tosa.mul"(%1473, %1478) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1480 = "tosa.mul"(%1470, %1479) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1481 = "tosa.identity"(%1480) : (tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1482 = "tosa.reshape"(%1481) <{new_shape = array<i64: 1024, 2560>}> : (tensor<1x1024x2560xf32>) -> tensor<1024x2560xf32>
    %1483 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1484 = "tosa.transpose"(%arg160, %1483) : (tensor<640x2560xf32>, tensor<2xi32>) -> tensor<2560x640xf32>
    %1485 = "tosa.reshape"(%1482) <{new_shape = array<i64: 1, 1024, 2560>}> : (tensor<1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %1486 = "tosa.reshape"(%1484) <{new_shape = array<i64: 1, 2560, 640>}> : (tensor<2560x640xf32>) -> tensor<1x2560x640xf32>
    %1487 = "tosa.matmul"(%1485, %1486) : (tensor<1x1024x2560xf32>, tensor<1x2560x640xf32>) -> tensor<1x1024x640xf32>
    %1488 = "tosa.reshape"(%1487) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1489 = "tosa.reshape"(%arg161) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1490 = "tosa.add"(%1489, %1488) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1491 = "tosa.reshape"(%1490) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1492 = "tosa.add"(%1491, %1440) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %1493 = "tosa.reshape"(%1492) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1494 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1495 = "tosa.transpose"(%arg162, %1494) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %1496 = "tosa.reshape"(%1493) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1497 = "tosa.reshape"(%1495) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %1498 = "tosa.matmul"(%1496, %1497) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %1499 = "tosa.reshape"(%1498) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %1500 = "tosa.reshape"(%arg163) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1501 = "tosa.add"(%1500, %1499) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %1502 = "tosa.reshape"(%1501) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %1503 = "tosa.reshape"(%1502) <{new_shape = array<i64: 1, 32, 32, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1x32x32x640xf32>
    %1504 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1505 = "tosa.transpose"(%1503, %1504) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %1506 = "tosa.identity"(%1505) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1507 = "tosa.add"(%1506, %1240) : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %1508 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1509 = "tosa.transpose"(%1507, %1508) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %1510 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1511 = "tosa.transpose"(%arg164, %1510) : (tensor<640x640x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x640xf32>
    %1512 = "tosa.conv2d"(%1509, %1511, %arg165) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<1x32x32x640xf32>, tensor<640x3x3x640xf32>, tensor<640xf32>) -> tensor<1x16x16x640xf32>
    %1513 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1514 = "tosa.transpose"(%1512, %1513) : (tensor<1x16x16x640xf32>, tensor<4xi32>) -> tensor<1x640x16x16xf32>
    %1515 = "tosa.reshape"(%1514) <{new_shape = array<i64: 1, 32, 20, 256>}> : (tensor<1x640x16x16xf32>) -> tensor<1x32x20x256xf32>
    %1516 = "tosa.reduce_sum"(%1515) <{axis = 2 : i32}> : (tensor<1x32x20x256xf32>) -> tensor<1x32x1x256xf32>
    %1517 = "tosa.reduce_sum"(%1516) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1518 = "tosa.const"() <{value = dense<5.120000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1519 = "tosa.reciprocal"(%1518) : (tensor<1xf32>) -> tensor<1xf32>
    %1520 = "tosa.mul"(%1519, %1517) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1521 = "tosa.sub"(%1515, %1520) : (tensor<1x32x20x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x256xf32>
    %1522 = "tosa.mul"(%1521, %1521) <{shift = 0 : i8}> : (tensor<1x32x20x256xf32>, tensor<1x32x20x256xf32>) -> tensor<1x32x20x256xf32>
    %1523 = "tosa.reduce_sum"(%1522) <{axis = 2 : i32}> : (tensor<1x32x20x256xf32>) -> tensor<1x32x1x256xf32>
    %1524 = "tosa.reduce_sum"(%1523) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1525 = "tosa.const"() <{value = dense<5.120000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1526 = "tosa.reciprocal"(%1525) : (tensor<1xf32>) -> tensor<1xf32>
    %1527 = "tosa.mul"(%1526, %1524) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1528 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1529 = "tosa.add"(%1527, %1528) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1530 = "tosa.rsqrt"(%1529) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1531 = "tosa.sub"(%1515, %1520) : (tensor<1x32x20x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x256xf32>
    %1532 = "tosa.mul"(%1531, %1530) <{shift = 0 : i8}> : (tensor<1x32x20x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x256xf32>
    %1533 = "tosa.reshape"(%1532) <{new_shape = array<i64: 1, 640, 16, 16>}> : (tensor<1x32x20x256xf32>) -> tensor<1x640x16x16xf32>
    %1534 = "tosa.reshape"(%arg166) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1535 = "tosa.reshape"(%1534) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %1536 = "tosa.reshape"(%1535) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %1537 = "tosa.reshape"(%arg167) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %1538 = "tosa.reshape"(%1537) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %1539 = "tosa.reshape"(%1538) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %1540 = "tosa.mul"(%1533, %1539) <{shift = 0 : i8}> : (tensor<1x640x16x16xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x16x16xf32>
    %1541 = "tosa.add"(%1540, %1536) : (tensor<1x640x16x16xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x16x16xf32>
    %1542 = "tosa.sigmoid"(%1541) : (tensor<1x640x16x16xf32>) -> tensor<1x640x16x16xf32>
    %1543 = "tosa.mul"(%1541, %1542) <{shift = 0 : i8}> : (tensor<1x640x16x16xf32>, tensor<1x640x16x16xf32>) -> tensor<1x640x16x16xf32>
    %1544 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1545 = "tosa.transpose"(%1543, %1544) : (tensor<1x640x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x640xf32>
    %1546 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1547 = "tosa.transpose"(%arg168, %1546) : (tensor<1280x640x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x640xf32>
    %1548 = "tosa.conv2d"(%1545, %1547, %arg169) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x640xf32>, tensor<1280x3x3x640xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %1549 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1550 = "tosa.transpose"(%1548, %1549) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %1551 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1552 = "tosa.mul"(%50, %1551) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1553 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1554 = "tosa.transpose"(%arg170, %1553) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %1555 = "tosa.reshape"(%1552) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %1556 = "tosa.reshape"(%1554) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %1557 = "tosa.matmul"(%1555, %1556) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %1558 = "tosa.reshape"(%1557) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %1559 = "tosa.reshape"(%arg171) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1560 = "tosa.add"(%1559, %1558) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1561 = "tensor.extract_slice"(%1560) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1562 = "tensor.extract_slice"(%1561) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1563 = "tosa.reshape"(%1562) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1564 = "tosa.reshape"(%1563) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1565 = "tosa.add"(%1550, %1564) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %1566 = "tosa.reshape"(%1565) <{new_shape = array<i64: 1, 32, 40, 256>}> : (tensor<1x1280x16x16xf32>) -> tensor<1x32x40x256xf32>
    %1567 = "tosa.reduce_sum"(%1566) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %1568 = "tosa.reduce_sum"(%1567) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1569 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1570 = "tosa.reciprocal"(%1569) : (tensor<1xf32>) -> tensor<1xf32>
    %1571 = "tosa.mul"(%1570, %1568) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1572 = "tosa.sub"(%1566, %1571) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1573 = "tosa.mul"(%1572, %1572) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x40x256xf32>) -> tensor<1x32x40x256xf32>
    %1574 = "tosa.reduce_sum"(%1573) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %1575 = "tosa.reduce_sum"(%1574) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1576 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1577 = "tosa.reciprocal"(%1576) : (tensor<1xf32>) -> tensor<1xf32>
    %1578 = "tosa.mul"(%1577, %1575) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1579 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1580 = "tosa.add"(%1578, %1579) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1581 = "tosa.rsqrt"(%1580) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1582 = "tosa.sub"(%1566, %1571) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1583 = "tosa.mul"(%1582, %1581) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1584 = "tosa.reshape"(%1583) <{new_shape = array<i64: 1, 1280, 16, 16>}> : (tensor<1x32x40x256xf32>) -> tensor<1x1280x16x16xf32>
    %1585 = "tosa.reshape"(%arg172) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1586 = "tosa.reshape"(%1585) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1587 = "tosa.reshape"(%1586) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1588 = "tosa.reshape"(%arg173) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1589 = "tosa.reshape"(%1588) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1590 = "tosa.reshape"(%1589) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1591 = "tosa.mul"(%1584, %1590) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %1592 = "tosa.add"(%1591, %1587) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %1593 = "tosa.sigmoid"(%1592) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1594 = "tosa.mul"(%1592, %1593) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1595 = "tosa.identity"(%1594) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1596 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1597 = "tosa.transpose"(%1595, %1596) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %1598 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1599 = "tosa.transpose"(%arg174, %1598) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %1600 = "tosa.conv2d"(%1597, %1599, %arg175) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %1601 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1602 = "tosa.transpose"(%1600, %1601) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %1603 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1604 = "tosa.transpose"(%1514, %1603) : (tensor<1x640x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x640xf32>
    %1605 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1606 = "tosa.transpose"(%arg176, %1605) : (tensor<1280x640x1x1xf32>, tensor<4xi32>) -> tensor<1280x1x1x640xf32>
    %1607 = "tosa.conv2d"(%1604, %1606, %arg177) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x640xf32>, tensor<1280x1x1x640xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %1608 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1609 = "tosa.transpose"(%1607, %1608) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %1610 = "tosa.add"(%1609, %1602) : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1611 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x16x16xf32>}> : () -> tensor<1x1280x16x16xf32>
    %1612 = "tosa.reciprocal"(%1611) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1613 = "tosa.mul"(%1610, %1612) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1614 = "tosa.reshape"(%1613) <{new_shape = array<i64: 1, 32, 40, 256>}> : (tensor<1x1280x16x16xf32>) -> tensor<1x32x40x256xf32>
    %1615 = "tosa.reduce_sum"(%1614) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %1616 = "tosa.reduce_sum"(%1615) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1617 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1618 = "tosa.reciprocal"(%1617) : (tensor<1xf32>) -> tensor<1xf32>
    %1619 = "tosa.mul"(%1618, %1616) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1620 = "tosa.sub"(%1614, %1619) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1621 = "tosa.mul"(%1620, %1620) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x40x256xf32>) -> tensor<1x32x40x256xf32>
    %1622 = "tosa.reduce_sum"(%1621) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %1623 = "tosa.reduce_sum"(%1622) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1624 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1625 = "tosa.reciprocal"(%1624) : (tensor<1xf32>) -> tensor<1xf32>
    %1626 = "tosa.mul"(%1625, %1623) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1627 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1628 = "tosa.add"(%1626, %1627) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1629 = "tosa.rsqrt"(%1628) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1630 = "tosa.sub"(%1614, %1619) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1631 = "tosa.mul"(%1630, %1629) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1632 = "tosa.reshape"(%1631) <{new_shape = array<i64: 1, 1280, 16, 16>}> : (tensor<1x32x40x256xf32>) -> tensor<1x1280x16x16xf32>
    %1633 = "tosa.reshape"(%arg178) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1634 = "tosa.reshape"(%1633) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1635 = "tosa.reshape"(%1634) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1636 = "tosa.reshape"(%arg179) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1637 = "tosa.reshape"(%1636) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1638 = "tosa.reshape"(%1637) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1639 = "tosa.mul"(%1632, %1638) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %1640 = "tosa.add"(%1639, %1635) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %1641 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1642 = "tosa.transpose"(%1640, %1641) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %1643 = "tosa.reshape"(%1642) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x16x16x1280xf32>) -> tensor<1x256x1280xf32>
    %1644 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1645 = "tosa.transpose"(%arg180, %1644) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %1646 = "tosa.reshape"(%1643) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1647 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %1648 = "linalg.matmul"(%1646, %1645, %1647) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1649 = "tosa.reshape"(%1648) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1650 = "tosa.reshape"(%arg181) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %1651 = "tosa.add"(%1649, %1650) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %1652 = "tosa.reduce_sum"(%1651) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %1653 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1654 = "tosa.reciprocal"(%1653) : (tensor<1xf32>) -> tensor<1xf32>
    %1655 = "tosa.mul"(%1654, %1652) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1656 = "tosa.sub"(%1651, %1655) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1657 = "tosa.mul"(%1656, %1656) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1658 = "tosa.reduce_sum"(%1657) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %1659 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1660 = "tosa.reciprocal"(%1659) : (tensor<1xf32>) -> tensor<1xf32>
    %1661 = "tosa.mul"(%1660, %1658) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1662 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %1663 = "tosa.add"(%1661, %1662) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1664 = "tosa.rsqrt"(%1663) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1665 = "tosa.sub"(%1651, %1655) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1666 = "tosa.mul"(%1665, %1664) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1667 = "tosa.reshape"(%arg182) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %1668 = "tosa.mul"(%1666, %1667) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %1669 = "tosa.reshape"(%arg183) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %1670 = "tosa.add"(%1668, %1669) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %1671 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1672 = "tosa.transpose"(%arg184, %1671) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %1673 = "tosa.reshape"(%1670) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1674 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %1675 = "linalg.matmul"(%1673, %1672, %1674) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1676 = "tosa.reshape"(%1675) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1677 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1678 = "tosa.transpose"(%arg185, %1677) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %1679 = "tosa.reshape"(%1670) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1680 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %1681 = "linalg.matmul"(%1679, %1678, %1680) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1682 = "tosa.reshape"(%1681) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1683 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1684 = "tosa.transpose"(%arg186, %1683) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %1685 = "tosa.reshape"(%1670) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1686 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %1687 = "linalg.matmul"(%1685, %1684, %1686) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1688 = "tosa.reshape"(%1687) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1689 = "tosa.reshape"(%1676) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %1690 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1691 = "tosa.transpose"(%1689, %1690) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %1692 = "tosa.reshape"(%1682) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %1693 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1694 = "tosa.transpose"(%1692, %1693) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %1695 = "tosa.reshape"(%1688) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %1696 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1697 = "tosa.transpose"(%1695, %1696) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %1698 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x256xf32>}> : () -> tensor<256x256xf32>
    %1699 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1700 = "tosa.transpose"(%1694, %1699) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x20x64x256xf32>
    %1701 = "tosa.reshape"(%1691) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %1702 = "tosa.reshape"(%1700) <{new_shape = array<i64: 20, 64, 256>}> : (tensor<1x20x64x256xf32>) -> tensor<20x64x256xf32>
    %1703 = "tosa.matmul"(%1701, %1702) : (tensor<20x256x64xf32>, tensor<20x64x256xf32>) -> tensor<20x256x256xf32>
    %1704 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x256x256xf32>}> : () -> tensor<20x256x256xf32>
    %1705 = "tosa.mul"(%1703, %1704) <{shift = 0 : i8}> : (tensor<20x256x256xf32>, tensor<20x256x256xf32>) -> tensor<20x256x256xf32>
    %1706 = "tosa.add"(%1705, %1698) : (tensor<20x256x256xf32>, tensor<256x256xf32>) -> tensor<20x256x256xf32>
    %1707 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x256x256xf32>}> : () -> tensor<20x256x256xf32>
    %1708 = "linalg.softmax"(%1706, %1707) <{dimension = 3 : i64}> : (tensor<20x256x256xf32>, tensor<20x256x256xf32>) -> tensor<20x256x256xf32>
    %1709 = "tosa.reshape"(%1697) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %1710 = "tosa.matmul"(%1708, %1709) : (tensor<20x256x256xf32>, tensor<20x256x64xf32>) -> tensor<20x256x64xf32>
    %1711 = "tosa.reshape"(%1710) <{new_shape = array<i64: 1, 20, 256, 64>}> : (tensor<20x256x64xf32>) -> tensor<1x20x256x64xf32>
    %1712 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1713 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1714 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1715 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1716 = "tosa.transpose"(%1711, %1715) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x256x20x64xf32>
    %1717 = "tosa.reshape"(%1716) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x256x20x64xf32>) -> tensor<1x256x1280xf32>
    %1718 = "tosa.reshape"(%1717) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1719 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1720 = "tosa.transpose"(%arg187, %1719) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %1721 = "tosa.reshape"(%1718) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1722 = "tosa.reshape"(%1720) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %1723 = "tosa.matmul"(%1721, %1722) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %1724 = "tosa.reshape"(%1723) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1725 = "tosa.reshape"(%arg188) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1726 = "tosa.add"(%1725, %1724) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1727 = "tosa.reshape"(%1726) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1728 = "tosa.identity"(%1727) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1729 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x1280xf32>}> : () -> tensor<1x256x1280xf32>
    %1730 = "tosa.reciprocal"(%1729) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1731 = "tosa.mul"(%1728, %1730) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1732 = "tosa.add"(%1731, %1651) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1733 = "tosa.reduce_sum"(%1732) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %1734 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1735 = "tosa.reciprocal"(%1734) : (tensor<1xf32>) -> tensor<1xf32>
    %1736 = "tosa.mul"(%1735, %1733) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1737 = "tosa.sub"(%1732, %1736) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1738 = "tosa.mul"(%1737, %1737) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1739 = "tosa.reduce_sum"(%1738) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %1740 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1741 = "tosa.reciprocal"(%1740) : (tensor<1xf32>) -> tensor<1xf32>
    %1742 = "tosa.mul"(%1741, %1739) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1743 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %1744 = "tosa.add"(%1742, %1743) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1745 = "tosa.rsqrt"(%1744) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1746 = "tosa.sub"(%1732, %1736) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1747 = "tosa.mul"(%1746, %1745) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1748 = "tosa.reshape"(%arg189) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %1749 = "tosa.mul"(%1747, %1748) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %1750 = "tosa.reshape"(%arg190) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %1751 = "tosa.add"(%1749, %1750) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %1752 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1753 = "tosa.transpose"(%arg191, %1752) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %1754 = "tosa.reshape"(%1751) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1755 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %1756 = "linalg.matmul"(%1754, %1753, %1755) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1757 = "tosa.reshape"(%1756) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1758 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1759 = "tosa.transpose"(%arg192, %1758) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %1760 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1761 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %1762 = "linalg.matmul"(%1760, %1759, %1761) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %1763 = "tosa.reshape"(%1762) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %1764 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1765 = "tosa.transpose"(%arg194, %1764) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %1766 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %1767 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %1768 = "linalg.matmul"(%1766, %1765, %1767) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %1769 = "tosa.reshape"(%1768) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %1770 = "tosa.reshape"(%1757) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %1771 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1772 = "tosa.transpose"(%1770, %1771) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %1773 = "tosa.reshape"(%1763) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %1774 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1775 = "tosa.transpose"(%1773, %1774) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %1776 = "tosa.reshape"(%1769) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %1777 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1778 = "tosa.transpose"(%1776, %1777) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %1779 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x12xf32>}> : () -> tensor<256x12xf32>
    %1780 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1781 = "tosa.transpose"(%1775, %1780) : (tensor<1x20x12x64xf32>, tensor<4xi32>) -> tensor<1x20x64x12xf32>
    %1782 = "tosa.reshape"(%1772) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %1783 = "tosa.reshape"(%1781) <{new_shape = array<i64: 20, 64, 12>}> : (tensor<1x20x64x12xf32>) -> tensor<20x64x12xf32>
    %1784 = "tosa.matmul"(%1782, %1783) : (tensor<20x256x64xf32>, tensor<20x64x12xf32>) -> tensor<20x256x12xf32>
    %1785 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x256x12xf32>}> : () -> tensor<20x256x12xf32>
    %1786 = "tosa.mul"(%1784, %1785) <{shift = 0 : i8}> : (tensor<20x256x12xf32>, tensor<20x256x12xf32>) -> tensor<20x256x12xf32>
    %1787 = "tosa.add"(%1786, %1779) : (tensor<20x256x12xf32>, tensor<256x12xf32>) -> tensor<20x256x12xf32>
    %1788 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x256x12xf32>}> : () -> tensor<20x256x12xf32>
    %1789 = "linalg.softmax"(%1787, %1788) <{dimension = 3 : i64}> : (tensor<20x256x12xf32>, tensor<20x256x12xf32>) -> tensor<20x256x12xf32>
    %1790 = "tosa.reshape"(%1778) <{new_shape = array<i64: 20, 12, 64>}> : (tensor<1x20x12x64xf32>) -> tensor<20x12x64xf32>
    %1791 = "tosa.matmul"(%1789, %1790) : (tensor<20x256x12xf32>, tensor<20x12x64xf32>) -> tensor<20x256x64xf32>
    %1792 = "tosa.reshape"(%1791) <{new_shape = array<i64: 1, 20, 256, 64>}> : (tensor<20x256x64xf32>) -> tensor<1x20x256x64xf32>
    %1793 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1794 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1795 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %1796 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1797 = "tosa.transpose"(%1792, %1796) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x256x20x64xf32>
    %1798 = "tosa.reshape"(%1797) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x256x20x64xf32>) -> tensor<1x256x1280xf32>
    %1799 = "tosa.reshape"(%1798) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1800 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1801 = "tosa.transpose"(%arg196, %1800) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %1802 = "tosa.reshape"(%1799) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1803 = "tosa.reshape"(%1801) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %1804 = "tosa.matmul"(%1802, %1803) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %1805 = "tosa.reshape"(%1804) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1806 = "tosa.reshape"(%arg197) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1807 = "tosa.add"(%1806, %1805) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1808 = "tosa.reshape"(%1807) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1809 = "tosa.identity"(%1808) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1810 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x1280xf32>}> : () -> tensor<1x256x1280xf32>
    %1811 = "tosa.reciprocal"(%1810) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1812 = "tosa.mul"(%1809, %1811) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1813 = "tosa.add"(%1812, %1732) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1814 = "tosa.reduce_sum"(%1813) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %1815 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1816 = "tosa.reciprocal"(%1815) : (tensor<1xf32>) -> tensor<1xf32>
    %1817 = "tosa.mul"(%1816, %1814) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1818 = "tosa.sub"(%1813, %1817) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1819 = "tosa.mul"(%1818, %1818) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1820 = "tosa.reduce_sum"(%1819) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %1821 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1822 = "tosa.reciprocal"(%1821) : (tensor<1xf32>) -> tensor<1xf32>
    %1823 = "tosa.mul"(%1822, %1820) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1824 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %1825 = "tosa.add"(%1823, %1824) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1826 = "tosa.rsqrt"(%1825) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %1827 = "tosa.sub"(%1813, %1817) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1828 = "tosa.mul"(%1827, %1826) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %1829 = "tosa.reshape"(%arg198) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %1830 = "tosa.mul"(%1828, %1829) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %1831 = "tosa.reshape"(%arg199) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %1832 = "tosa.add"(%1830, %1831) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %1833 = "tosa.reshape"(%1832) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1834 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1835 = "tosa.transpose"(%arg200, %1834) : (tensor<10240x1280xf32>, tensor<2xi32>) -> tensor<1280x10240xf32>
    %1836 = "tosa.reshape"(%1833) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1837 = "tosa.reshape"(%1835) <{new_shape = array<i64: 1, 1280, 10240>}> : (tensor<1280x10240xf32>) -> tensor<1x1280x10240xf32>
    %1838 = "tosa.matmul"(%1836, %1837) : (tensor<1x256x1280xf32>, tensor<1x1280x10240xf32>) -> tensor<1x256x10240xf32>
    %1839 = "tosa.reshape"(%1838) <{new_shape = array<i64: 256, 10240>}> : (tensor<1x256x10240xf32>) -> tensor<256x10240xf32>
    %1840 = "tosa.reshape"(%arg201) <{new_shape = array<i64: 1, 10240>}> : (tensor<10240xf32>) -> tensor<1x10240xf32>
    %1841 = "tosa.add"(%1840, %1839) : (tensor<1x10240xf32>, tensor<256x10240xf32>) -> tensor<256x10240xf32>
    %1842 = "tosa.reshape"(%1841) <{new_shape = array<i64: 1, 256, 10240>}> : (tensor<256x10240xf32>) -> tensor<1x256x10240xf32>
    %1843 = "tosa.slice"(%1842) <{size = array<i64: 0, 0, 5120>, start = array<i64: 0, 0, 0>}> : (tensor<1x256x10240xf32>) -> tensor<1x256x5120xf32>
    %1844 = "tosa.slice"(%1842) <{size = array<i64: 0, 0, 10240>, start = array<i64: 0, 0, 5120>}> : (tensor<1x256x10240xf32>) -> tensor<1x256x5120xf32>
    %1845 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %1846 = "tosa.mul"(%1844, %1845) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %1847 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %1848 = "tosa.mul"(%1844, %1847) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %1849 = "math.erf"(%1848) <{fastmath = #arith.fastmath<none>}> : (tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %1850 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %1851 = "tosa.add"(%1849, %1850) : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %1852 = "tosa.mul"(%1846, %1851) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %1853 = "tosa.mul"(%1843, %1852) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %1854 = "tosa.identity"(%1853) : (tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %1855 = "tosa.reshape"(%1854) <{new_shape = array<i64: 256, 5120>}> : (tensor<1x256x5120xf32>) -> tensor<256x5120xf32>
    %1856 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1857 = "tosa.transpose"(%arg202, %1856) : (tensor<1280x5120xf32>, tensor<2xi32>) -> tensor<5120x1280xf32>
    %1858 = "tosa.reshape"(%1855) <{new_shape = array<i64: 1, 256, 5120>}> : (tensor<256x5120xf32>) -> tensor<1x256x5120xf32>
    %1859 = "tosa.reshape"(%1857) <{new_shape = array<i64: 1, 5120, 1280>}> : (tensor<5120x1280xf32>) -> tensor<1x5120x1280xf32>
    %1860 = "tosa.matmul"(%1858, %1859) : (tensor<1x256x5120xf32>, tensor<1x5120x1280xf32>) -> tensor<1x256x1280xf32>
    %1861 = "tosa.reshape"(%1860) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1862 = "tosa.reshape"(%arg203) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1863 = "tosa.add"(%1862, %1861) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1864 = "tosa.reshape"(%1863) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1865 = "tosa.add"(%1864, %1813) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %1866 = "tosa.reshape"(%1865) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1867 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1868 = "tosa.transpose"(%arg204, %1867) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %1869 = "tosa.reshape"(%1866) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1870 = "tosa.reshape"(%1868) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %1871 = "tosa.matmul"(%1869, %1870) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %1872 = "tosa.reshape"(%1871) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %1873 = "tosa.reshape"(%arg205) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1874 = "tosa.add"(%1873, %1872) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %1875 = "tosa.reshape"(%1874) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %1876 = "tosa.reshape"(%1875) <{new_shape = array<i64: 1, 16, 16, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<1x16x16x1280xf32>
    %1877 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1878 = "tosa.transpose"(%1876, %1877) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %1879 = "tosa.identity"(%1878) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1880 = "tosa.add"(%1879, %1613) : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1881 = "tosa.reshape"(%1880) <{new_shape = array<i64: 1, 32, 40, 256>}> : (tensor<1x1280x16x16xf32>) -> tensor<1x32x40x256xf32>
    %1882 = "tosa.reduce_sum"(%1881) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %1883 = "tosa.reduce_sum"(%1882) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1884 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1885 = "tosa.reciprocal"(%1884) : (tensor<1xf32>) -> tensor<1xf32>
    %1886 = "tosa.mul"(%1885, %1883) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1887 = "tosa.sub"(%1881, %1886) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1888 = "tosa.mul"(%1887, %1887) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x40x256xf32>) -> tensor<1x32x40x256xf32>
    %1889 = "tosa.reduce_sum"(%1888) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %1890 = "tosa.reduce_sum"(%1889) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1891 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1892 = "tosa.reciprocal"(%1891) : (tensor<1xf32>) -> tensor<1xf32>
    %1893 = "tosa.mul"(%1892, %1890) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1894 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1895 = "tosa.add"(%1893, %1894) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1896 = "tosa.rsqrt"(%1895) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1897 = "tosa.sub"(%1881, %1886) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1898 = "tosa.mul"(%1897, %1896) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1899 = "tosa.reshape"(%1898) <{new_shape = array<i64: 1, 1280, 16, 16>}> : (tensor<1x32x40x256xf32>) -> tensor<1x1280x16x16xf32>
    %1900 = "tosa.reshape"(%arg206) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1901 = "tosa.reshape"(%1900) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1902 = "tosa.reshape"(%1901) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1903 = "tosa.reshape"(%arg207) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1904 = "tosa.reshape"(%1903) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1905 = "tosa.reshape"(%1904) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1906 = "tosa.mul"(%1899, %1905) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %1907 = "tosa.add"(%1906, %1902) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %1908 = "tosa.sigmoid"(%1907) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1909 = "tosa.mul"(%1907, %1908) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1910 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1911 = "tosa.transpose"(%1909, %1910) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %1912 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1913 = "tosa.transpose"(%arg208, %1912) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %1914 = "tosa.conv2d"(%1911, %1913, %arg209) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %1915 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1916 = "tosa.transpose"(%1914, %1915) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %1917 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1918 = "tosa.mul"(%50, %1917) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1919 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1920 = "tosa.transpose"(%arg210, %1919) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %1921 = "tosa.reshape"(%1918) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %1922 = "tosa.reshape"(%1920) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %1923 = "tosa.matmul"(%1921, %1922) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %1924 = "tosa.reshape"(%1923) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %1925 = "tosa.reshape"(%arg211) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1926 = "tosa.add"(%1925, %1924) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1927 = "tensor.extract_slice"(%1926) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1928 = "tensor.extract_slice"(%1927) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %1929 = "tosa.reshape"(%1928) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1930 = "tosa.reshape"(%1929) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1931 = "tosa.add"(%1916, %1930) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %1932 = "tosa.reshape"(%1931) <{new_shape = array<i64: 1, 32, 40, 256>}> : (tensor<1x1280x16x16xf32>) -> tensor<1x32x40x256xf32>
    %1933 = "tosa.reduce_sum"(%1932) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %1934 = "tosa.reduce_sum"(%1933) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1935 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1936 = "tosa.reciprocal"(%1935) : (tensor<1xf32>) -> tensor<1xf32>
    %1937 = "tosa.mul"(%1936, %1934) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1938 = "tosa.sub"(%1932, %1937) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1939 = "tosa.mul"(%1938, %1938) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x40x256xf32>) -> tensor<1x32x40x256xf32>
    %1940 = "tosa.reduce_sum"(%1939) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %1941 = "tosa.reduce_sum"(%1940) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1942 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1943 = "tosa.reciprocal"(%1942) : (tensor<1xf32>) -> tensor<1xf32>
    %1944 = "tosa.mul"(%1943, %1941) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1945 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1946 = "tosa.add"(%1944, %1945) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1947 = "tosa.rsqrt"(%1946) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1948 = "tosa.sub"(%1932, %1937) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1949 = "tosa.mul"(%1948, %1947) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1950 = "tosa.reshape"(%1949) <{new_shape = array<i64: 1, 1280, 16, 16>}> : (tensor<1x32x40x256xf32>) -> tensor<1x1280x16x16xf32>
    %1951 = "tosa.reshape"(%arg212) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1952 = "tosa.reshape"(%1951) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1953 = "tosa.reshape"(%1952) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1954 = "tosa.reshape"(%arg213) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1955 = "tosa.reshape"(%1954) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1956 = "tosa.reshape"(%1955) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1957 = "tosa.mul"(%1950, %1956) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %1958 = "tosa.add"(%1957, %1953) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %1959 = "tosa.sigmoid"(%1958) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1960 = "tosa.mul"(%1958, %1959) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1961 = "tosa.identity"(%1960) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1962 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1963 = "tosa.transpose"(%1961, %1962) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %1964 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1965 = "tosa.transpose"(%arg214, %1964) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %1966 = "tosa.conv2d"(%1963, %1965, %arg215) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %1967 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1968 = "tosa.transpose"(%1966, %1967) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %1969 = "tosa.add"(%1880, %1968) : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1970 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x16x16xf32>}> : () -> tensor<1x1280x16x16xf32>
    %1971 = "tosa.reciprocal"(%1970) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1972 = "tosa.mul"(%1969, %1971) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %1973 = "tosa.reshape"(%1972) <{new_shape = array<i64: 1, 32, 40, 256>}> : (tensor<1x1280x16x16xf32>) -> tensor<1x32x40x256xf32>
    %1974 = "tosa.reduce_sum"(%1973) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %1975 = "tosa.reduce_sum"(%1974) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1976 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1977 = "tosa.reciprocal"(%1976) : (tensor<1xf32>) -> tensor<1xf32>
    %1978 = "tosa.mul"(%1977, %1975) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1979 = "tosa.sub"(%1973, %1978) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1980 = "tosa.mul"(%1979, %1979) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x40x256xf32>) -> tensor<1x32x40x256xf32>
    %1981 = "tosa.reduce_sum"(%1980) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %1982 = "tosa.reduce_sum"(%1981) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %1983 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1984 = "tosa.reciprocal"(%1983) : (tensor<1xf32>) -> tensor<1xf32>
    %1985 = "tosa.mul"(%1984, %1982) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1986 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %1987 = "tosa.add"(%1985, %1986) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1988 = "tosa.rsqrt"(%1987) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %1989 = "tosa.sub"(%1973, %1978) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1990 = "tosa.mul"(%1989, %1988) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %1991 = "tosa.reshape"(%1990) <{new_shape = array<i64: 1, 1280, 16, 16>}> : (tensor<1x32x40x256xf32>) -> tensor<1x1280x16x16xf32>
    %1992 = "tosa.reshape"(%arg216) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1993 = "tosa.reshape"(%1992) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1994 = "tosa.reshape"(%1993) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1995 = "tosa.reshape"(%arg217) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %1996 = "tosa.reshape"(%1995) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %1997 = "tosa.reshape"(%1996) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %1998 = "tosa.mul"(%1991, %1997) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %1999 = "tosa.add"(%1998, %1994) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %2000 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2001 = "tosa.transpose"(%1999, %2000) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %2002 = "tosa.reshape"(%2001) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x16x16x1280xf32>) -> tensor<1x256x1280xf32>
    %2003 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2004 = "tosa.transpose"(%arg218, %2003) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2005 = "tosa.reshape"(%2002) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2006 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %2007 = "linalg.matmul"(%2005, %2004, %2006) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2008 = "tosa.reshape"(%2007) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2009 = "tosa.reshape"(%arg219) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2010 = "tosa.add"(%2008, %2009) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %2011 = "tosa.reduce_sum"(%2010) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %2012 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2013 = "tosa.reciprocal"(%2012) : (tensor<1xf32>) -> tensor<1xf32>
    %2014 = "tosa.mul"(%2013, %2011) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2015 = "tosa.sub"(%2010, %2014) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2016 = "tosa.mul"(%2015, %2015) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2017 = "tosa.reduce_sum"(%2016) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %2018 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2019 = "tosa.reciprocal"(%2018) : (tensor<1xf32>) -> tensor<1xf32>
    %2020 = "tosa.mul"(%2019, %2017) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2021 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %2022 = "tosa.add"(%2020, %2021) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2023 = "tosa.rsqrt"(%2022) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2024 = "tosa.sub"(%2010, %2014) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2025 = "tosa.mul"(%2024, %2023) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2026 = "tosa.reshape"(%arg220) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2027 = "tosa.mul"(%2025, %2026) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %2028 = "tosa.reshape"(%arg221) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2029 = "tosa.add"(%2027, %2028) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %2030 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2031 = "tosa.transpose"(%arg222, %2030) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2032 = "tosa.reshape"(%2029) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2033 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %2034 = "linalg.matmul"(%2032, %2031, %2033) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2035 = "tosa.reshape"(%2034) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2036 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2037 = "tosa.transpose"(%arg223, %2036) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2038 = "tosa.reshape"(%2029) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2039 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %2040 = "linalg.matmul"(%2038, %2037, %2039) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2041 = "tosa.reshape"(%2040) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2042 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2043 = "tosa.transpose"(%arg224, %2042) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2044 = "tosa.reshape"(%2029) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2045 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %2046 = "linalg.matmul"(%2044, %2043, %2045) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2047 = "tosa.reshape"(%2046) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2048 = "tosa.reshape"(%2035) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %2049 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2050 = "tosa.transpose"(%2048, %2049) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %2051 = "tosa.reshape"(%2041) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %2052 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2053 = "tosa.transpose"(%2051, %2052) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %2054 = "tosa.reshape"(%2047) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %2055 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2056 = "tosa.transpose"(%2054, %2055) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %2057 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x256xf32>}> : () -> tensor<256x256xf32>
    %2058 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2059 = "tosa.transpose"(%2053, %2058) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x20x64x256xf32>
    %2060 = "tosa.reshape"(%2050) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %2061 = "tosa.reshape"(%2059) <{new_shape = array<i64: 20, 64, 256>}> : (tensor<1x20x64x256xf32>) -> tensor<20x64x256xf32>
    %2062 = "tosa.matmul"(%2060, %2061) : (tensor<20x256x64xf32>, tensor<20x64x256xf32>) -> tensor<20x256x256xf32>
    %2063 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x256x256xf32>}> : () -> tensor<20x256x256xf32>
    %2064 = "tosa.mul"(%2062, %2063) <{shift = 0 : i8}> : (tensor<20x256x256xf32>, tensor<20x256x256xf32>) -> tensor<20x256x256xf32>
    %2065 = "tosa.add"(%2064, %2057) : (tensor<20x256x256xf32>, tensor<256x256xf32>) -> tensor<20x256x256xf32>
    %2066 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x256x256xf32>}> : () -> tensor<20x256x256xf32>
    %2067 = "linalg.softmax"(%2065, %2066) <{dimension = 3 : i64}> : (tensor<20x256x256xf32>, tensor<20x256x256xf32>) -> tensor<20x256x256xf32>
    %2068 = "tosa.reshape"(%2056) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %2069 = "tosa.matmul"(%2067, %2068) : (tensor<20x256x256xf32>, tensor<20x256x64xf32>) -> tensor<20x256x64xf32>
    %2070 = "tosa.reshape"(%2069) <{new_shape = array<i64: 1, 20, 256, 64>}> : (tensor<20x256x64xf32>) -> tensor<1x20x256x64xf32>
    %2071 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2072 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2073 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2074 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2075 = "tosa.transpose"(%2070, %2074) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x256x20x64xf32>
    %2076 = "tosa.reshape"(%2075) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x256x20x64xf32>) -> tensor<1x256x1280xf32>
    %2077 = "tosa.reshape"(%2076) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2078 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2079 = "tosa.transpose"(%arg225, %2078) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2080 = "tosa.reshape"(%2077) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2081 = "tosa.reshape"(%2079) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %2082 = "tosa.matmul"(%2080, %2081) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %2083 = "tosa.reshape"(%2082) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2084 = "tosa.reshape"(%arg226) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2085 = "tosa.add"(%2084, %2083) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2086 = "tosa.reshape"(%2085) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2087 = "tosa.identity"(%2086) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2088 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x1280xf32>}> : () -> tensor<1x256x1280xf32>
    %2089 = "tosa.reciprocal"(%2088) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2090 = "tosa.mul"(%2087, %2089) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2091 = "tosa.add"(%2090, %2010) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2092 = "tosa.reduce_sum"(%2091) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %2093 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2094 = "tosa.reciprocal"(%2093) : (tensor<1xf32>) -> tensor<1xf32>
    %2095 = "tosa.mul"(%2094, %2092) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2096 = "tosa.sub"(%2091, %2095) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2097 = "tosa.mul"(%2096, %2096) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2098 = "tosa.reduce_sum"(%2097) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %2099 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2100 = "tosa.reciprocal"(%2099) : (tensor<1xf32>) -> tensor<1xf32>
    %2101 = "tosa.mul"(%2100, %2098) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2102 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %2103 = "tosa.add"(%2101, %2102) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2104 = "tosa.rsqrt"(%2103) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2105 = "tosa.sub"(%2091, %2095) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2106 = "tosa.mul"(%2105, %2104) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2107 = "tosa.reshape"(%arg227) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2108 = "tosa.mul"(%2106, %2107) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %2109 = "tosa.reshape"(%arg228) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2110 = "tosa.add"(%2108, %2109) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %2111 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2112 = "tosa.transpose"(%arg229, %2111) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2113 = "tosa.reshape"(%2110) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2114 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %2115 = "linalg.matmul"(%2113, %2112, %2114) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2116 = "tosa.reshape"(%2115) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2117 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2118 = "tosa.transpose"(%arg230, %2117) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %2119 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2120 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %2121 = "linalg.matmul"(%2119, %2118, %2120) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %2122 = "tosa.reshape"(%2121) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %2123 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2124 = "tosa.transpose"(%arg232, %2123) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %2125 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2126 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %2127 = "linalg.matmul"(%2125, %2124, %2126) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %2128 = "tosa.reshape"(%2127) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %2129 = "tosa.reshape"(%2116) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %2130 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2131 = "tosa.transpose"(%2129, %2130) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %2132 = "tosa.reshape"(%2122) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %2133 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2134 = "tosa.transpose"(%2132, %2133) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %2135 = "tosa.reshape"(%2128) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %2136 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2137 = "tosa.transpose"(%2135, %2136) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %2138 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x12xf32>}> : () -> tensor<256x12xf32>
    %2139 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2140 = "tosa.transpose"(%2134, %2139) : (tensor<1x20x12x64xf32>, tensor<4xi32>) -> tensor<1x20x64x12xf32>
    %2141 = "tosa.reshape"(%2131) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %2142 = "tosa.reshape"(%2140) <{new_shape = array<i64: 20, 64, 12>}> : (tensor<1x20x64x12xf32>) -> tensor<20x64x12xf32>
    %2143 = "tosa.matmul"(%2141, %2142) : (tensor<20x256x64xf32>, tensor<20x64x12xf32>) -> tensor<20x256x12xf32>
    %2144 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x256x12xf32>}> : () -> tensor<20x256x12xf32>
    %2145 = "tosa.mul"(%2143, %2144) <{shift = 0 : i8}> : (tensor<20x256x12xf32>, tensor<20x256x12xf32>) -> tensor<20x256x12xf32>
    %2146 = "tosa.add"(%2145, %2138) : (tensor<20x256x12xf32>, tensor<256x12xf32>) -> tensor<20x256x12xf32>
    %2147 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x256x12xf32>}> : () -> tensor<20x256x12xf32>
    %2148 = "linalg.softmax"(%2146, %2147) <{dimension = 3 : i64}> : (tensor<20x256x12xf32>, tensor<20x256x12xf32>) -> tensor<20x256x12xf32>
    %2149 = "tosa.reshape"(%2137) <{new_shape = array<i64: 20, 12, 64>}> : (tensor<1x20x12x64xf32>) -> tensor<20x12x64xf32>
    %2150 = "tosa.matmul"(%2148, %2149) : (tensor<20x256x12xf32>, tensor<20x12x64xf32>) -> tensor<20x256x64xf32>
    %2151 = "tosa.reshape"(%2150) <{new_shape = array<i64: 1, 20, 256, 64>}> : (tensor<20x256x64xf32>) -> tensor<1x20x256x64xf32>
    %2152 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2153 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2154 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2155 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2156 = "tosa.transpose"(%2151, %2155) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x256x20x64xf32>
    %2157 = "tosa.reshape"(%2156) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x256x20x64xf32>) -> tensor<1x256x1280xf32>
    %2158 = "tosa.reshape"(%2157) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2159 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2160 = "tosa.transpose"(%arg234, %2159) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2161 = "tosa.reshape"(%2158) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2162 = "tosa.reshape"(%2160) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %2163 = "tosa.matmul"(%2161, %2162) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %2164 = "tosa.reshape"(%2163) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2165 = "tosa.reshape"(%arg235) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2166 = "tosa.add"(%2165, %2164) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2167 = "tosa.reshape"(%2166) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2168 = "tosa.identity"(%2167) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2169 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x1280xf32>}> : () -> tensor<1x256x1280xf32>
    %2170 = "tosa.reciprocal"(%2169) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2171 = "tosa.mul"(%2168, %2170) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2172 = "tosa.add"(%2171, %2091) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2173 = "tosa.reduce_sum"(%2172) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %2174 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2175 = "tosa.reciprocal"(%2174) : (tensor<1xf32>) -> tensor<1xf32>
    %2176 = "tosa.mul"(%2175, %2173) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2177 = "tosa.sub"(%2172, %2176) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2178 = "tosa.mul"(%2177, %2177) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2179 = "tosa.reduce_sum"(%2178) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %2180 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2181 = "tosa.reciprocal"(%2180) : (tensor<1xf32>) -> tensor<1xf32>
    %2182 = "tosa.mul"(%2181, %2179) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2183 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %2184 = "tosa.add"(%2182, %2183) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2185 = "tosa.rsqrt"(%2184) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %2186 = "tosa.sub"(%2172, %2176) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2187 = "tosa.mul"(%2186, %2185) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %2188 = "tosa.reshape"(%arg236) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2189 = "tosa.mul"(%2187, %2188) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %2190 = "tosa.reshape"(%arg237) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2191 = "tosa.add"(%2189, %2190) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %2192 = "tosa.reshape"(%2191) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2193 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2194 = "tosa.transpose"(%arg238, %2193) : (tensor<10240x1280xf32>, tensor<2xi32>) -> tensor<1280x10240xf32>
    %2195 = "tosa.reshape"(%2192) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2196 = "tosa.reshape"(%2194) <{new_shape = array<i64: 1, 1280, 10240>}> : (tensor<1280x10240xf32>) -> tensor<1x1280x10240xf32>
    %2197 = "tosa.matmul"(%2195, %2196) : (tensor<1x256x1280xf32>, tensor<1x1280x10240xf32>) -> tensor<1x256x10240xf32>
    %2198 = "tosa.reshape"(%2197) <{new_shape = array<i64: 256, 10240>}> : (tensor<1x256x10240xf32>) -> tensor<256x10240xf32>
    %2199 = "tosa.reshape"(%arg239) <{new_shape = array<i64: 1, 10240>}> : (tensor<10240xf32>) -> tensor<1x10240xf32>
    %2200 = "tosa.add"(%2199, %2198) : (tensor<1x10240xf32>, tensor<256x10240xf32>) -> tensor<256x10240xf32>
    %2201 = "tosa.reshape"(%2200) <{new_shape = array<i64: 1, 256, 10240>}> : (tensor<256x10240xf32>) -> tensor<1x256x10240xf32>
    %2202 = "tosa.slice"(%2201) <{size = array<i64: 0, 0, 5120>, start = array<i64: 0, 0, 0>}> : (tensor<1x256x10240xf32>) -> tensor<1x256x5120xf32>
    %2203 = "tosa.slice"(%2201) <{size = array<i64: 0, 0, 10240>, start = array<i64: 0, 0, 5120>}> : (tensor<1x256x10240xf32>) -> tensor<1x256x5120xf32>
    %2204 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %2205 = "tosa.mul"(%2203, %2204) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %2206 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %2207 = "tosa.mul"(%2203, %2206) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %2208 = "math.erf"(%2207) <{fastmath = #arith.fastmath<none>}> : (tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %2209 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %2210 = "tosa.add"(%2208, %2209) : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %2211 = "tosa.mul"(%2205, %2210) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %2212 = "tosa.mul"(%2202, %2211) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %2213 = "tosa.identity"(%2212) : (tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %2214 = "tosa.reshape"(%2213) <{new_shape = array<i64: 256, 5120>}> : (tensor<1x256x5120xf32>) -> tensor<256x5120xf32>
    %2215 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2216 = "tosa.transpose"(%arg240, %2215) : (tensor<1280x5120xf32>, tensor<2xi32>) -> tensor<5120x1280xf32>
    %2217 = "tosa.reshape"(%2214) <{new_shape = array<i64: 1, 256, 5120>}> : (tensor<256x5120xf32>) -> tensor<1x256x5120xf32>
    %2218 = "tosa.reshape"(%2216) <{new_shape = array<i64: 1, 5120, 1280>}> : (tensor<5120x1280xf32>) -> tensor<1x5120x1280xf32>
    %2219 = "tosa.matmul"(%2217, %2218) : (tensor<1x256x5120xf32>, tensor<1x5120x1280xf32>) -> tensor<1x256x1280xf32>
    %2220 = "tosa.reshape"(%2219) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2221 = "tosa.reshape"(%arg241) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2222 = "tosa.add"(%2221, %2220) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2223 = "tosa.reshape"(%2222) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2224 = "tosa.add"(%2223, %2172) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %2225 = "tosa.reshape"(%2224) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2226 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2227 = "tosa.transpose"(%arg242, %2226) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2228 = "tosa.reshape"(%2225) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2229 = "tosa.reshape"(%2227) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %2230 = "tosa.matmul"(%2228, %2229) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %2231 = "tosa.reshape"(%2230) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %2232 = "tosa.reshape"(%arg243) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2233 = "tosa.add"(%2232, %2231) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %2234 = "tosa.reshape"(%2233) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %2235 = "tosa.reshape"(%2234) <{new_shape = array<i64: 1, 16, 16, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<1x16x16x1280xf32>
    %2236 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2237 = "tosa.transpose"(%2235, %2236) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %2238 = "tosa.identity"(%2237) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %2239 = "tosa.add"(%2238, %1972) : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %2240 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2241 = "tosa.transpose"(%2239, %2240) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %2242 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2243 = "tosa.transpose"(%arg244, %2242) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %2244 = "tosa.conv2d"(%2241, %2243, %arg245) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<1x16x16x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2245 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2246 = "tosa.transpose"(%2244, %2245) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2247 = "tosa.reshape"(%2246) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %2248 = "tosa.reduce_sum"(%2247) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2249 = "tosa.reduce_sum"(%2248) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2250 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2251 = "tosa.reciprocal"(%2250) : (tensor<1xf32>) -> tensor<1xf32>
    %2252 = "tosa.mul"(%2251, %2249) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2253 = "tosa.sub"(%2247, %2252) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2254 = "tosa.mul"(%2253, %2253) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2255 = "tosa.reduce_sum"(%2254) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2256 = "tosa.reduce_sum"(%2255) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2257 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2258 = "tosa.reciprocal"(%2257) : (tensor<1xf32>) -> tensor<1xf32>
    %2259 = "tosa.mul"(%2258, %2256) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2260 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %2261 = "tosa.add"(%2259, %2260) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2262 = "tosa.rsqrt"(%2261) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2263 = "tosa.sub"(%2247, %2252) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2264 = "tosa.mul"(%2263, %2262) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2265 = "tosa.reshape"(%2264) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %2266 = "tosa.reshape"(%arg246) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2267 = "tosa.reshape"(%2266) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2268 = "tosa.reshape"(%2267) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2269 = "tosa.reshape"(%arg247) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2270 = "tosa.reshape"(%2269) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2271 = "tosa.reshape"(%2270) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2272 = "tosa.mul"(%2265, %2271) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2273 = "tosa.add"(%2272, %2268) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2274 = "tosa.sigmoid"(%2273) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2275 = "tosa.mul"(%2273, %2274) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2276 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2277 = "tosa.transpose"(%2275, %2276) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %2278 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2279 = "tosa.transpose"(%arg248, %2278) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %2280 = "tosa.conv2d"(%2277, %2279, %arg249) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2281 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2282 = "tosa.transpose"(%2280, %2281) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2283 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2284 = "tosa.mul"(%50, %2283) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2285 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2286 = "tosa.transpose"(%arg250, %2285) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2287 = "tosa.reshape"(%2284) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %2288 = "tosa.reshape"(%2286) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %2289 = "tosa.matmul"(%2287, %2288) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %2290 = "tosa.reshape"(%2289) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %2291 = "tosa.reshape"(%arg251) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2292 = "tosa.add"(%2291, %2290) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2293 = "tensor.extract_slice"(%2292) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2294 = "tensor.extract_slice"(%2293) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2295 = "tosa.reshape"(%2294) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2296 = "tosa.reshape"(%2295) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2297 = "tosa.add"(%2282, %2296) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2298 = "tosa.reshape"(%2297) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %2299 = "tosa.reduce_sum"(%2298) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2300 = "tosa.reduce_sum"(%2299) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2301 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2302 = "tosa.reciprocal"(%2301) : (tensor<1xf32>) -> tensor<1xf32>
    %2303 = "tosa.mul"(%2302, %2300) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2304 = "tosa.sub"(%2298, %2303) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2305 = "tosa.mul"(%2304, %2304) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2306 = "tosa.reduce_sum"(%2305) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2307 = "tosa.reduce_sum"(%2306) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2308 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2309 = "tosa.reciprocal"(%2308) : (tensor<1xf32>) -> tensor<1xf32>
    %2310 = "tosa.mul"(%2309, %2307) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2311 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %2312 = "tosa.add"(%2310, %2311) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2313 = "tosa.rsqrt"(%2312) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2314 = "tosa.sub"(%2298, %2303) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2315 = "tosa.mul"(%2314, %2313) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2316 = "tosa.reshape"(%2315) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %2317 = "tosa.reshape"(%arg252) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2318 = "tosa.reshape"(%2317) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2319 = "tosa.reshape"(%2318) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2320 = "tosa.reshape"(%arg253) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2321 = "tosa.reshape"(%2320) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2322 = "tosa.reshape"(%2321) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2323 = "tosa.mul"(%2316, %2322) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2324 = "tosa.add"(%2323, %2319) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2325 = "tosa.sigmoid"(%2324) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2326 = "tosa.mul"(%2324, %2325) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2327 = "tosa.identity"(%2326) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2328 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2329 = "tosa.transpose"(%2327, %2328) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %2330 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2331 = "tosa.transpose"(%arg254, %2330) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %2332 = "tosa.conv2d"(%2329, %2331, %arg255) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2333 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2334 = "tosa.transpose"(%2332, %2333) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2335 = "tosa.add"(%2246, %2334) : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2336 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x8x8xf32>}> : () -> tensor<1x1280x8x8xf32>
    %2337 = "tosa.reciprocal"(%2336) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2338 = "tosa.mul"(%2335, %2337) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2339 = "tosa.reshape"(%2338) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %2340 = "tosa.reduce_sum"(%2339) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2341 = "tosa.reduce_sum"(%2340) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2342 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2343 = "tosa.reciprocal"(%2342) : (tensor<1xf32>) -> tensor<1xf32>
    %2344 = "tosa.mul"(%2343, %2341) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2345 = "tosa.sub"(%2339, %2344) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2346 = "tosa.mul"(%2345, %2345) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2347 = "tosa.reduce_sum"(%2346) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2348 = "tosa.reduce_sum"(%2347) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2349 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2350 = "tosa.reciprocal"(%2349) : (tensor<1xf32>) -> tensor<1xf32>
    %2351 = "tosa.mul"(%2350, %2348) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2352 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %2353 = "tosa.add"(%2351, %2352) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2354 = "tosa.rsqrt"(%2353) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2355 = "tosa.sub"(%2339, %2344) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2356 = "tosa.mul"(%2355, %2354) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2357 = "tosa.reshape"(%2356) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %2358 = "tosa.reshape"(%arg256) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2359 = "tosa.reshape"(%2358) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2360 = "tosa.reshape"(%2359) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2361 = "tosa.reshape"(%arg257) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2362 = "tosa.reshape"(%2361) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2363 = "tosa.reshape"(%2362) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2364 = "tosa.mul"(%2357, %2363) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2365 = "tosa.add"(%2364, %2360) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2366 = "tosa.sigmoid"(%2365) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2367 = "tosa.mul"(%2365, %2366) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2368 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2369 = "tosa.transpose"(%2367, %2368) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %2370 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2371 = "tosa.transpose"(%arg258, %2370) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %2372 = "tosa.conv2d"(%2369, %2371, %arg259) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2373 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2374 = "tosa.transpose"(%2372, %2373) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2375 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2376 = "tosa.mul"(%50, %2375) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2377 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2378 = "tosa.transpose"(%arg260, %2377) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2379 = "tosa.reshape"(%2376) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %2380 = "tosa.reshape"(%2378) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %2381 = "tosa.matmul"(%2379, %2380) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %2382 = "tosa.reshape"(%2381) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %2383 = "tosa.reshape"(%arg261) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2384 = "tosa.add"(%2383, %2382) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2385 = "tensor.extract_slice"(%2384) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2386 = "tensor.extract_slice"(%2385) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2387 = "tosa.reshape"(%2386) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2388 = "tosa.reshape"(%2387) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2389 = "tosa.add"(%2374, %2388) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2390 = "tosa.reshape"(%2389) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %2391 = "tosa.reduce_sum"(%2390) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2392 = "tosa.reduce_sum"(%2391) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2393 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2394 = "tosa.reciprocal"(%2393) : (tensor<1xf32>) -> tensor<1xf32>
    %2395 = "tosa.mul"(%2394, %2392) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2396 = "tosa.sub"(%2390, %2395) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2397 = "tosa.mul"(%2396, %2396) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2398 = "tosa.reduce_sum"(%2397) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2399 = "tosa.reduce_sum"(%2398) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2400 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2401 = "tosa.reciprocal"(%2400) : (tensor<1xf32>) -> tensor<1xf32>
    %2402 = "tosa.mul"(%2401, %2399) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2403 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %2404 = "tosa.add"(%2402, %2403) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2405 = "tosa.rsqrt"(%2404) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2406 = "tosa.sub"(%2390, %2395) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2407 = "tosa.mul"(%2406, %2405) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2408 = "tosa.reshape"(%2407) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %2409 = "tosa.reshape"(%arg262) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2410 = "tosa.reshape"(%2409) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2411 = "tosa.reshape"(%2410) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2412 = "tosa.reshape"(%arg263) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2413 = "tosa.reshape"(%2412) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2414 = "tosa.reshape"(%2413) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2415 = "tosa.mul"(%2408, %2414) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2416 = "tosa.add"(%2415, %2411) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2417 = "tosa.sigmoid"(%2416) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2418 = "tosa.mul"(%2416, %2417) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2419 = "tosa.identity"(%2418) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2420 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2421 = "tosa.transpose"(%2419, %2420) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %2422 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2423 = "tosa.transpose"(%arg264, %2422) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %2424 = "tosa.conv2d"(%2421, %2423, %arg265) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2425 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2426 = "tosa.transpose"(%2424, %2425) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2427 = "tosa.add"(%2338, %2426) : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2428 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x8x8xf32>}> : () -> tensor<1x1280x8x8xf32>
    %2429 = "tosa.reciprocal"(%2428) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2430 = "tosa.mul"(%2427, %2429) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2431 = "tosa.reshape"(%2430) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %2432 = "tosa.reduce_sum"(%2431) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2433 = "tosa.reduce_sum"(%2432) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2434 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2435 = "tosa.reciprocal"(%2434) : (tensor<1xf32>) -> tensor<1xf32>
    %2436 = "tosa.mul"(%2435, %2433) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2437 = "tosa.sub"(%2431, %2436) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2438 = "tosa.mul"(%2437, %2437) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2439 = "tosa.reduce_sum"(%2438) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2440 = "tosa.reduce_sum"(%2439) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2441 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2442 = "tosa.reciprocal"(%2441) : (tensor<1xf32>) -> tensor<1xf32>
    %2443 = "tosa.mul"(%2442, %2440) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2444 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %2445 = "tosa.add"(%2443, %2444) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2446 = "tosa.rsqrt"(%2445) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2447 = "tosa.sub"(%2431, %2436) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2448 = "tosa.mul"(%2447, %2446) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2449 = "tosa.reshape"(%2448) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %2450 = "tosa.reshape"(%arg266) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2451 = "tosa.reshape"(%2450) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2452 = "tosa.reshape"(%2451) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2453 = "tosa.reshape"(%arg267) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2454 = "tosa.reshape"(%2453) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2455 = "tosa.reshape"(%2454) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2456 = "tosa.mul"(%2449, %2455) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2457 = "tosa.add"(%2456, %2452) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2458 = "tosa.sigmoid"(%2457) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2459 = "tosa.mul"(%2457, %2458) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2460 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2461 = "tosa.transpose"(%2459, %2460) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %2462 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2463 = "tosa.transpose"(%arg268, %2462) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %2464 = "tosa.conv2d"(%2461, %2463, %arg269) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2465 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2466 = "tosa.transpose"(%2464, %2465) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2467 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2468 = "tosa.mul"(%50, %2467) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2469 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2470 = "tosa.transpose"(%arg270, %2469) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2471 = "tosa.reshape"(%2468) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %2472 = "tosa.reshape"(%2470) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %2473 = "tosa.matmul"(%2471, %2472) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %2474 = "tosa.reshape"(%2473) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %2475 = "tosa.reshape"(%arg271) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2476 = "tosa.add"(%2475, %2474) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2477 = "tensor.extract_slice"(%2476) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2478 = "tensor.extract_slice"(%2477) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2479 = "tosa.reshape"(%2478) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2480 = "tosa.reshape"(%2479) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2481 = "tosa.add"(%2466, %2480) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2482 = "tosa.reshape"(%2481) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %2483 = "tosa.reduce_sum"(%2482) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2484 = "tosa.reduce_sum"(%2483) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2485 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2486 = "tosa.reciprocal"(%2485) : (tensor<1xf32>) -> tensor<1xf32>
    %2487 = "tosa.mul"(%2486, %2484) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2488 = "tosa.sub"(%2482, %2487) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2489 = "tosa.mul"(%2488, %2488) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2490 = "tosa.reduce_sum"(%2489) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2491 = "tosa.reduce_sum"(%2490) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2492 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2493 = "tosa.reciprocal"(%2492) : (tensor<1xf32>) -> tensor<1xf32>
    %2494 = "tosa.mul"(%2493, %2491) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2495 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %2496 = "tosa.add"(%2494, %2495) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2497 = "tosa.rsqrt"(%2496) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2498 = "tosa.sub"(%2482, %2487) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2499 = "tosa.mul"(%2498, %2497) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2500 = "tosa.reshape"(%2499) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %2501 = "tosa.reshape"(%arg272) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2502 = "tosa.reshape"(%2501) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2503 = "tosa.reshape"(%2502) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2504 = "tosa.reshape"(%arg273) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2505 = "tosa.reshape"(%2504) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2506 = "tosa.reshape"(%2505) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2507 = "tosa.mul"(%2500, %2506) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2508 = "tosa.add"(%2507, %2503) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2509 = "tosa.sigmoid"(%2508) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2510 = "tosa.mul"(%2508, %2509) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2511 = "tosa.identity"(%2510) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2512 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2513 = "tosa.transpose"(%2511, %2512) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %2514 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2515 = "tosa.transpose"(%arg274, %2514) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %2516 = "tosa.conv2d"(%2513, %2515, %arg275) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2517 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2518 = "tosa.transpose"(%2516, %2517) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2519 = "tosa.add"(%2430, %2518) : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2520 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x8x8xf32>}> : () -> tensor<1x1280x8x8xf32>
    %2521 = "tosa.reciprocal"(%2520) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2522 = "tosa.mul"(%2519, %2521) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2523 = "tosa.reshape"(%2522) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %2524 = "tosa.reduce_sum"(%2523) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2525 = "tosa.reduce_sum"(%2524) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2526 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2527 = "tosa.reciprocal"(%2526) : (tensor<1xf32>) -> tensor<1xf32>
    %2528 = "tosa.mul"(%2527, %2525) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2529 = "tosa.sub"(%2523, %2528) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2530 = "tosa.mul"(%2529, %2529) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2531 = "tosa.reduce_sum"(%2530) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2532 = "tosa.reduce_sum"(%2531) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2533 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2534 = "tosa.reciprocal"(%2533) : (tensor<1xf32>) -> tensor<1xf32>
    %2535 = "tosa.mul"(%2534, %2532) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2536 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %2537 = "tosa.add"(%2535, %2536) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2538 = "tosa.rsqrt"(%2537) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2539 = "tosa.sub"(%2523, %2528) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2540 = "tosa.mul"(%2539, %2538) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2541 = "tosa.reshape"(%2540) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %2542 = "tosa.reshape"(%arg276) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2543 = "tosa.reshape"(%2542) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2544 = "tosa.reshape"(%2543) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2545 = "tosa.reshape"(%arg277) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2546 = "tosa.reshape"(%2545) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2547 = "tosa.reshape"(%2546) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2548 = "tosa.mul"(%2541, %2547) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2549 = "tosa.add"(%2548, %2544) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2550 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2551 = "tosa.transpose"(%2549, %2550) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %2552 = "tosa.reshape"(%2551) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<1x8x8x1280xf32>) -> tensor<1x64x1280xf32>
    %2553 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2554 = "tosa.transpose"(%arg278, %2553) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2555 = "tosa.reshape"(%2552) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2556 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64x1280xf32>}> : () -> tensor<64x1280xf32>
    %2557 = "linalg.matmul"(%2555, %2554, %2556) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<64x1280xf32>, tensor<1280x1280xf32>, tensor<64x1280xf32>) -> tensor<64x1280xf32>
    %2558 = "tosa.reshape"(%2557) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2559 = "tosa.reshape"(%arg279) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2560 = "tosa.add"(%2558, %2559) : (tensor<1x64x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x64x1280xf32>
    %2561 = "tosa.reduce_sum"(%2560) <{axis = 2 : i32}> : (tensor<1x64x1280xf32>) -> tensor<1x64x1xf32>
    %2562 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2563 = "tosa.reciprocal"(%2562) : (tensor<1xf32>) -> tensor<1xf32>
    %2564 = "tosa.mul"(%2563, %2561) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2565 = "tosa.sub"(%2560, %2564) : (tensor<1x64x1280xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1280xf32>
    %2566 = "tosa.mul"(%2565, %2565) <{shift = 0 : i8}> : (tensor<1x64x1280xf32>, tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2567 = "tosa.reduce_sum"(%2566) <{axis = 2 : i32}> : (tensor<1x64x1280xf32>) -> tensor<1x64x1xf32>
    %2568 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2569 = "tosa.reciprocal"(%2568) : (tensor<1xf32>) -> tensor<1xf32>
    %2570 = "tosa.mul"(%2569, %2567) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2571 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x64x1xf32>}> : () -> tensor<1x64x1xf32>
    %2572 = "tosa.add"(%2570, %2571) : (tensor<1x64x1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2573 = "tosa.rsqrt"(%2572) : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2574 = "tosa.sub"(%2560, %2564) : (tensor<1x64x1280xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1280xf32>
    %2575 = "tosa.mul"(%2574, %2573) <{shift = 0 : i8}> : (tensor<1x64x1280xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1280xf32>
    %2576 = "tosa.reshape"(%arg280) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2577 = "tosa.mul"(%2575, %2576) <{shift = 0 : i8}> : (tensor<1x64x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x64x1280xf32>
    %2578 = "tosa.reshape"(%arg281) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2579 = "tosa.add"(%2577, %2578) : (tensor<1x64x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x64x1280xf32>
    %2580 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2581 = "tosa.transpose"(%arg282, %2580) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2582 = "tosa.reshape"(%2579) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2583 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64x1280xf32>}> : () -> tensor<64x1280xf32>
    %2584 = "linalg.matmul"(%2582, %2581, %2583) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<64x1280xf32>, tensor<1280x1280xf32>, tensor<64x1280xf32>) -> tensor<64x1280xf32>
    %2585 = "tosa.reshape"(%2584) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2586 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2587 = "tosa.transpose"(%arg283, %2586) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2588 = "tosa.reshape"(%2579) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2589 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64x1280xf32>}> : () -> tensor<64x1280xf32>
    %2590 = "linalg.matmul"(%2588, %2587, %2589) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<64x1280xf32>, tensor<1280x1280xf32>, tensor<64x1280xf32>) -> tensor<64x1280xf32>
    %2591 = "tosa.reshape"(%2590) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2592 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2593 = "tosa.transpose"(%arg284, %2592) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2594 = "tosa.reshape"(%2579) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2595 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64x1280xf32>}> : () -> tensor<64x1280xf32>
    %2596 = "linalg.matmul"(%2594, %2593, %2595) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<64x1280xf32>, tensor<1280x1280xf32>, tensor<64x1280xf32>) -> tensor<64x1280xf32>
    %2597 = "tosa.reshape"(%2596) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2598 = "tosa.reshape"(%2585) <{new_shape = array<i64: 1, 64, 20, 64>}> : (tensor<1x64x1280xf32>) -> tensor<1x64x20x64xf32>
    %2599 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2600 = "tosa.transpose"(%2598, %2599) : (tensor<1x64x20x64xf32>, tensor<4xi32>) -> tensor<1x20x64x64xf32>
    %2601 = "tosa.reshape"(%2591) <{new_shape = array<i64: 1, 64, 20, 64>}> : (tensor<1x64x1280xf32>) -> tensor<1x64x20x64xf32>
    %2602 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2603 = "tosa.transpose"(%2601, %2602) : (tensor<1x64x20x64xf32>, tensor<4xi32>) -> tensor<1x20x64x64xf32>
    %2604 = "tosa.reshape"(%2597) <{new_shape = array<i64: 1, 64, 20, 64>}> : (tensor<1x64x1280xf32>) -> tensor<1x64x20x64xf32>
    %2605 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2606 = "tosa.transpose"(%2604, %2605) : (tensor<1x64x20x64xf32>, tensor<4xi32>) -> tensor<1x20x64x64xf32>
    %2607 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64x64xf32>}> : () -> tensor<64x64xf32>
    %2608 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2609 = "tosa.transpose"(%2603, %2608) : (tensor<1x20x64x64xf32>, tensor<4xi32>) -> tensor<1x20x64x64xf32>
    %2610 = "tosa.reshape"(%2600) <{new_shape = array<i64: 20, 64, 64>}> : (tensor<1x20x64x64xf32>) -> tensor<20x64x64xf32>
    %2611 = "tosa.reshape"(%2609) <{new_shape = array<i64: 20, 64, 64>}> : (tensor<1x20x64x64xf32>) -> tensor<20x64x64xf32>
    %2612 = "tosa.matmul"(%2610, %2611) : (tensor<20x64x64xf32>, tensor<20x64x64xf32>) -> tensor<20x64x64xf32>
    %2613 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x64x64xf32>}> : () -> tensor<20x64x64xf32>
    %2614 = "tosa.mul"(%2612, %2613) <{shift = 0 : i8}> : (tensor<20x64x64xf32>, tensor<20x64x64xf32>) -> tensor<20x64x64xf32>
    %2615 = "tosa.add"(%2614, %2607) : (tensor<20x64x64xf32>, tensor<64x64xf32>) -> tensor<20x64x64xf32>
    %2616 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x64x64xf32>}> : () -> tensor<20x64x64xf32>
    %2617 = "linalg.softmax"(%2615, %2616) <{dimension = 3 : i64}> : (tensor<20x64x64xf32>, tensor<20x64x64xf32>) -> tensor<20x64x64xf32>
    %2618 = "tosa.reshape"(%2606) <{new_shape = array<i64: 20, 64, 64>}> : (tensor<1x20x64x64xf32>) -> tensor<20x64x64xf32>
    %2619 = "tosa.matmul"(%2617, %2618) : (tensor<20x64x64xf32>, tensor<20x64x64xf32>) -> tensor<20x64x64xf32>
    %2620 = "tosa.reshape"(%2619) <{new_shape = array<i64: 1, 20, 64, 64>}> : (tensor<20x64x64xf32>) -> tensor<1x20x64x64xf32>
    %2621 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2622 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2623 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2624 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2625 = "tosa.transpose"(%2620, %2624) : (tensor<1x20x64x64xf32>, tensor<4xi32>) -> tensor<1x64x20x64xf32>
    %2626 = "tosa.reshape"(%2625) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<1x64x20x64xf32>) -> tensor<1x64x1280xf32>
    %2627 = "tosa.reshape"(%2626) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2628 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2629 = "tosa.transpose"(%arg285, %2628) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2630 = "tosa.reshape"(%2627) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2631 = "tosa.reshape"(%2629) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %2632 = "tosa.matmul"(%2630, %2631) : (tensor<1x64x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x64x1280xf32>
    %2633 = "tosa.reshape"(%2632) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2634 = "tosa.reshape"(%arg286) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2635 = "tosa.add"(%2634, %2633) : (tensor<1x1280xf32>, tensor<64x1280xf32>) -> tensor<64x1280xf32>
    %2636 = "tosa.reshape"(%2635) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2637 = "tosa.identity"(%2636) : (tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2638 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x64x1280xf32>}> : () -> tensor<1x64x1280xf32>
    %2639 = "tosa.reciprocal"(%2638) : (tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2640 = "tosa.mul"(%2637, %2639) <{shift = 0 : i8}> : (tensor<1x64x1280xf32>, tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2641 = "tosa.add"(%2640, %2560) : (tensor<1x64x1280xf32>, tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2642 = "tosa.reduce_sum"(%2641) <{axis = 2 : i32}> : (tensor<1x64x1280xf32>) -> tensor<1x64x1xf32>
    %2643 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2644 = "tosa.reciprocal"(%2643) : (tensor<1xf32>) -> tensor<1xf32>
    %2645 = "tosa.mul"(%2644, %2642) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2646 = "tosa.sub"(%2641, %2645) : (tensor<1x64x1280xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1280xf32>
    %2647 = "tosa.mul"(%2646, %2646) <{shift = 0 : i8}> : (tensor<1x64x1280xf32>, tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2648 = "tosa.reduce_sum"(%2647) <{axis = 2 : i32}> : (tensor<1x64x1280xf32>) -> tensor<1x64x1xf32>
    %2649 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2650 = "tosa.reciprocal"(%2649) : (tensor<1xf32>) -> tensor<1xf32>
    %2651 = "tosa.mul"(%2650, %2648) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2652 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x64x1xf32>}> : () -> tensor<1x64x1xf32>
    %2653 = "tosa.add"(%2651, %2652) : (tensor<1x64x1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2654 = "tosa.rsqrt"(%2653) : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2655 = "tosa.sub"(%2641, %2645) : (tensor<1x64x1280xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1280xf32>
    %2656 = "tosa.mul"(%2655, %2654) <{shift = 0 : i8}> : (tensor<1x64x1280xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1280xf32>
    %2657 = "tosa.reshape"(%arg287) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2658 = "tosa.mul"(%2656, %2657) <{shift = 0 : i8}> : (tensor<1x64x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x64x1280xf32>
    %2659 = "tosa.reshape"(%arg288) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2660 = "tosa.add"(%2658, %2659) : (tensor<1x64x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x64x1280xf32>
    %2661 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2662 = "tosa.transpose"(%arg289, %2661) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2663 = "tosa.reshape"(%2660) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2664 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64x1280xf32>}> : () -> tensor<64x1280xf32>
    %2665 = "linalg.matmul"(%2663, %2662, %2664) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<64x1280xf32>, tensor<1280x1280xf32>, tensor<64x1280xf32>) -> tensor<64x1280xf32>
    %2666 = "tosa.reshape"(%2665) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2667 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2668 = "tosa.transpose"(%arg290, %2667) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %2669 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2670 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %2671 = "linalg.matmul"(%2669, %2668, %2670) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %2672 = "tosa.reshape"(%2671) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %2673 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2674 = "tosa.transpose"(%arg292, %2673) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %2675 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %2676 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %2677 = "linalg.matmul"(%2675, %2674, %2676) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %2678 = "tosa.reshape"(%2677) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %2679 = "tosa.reshape"(%2666) <{new_shape = array<i64: 1, 64, 20, 64>}> : (tensor<1x64x1280xf32>) -> tensor<1x64x20x64xf32>
    %2680 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2681 = "tosa.transpose"(%2679, %2680) : (tensor<1x64x20x64xf32>, tensor<4xi32>) -> tensor<1x20x64x64xf32>
    %2682 = "tosa.reshape"(%2672) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %2683 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2684 = "tosa.transpose"(%2682, %2683) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %2685 = "tosa.reshape"(%2678) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %2686 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2687 = "tosa.transpose"(%2685, %2686) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %2688 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64x12xf32>}> : () -> tensor<64x12xf32>
    %2689 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2690 = "tosa.transpose"(%2684, %2689) : (tensor<1x20x12x64xf32>, tensor<4xi32>) -> tensor<1x20x64x12xf32>
    %2691 = "tosa.reshape"(%2681) <{new_shape = array<i64: 20, 64, 64>}> : (tensor<1x20x64x64xf32>) -> tensor<20x64x64xf32>
    %2692 = "tosa.reshape"(%2690) <{new_shape = array<i64: 20, 64, 12>}> : (tensor<1x20x64x12xf32>) -> tensor<20x64x12xf32>
    %2693 = "tosa.matmul"(%2691, %2692) : (tensor<20x64x64xf32>, tensor<20x64x12xf32>) -> tensor<20x64x12xf32>
    %2694 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x64x12xf32>}> : () -> tensor<20x64x12xf32>
    %2695 = "tosa.mul"(%2693, %2694) <{shift = 0 : i8}> : (tensor<20x64x12xf32>, tensor<20x64x12xf32>) -> tensor<20x64x12xf32>
    %2696 = "tosa.add"(%2695, %2688) : (tensor<20x64x12xf32>, tensor<64x12xf32>) -> tensor<20x64x12xf32>
    %2697 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x64x12xf32>}> : () -> tensor<20x64x12xf32>
    %2698 = "linalg.softmax"(%2696, %2697) <{dimension = 3 : i64}> : (tensor<20x64x12xf32>, tensor<20x64x12xf32>) -> tensor<20x64x12xf32>
    %2699 = "tosa.reshape"(%2687) <{new_shape = array<i64: 20, 12, 64>}> : (tensor<1x20x12x64xf32>) -> tensor<20x12x64xf32>
    %2700 = "tosa.matmul"(%2698, %2699) : (tensor<20x64x12xf32>, tensor<20x12x64xf32>) -> tensor<20x64x64xf32>
    %2701 = "tosa.reshape"(%2700) <{new_shape = array<i64: 1, 20, 64, 64>}> : (tensor<20x64x64xf32>) -> tensor<1x20x64x64xf32>
    %2702 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2703 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2704 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %2705 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2706 = "tosa.transpose"(%2701, %2705) : (tensor<1x20x64x64xf32>, tensor<4xi32>) -> tensor<1x64x20x64xf32>
    %2707 = "tosa.reshape"(%2706) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<1x64x20x64xf32>) -> tensor<1x64x1280xf32>
    %2708 = "tosa.reshape"(%2707) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2709 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2710 = "tosa.transpose"(%arg294, %2709) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2711 = "tosa.reshape"(%2708) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2712 = "tosa.reshape"(%2710) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %2713 = "tosa.matmul"(%2711, %2712) : (tensor<1x64x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x64x1280xf32>
    %2714 = "tosa.reshape"(%2713) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2715 = "tosa.reshape"(%arg295) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2716 = "tosa.add"(%2715, %2714) : (tensor<1x1280xf32>, tensor<64x1280xf32>) -> tensor<64x1280xf32>
    %2717 = "tosa.reshape"(%2716) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2718 = "tosa.identity"(%2717) : (tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2719 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x64x1280xf32>}> : () -> tensor<1x64x1280xf32>
    %2720 = "tosa.reciprocal"(%2719) : (tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2721 = "tosa.mul"(%2718, %2720) <{shift = 0 : i8}> : (tensor<1x64x1280xf32>, tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2722 = "tosa.add"(%2721, %2641) : (tensor<1x64x1280xf32>, tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2723 = "tosa.reduce_sum"(%2722) <{axis = 2 : i32}> : (tensor<1x64x1280xf32>) -> tensor<1x64x1xf32>
    %2724 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2725 = "tosa.reciprocal"(%2724) : (tensor<1xf32>) -> tensor<1xf32>
    %2726 = "tosa.mul"(%2725, %2723) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2727 = "tosa.sub"(%2722, %2726) : (tensor<1x64x1280xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1280xf32>
    %2728 = "tosa.mul"(%2727, %2727) <{shift = 0 : i8}> : (tensor<1x64x1280xf32>, tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2729 = "tosa.reduce_sum"(%2728) <{axis = 2 : i32}> : (tensor<1x64x1280xf32>) -> tensor<1x64x1xf32>
    %2730 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2731 = "tosa.reciprocal"(%2730) : (tensor<1xf32>) -> tensor<1xf32>
    %2732 = "tosa.mul"(%2731, %2729) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2733 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x64x1xf32>}> : () -> tensor<1x64x1xf32>
    %2734 = "tosa.add"(%2732, %2733) : (tensor<1x64x1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2735 = "tosa.rsqrt"(%2734) : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %2736 = "tosa.sub"(%2722, %2726) : (tensor<1x64x1280xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1280xf32>
    %2737 = "tosa.mul"(%2736, %2735) <{shift = 0 : i8}> : (tensor<1x64x1280xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1280xf32>
    %2738 = "tosa.reshape"(%arg296) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2739 = "tosa.mul"(%2737, %2738) <{shift = 0 : i8}> : (tensor<1x64x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x64x1280xf32>
    %2740 = "tosa.reshape"(%arg297) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %2741 = "tosa.add"(%2739, %2740) : (tensor<1x64x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x64x1280xf32>
    %2742 = "tosa.reshape"(%2741) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2743 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2744 = "tosa.transpose"(%arg298, %2743) : (tensor<10240x1280xf32>, tensor<2xi32>) -> tensor<1280x10240xf32>
    %2745 = "tosa.reshape"(%2742) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2746 = "tosa.reshape"(%2744) <{new_shape = array<i64: 1, 1280, 10240>}> : (tensor<1280x10240xf32>) -> tensor<1x1280x10240xf32>
    %2747 = "tosa.matmul"(%2745, %2746) : (tensor<1x64x1280xf32>, tensor<1x1280x10240xf32>) -> tensor<1x64x10240xf32>
    %2748 = "tosa.reshape"(%2747) <{new_shape = array<i64: 64, 10240>}> : (tensor<1x64x10240xf32>) -> tensor<64x10240xf32>
    %2749 = "tosa.reshape"(%arg299) <{new_shape = array<i64: 1, 10240>}> : (tensor<10240xf32>) -> tensor<1x10240xf32>
    %2750 = "tosa.add"(%2749, %2748) : (tensor<1x10240xf32>, tensor<64x10240xf32>) -> tensor<64x10240xf32>
    %2751 = "tosa.reshape"(%2750) <{new_shape = array<i64: 1, 64, 10240>}> : (tensor<64x10240xf32>) -> tensor<1x64x10240xf32>
    %2752 = "tosa.slice"(%2751) <{size = array<i64: 0, 0, 5120>, start = array<i64: 0, 0, 0>}> : (tensor<1x64x10240xf32>) -> tensor<1x64x5120xf32>
    %2753 = "tosa.slice"(%2751) <{size = array<i64: 0, 0, 10240>, start = array<i64: 0, 0, 5120>}> : (tensor<1x64x10240xf32>) -> tensor<1x64x5120xf32>
    %2754 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x64x5120xf32>}> : () -> tensor<1x64x5120xf32>
    %2755 = "tosa.mul"(%2753, %2754) <{shift = 0 : i8}> : (tensor<1x64x5120xf32>, tensor<1x64x5120xf32>) -> tensor<1x64x5120xf32>
    %2756 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x64x5120xf32>}> : () -> tensor<1x64x5120xf32>
    %2757 = "tosa.mul"(%2753, %2756) <{shift = 0 : i8}> : (tensor<1x64x5120xf32>, tensor<1x64x5120xf32>) -> tensor<1x64x5120xf32>
    %2758 = "math.erf"(%2757) <{fastmath = #arith.fastmath<none>}> : (tensor<1x64x5120xf32>) -> tensor<1x64x5120xf32>
    %2759 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x64x5120xf32>}> : () -> tensor<1x64x5120xf32>
    %2760 = "tosa.add"(%2758, %2759) : (tensor<1x64x5120xf32>, tensor<1x64x5120xf32>) -> tensor<1x64x5120xf32>
    %2761 = "tosa.mul"(%2755, %2760) <{shift = 0 : i8}> : (tensor<1x64x5120xf32>, tensor<1x64x5120xf32>) -> tensor<1x64x5120xf32>
    %2762 = "tosa.mul"(%2752, %2761) <{shift = 0 : i8}> : (tensor<1x64x5120xf32>, tensor<1x64x5120xf32>) -> tensor<1x64x5120xf32>
    %2763 = "tosa.identity"(%2762) : (tensor<1x64x5120xf32>) -> tensor<1x64x5120xf32>
    %2764 = "tosa.reshape"(%2763) <{new_shape = array<i64: 64, 5120>}> : (tensor<1x64x5120xf32>) -> tensor<64x5120xf32>
    %2765 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2766 = "tosa.transpose"(%arg300, %2765) : (tensor<1280x5120xf32>, tensor<2xi32>) -> tensor<5120x1280xf32>
    %2767 = "tosa.reshape"(%2764) <{new_shape = array<i64: 1, 64, 5120>}> : (tensor<64x5120xf32>) -> tensor<1x64x5120xf32>
    %2768 = "tosa.reshape"(%2766) <{new_shape = array<i64: 1, 5120, 1280>}> : (tensor<5120x1280xf32>) -> tensor<1x5120x1280xf32>
    %2769 = "tosa.matmul"(%2767, %2768) : (tensor<1x64x5120xf32>, tensor<1x5120x1280xf32>) -> tensor<1x64x1280xf32>
    %2770 = "tosa.reshape"(%2769) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2771 = "tosa.reshape"(%arg301) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2772 = "tosa.add"(%2771, %2770) : (tensor<1x1280xf32>, tensor<64x1280xf32>) -> tensor<64x1280xf32>
    %2773 = "tosa.reshape"(%2772) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2774 = "tosa.add"(%2773, %2722) : (tensor<1x64x1280xf32>, tensor<1x64x1280xf32>) -> tensor<1x64x1280xf32>
    %2775 = "tosa.reshape"(%2774) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2776 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2777 = "tosa.transpose"(%arg302, %2776) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2778 = "tosa.reshape"(%2775) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2779 = "tosa.reshape"(%2777) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %2780 = "tosa.matmul"(%2778, %2779) : (tensor<1x64x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x64x1280xf32>
    %2781 = "tosa.reshape"(%2780) <{new_shape = array<i64: 64, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<64x1280xf32>
    %2782 = "tosa.reshape"(%arg303) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2783 = "tosa.add"(%2782, %2781) : (tensor<1x1280xf32>, tensor<64x1280xf32>) -> tensor<64x1280xf32>
    %2784 = "tosa.reshape"(%2783) <{new_shape = array<i64: 1, 64, 1280>}> : (tensor<64x1280xf32>) -> tensor<1x64x1280xf32>
    %2785 = "tosa.reshape"(%2784) <{new_shape = array<i64: 1, 8, 8, 1280>}> : (tensor<1x64x1280xf32>) -> tensor<1x8x8x1280xf32>
    %2786 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2787 = "tosa.transpose"(%2785, %2786) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2788 = "tosa.identity"(%2787) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2789 = "tosa.add"(%2788, %2522) : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2790 = "tosa.reshape"(%2789) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %2791 = "tosa.reduce_sum"(%2790) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2792 = "tosa.reduce_sum"(%2791) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2793 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2794 = "tosa.reciprocal"(%2793) : (tensor<1xf32>) -> tensor<1xf32>
    %2795 = "tosa.mul"(%2794, %2792) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2796 = "tosa.sub"(%2790, %2795) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2797 = "tosa.mul"(%2796, %2796) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2798 = "tosa.reduce_sum"(%2797) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2799 = "tosa.reduce_sum"(%2798) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2800 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2801 = "tosa.reciprocal"(%2800) : (tensor<1xf32>) -> tensor<1xf32>
    %2802 = "tosa.mul"(%2801, %2799) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2803 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %2804 = "tosa.add"(%2802, %2803) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2805 = "tosa.rsqrt"(%2804) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2806 = "tosa.sub"(%2790, %2795) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2807 = "tosa.mul"(%2806, %2805) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2808 = "tosa.reshape"(%2807) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %2809 = "tosa.reshape"(%arg304) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2810 = "tosa.reshape"(%2809) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2811 = "tosa.reshape"(%2810) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2812 = "tosa.reshape"(%arg305) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2813 = "tosa.reshape"(%2812) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2814 = "tosa.reshape"(%2813) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2815 = "tosa.mul"(%2808, %2814) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2816 = "tosa.add"(%2815, %2811) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2817 = "tosa.sigmoid"(%2816) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2818 = "tosa.mul"(%2816, %2817) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2819 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2820 = "tosa.transpose"(%2818, %2819) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %2821 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2822 = "tosa.transpose"(%arg306, %2821) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %2823 = "tosa.conv2d"(%2820, %2822, %arg307) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2824 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2825 = "tosa.transpose"(%2823, %2824) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2826 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2827 = "tosa.mul"(%50, %2826) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2828 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2829 = "tosa.transpose"(%arg308, %2828) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2830 = "tosa.reshape"(%2827) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %2831 = "tosa.reshape"(%2829) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %2832 = "tosa.matmul"(%2830, %2831) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %2833 = "tosa.reshape"(%2832) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %2834 = "tosa.reshape"(%arg309) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2835 = "tosa.add"(%2834, %2833) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2836 = "tensor.extract_slice"(%2835) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2837 = "tensor.extract_slice"(%2836) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2838 = "tosa.reshape"(%2837) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2839 = "tosa.reshape"(%2838) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2840 = "tosa.add"(%2825, %2839) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2841 = "tosa.reshape"(%2840) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %2842 = "tosa.reduce_sum"(%2841) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2843 = "tosa.reduce_sum"(%2842) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2844 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2845 = "tosa.reciprocal"(%2844) : (tensor<1xf32>) -> tensor<1xf32>
    %2846 = "tosa.mul"(%2845, %2843) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2847 = "tosa.sub"(%2841, %2846) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2848 = "tosa.mul"(%2847, %2847) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2849 = "tosa.reduce_sum"(%2848) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2850 = "tosa.reduce_sum"(%2849) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2851 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2852 = "tosa.reciprocal"(%2851) : (tensor<1xf32>) -> tensor<1xf32>
    %2853 = "tosa.mul"(%2852, %2850) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2854 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %2855 = "tosa.add"(%2853, %2854) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2856 = "tosa.rsqrt"(%2855) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2857 = "tosa.sub"(%2841, %2846) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2858 = "tosa.mul"(%2857, %2856) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2859 = "tosa.reshape"(%2858) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %2860 = "tosa.reshape"(%arg310) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2861 = "tosa.reshape"(%2860) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2862 = "tosa.reshape"(%2861) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2863 = "tosa.reshape"(%arg311) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2864 = "tosa.reshape"(%2863) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2865 = "tosa.reshape"(%2864) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2866 = "tosa.mul"(%2859, %2865) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2867 = "tosa.add"(%2866, %2862) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2868 = "tosa.sigmoid"(%2867) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2869 = "tosa.mul"(%2867, %2868) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2870 = "tosa.identity"(%2869) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2871 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2872 = "tosa.transpose"(%2870, %2871) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %2873 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2874 = "tosa.transpose"(%arg312, %2873) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %2875 = "tosa.conv2d"(%2872, %2874, %arg313) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2876 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2877 = "tosa.transpose"(%2875, %2876) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2878 = "tosa.add"(%2789, %2877) : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2879 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x8x8xf32>}> : () -> tensor<1x1280x8x8xf32>
    %2880 = "tosa.reciprocal"(%2879) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2881 = "tosa.mul"(%2878, %2880) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2882 = "tensor.empty"() : () -> tensor<1x2560x8x8xf32>
    %2883 = "tensor.insert_slice"(%2881, %2882) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1280, 8, 8>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x8x8xf32>, tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %2884 = "tensor.insert_slice"(%2430, %2883) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 1280, 0, 0>, static_sizes = array<i64: 1, 1280, 8, 8>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x8x8xf32>, tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %2885 = "tosa.reshape"(%2884) <{new_shape = array<i64: 1, 32, 80, 64>}> : (tensor<1x2560x8x8xf32>) -> tensor<1x32x80x64xf32>
    %2886 = "tosa.reduce_sum"(%2885) <{axis = 2 : i32}> : (tensor<1x32x80x64xf32>) -> tensor<1x32x1x64xf32>
    %2887 = "tosa.reduce_sum"(%2886) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2888 = "tosa.const"() <{value = dense<5.120000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2889 = "tosa.reciprocal"(%2888) : (tensor<1xf32>) -> tensor<1xf32>
    %2890 = "tosa.mul"(%2889, %2887) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2891 = "tosa.sub"(%2885, %2890) : (tensor<1x32x80x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x64xf32>
    %2892 = "tosa.mul"(%2891, %2891) <{shift = 0 : i8}> : (tensor<1x32x80x64xf32>, tensor<1x32x80x64xf32>) -> tensor<1x32x80x64xf32>
    %2893 = "tosa.reduce_sum"(%2892) <{axis = 2 : i32}> : (tensor<1x32x80x64xf32>) -> tensor<1x32x1x64xf32>
    %2894 = "tosa.reduce_sum"(%2893) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2895 = "tosa.const"() <{value = dense<5.120000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2896 = "tosa.reciprocal"(%2895) : (tensor<1xf32>) -> tensor<1xf32>
    %2897 = "tosa.mul"(%2896, %2894) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2898 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %2899 = "tosa.add"(%2897, %2898) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2900 = "tosa.rsqrt"(%2899) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2901 = "tosa.sub"(%2885, %2890) : (tensor<1x32x80x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x64xf32>
    %2902 = "tosa.mul"(%2901, %2900) <{shift = 0 : i8}> : (tensor<1x32x80x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x64xf32>
    %2903 = "tosa.reshape"(%2902) <{new_shape = array<i64: 1, 2560, 8, 8>}> : (tensor<1x32x80x64xf32>) -> tensor<1x2560x8x8xf32>
    %2904 = "tosa.reshape"(%arg314) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %2905 = "tosa.reshape"(%2904) <{new_shape = array<i64: 1, 2560, 1>}> : (tensor<1x2560xf32>) -> tensor<1x2560x1xf32>
    %2906 = "tosa.reshape"(%2905) <{new_shape = array<i64: 1, 2560, 1, 1>}> : (tensor<1x2560x1xf32>) -> tensor<1x2560x1x1xf32>
    %2907 = "tosa.reshape"(%arg315) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %2908 = "tosa.reshape"(%2907) <{new_shape = array<i64: 1, 2560, 1>}> : (tensor<1x2560xf32>) -> tensor<1x2560x1xf32>
    %2909 = "tosa.reshape"(%2908) <{new_shape = array<i64: 1, 2560, 1, 1>}> : (tensor<1x2560x1xf32>) -> tensor<1x2560x1x1xf32>
    %2910 = "tosa.mul"(%2903, %2909) <{shift = 0 : i8}> : (tensor<1x2560x8x8xf32>, tensor<1x2560x1x1xf32>) -> tensor<1x2560x8x8xf32>
    %2911 = "tosa.add"(%2910, %2906) : (tensor<1x2560x8x8xf32>, tensor<1x2560x1x1xf32>) -> tensor<1x2560x8x8xf32>
    %2912 = "tosa.sigmoid"(%2911) : (tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %2913 = "tosa.mul"(%2911, %2912) <{shift = 0 : i8}> : (tensor<1x2560x8x8xf32>, tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %2914 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2915 = "tosa.transpose"(%2913, %2914) : (tensor<1x2560x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x2560xf32>
    %2916 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2917 = "tosa.transpose"(%arg316, %2916) : (tensor<1280x2560x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x2560xf32>
    %2918 = "tosa.conv2d"(%2915, %2917, %arg317) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x2560xf32>, tensor<1280x3x3x2560xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2919 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2920 = "tosa.transpose"(%2918, %2919) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2921 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2922 = "tosa.mul"(%50, %2921) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2923 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2924 = "tosa.transpose"(%arg318, %2923) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %2925 = "tosa.reshape"(%2922) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %2926 = "tosa.reshape"(%2924) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %2927 = "tosa.matmul"(%2925, %2926) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %2928 = "tosa.reshape"(%2927) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %2929 = "tosa.reshape"(%arg319) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2930 = "tosa.add"(%2929, %2928) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2931 = "tensor.extract_slice"(%2930) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2932 = "tensor.extract_slice"(%2931) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %2933 = "tosa.reshape"(%2932) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2934 = "tosa.reshape"(%2933) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2935 = "tosa.add"(%2920, %2934) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2936 = "tosa.reshape"(%2935) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %2937 = "tosa.reduce_sum"(%2936) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2938 = "tosa.reduce_sum"(%2937) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2939 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2940 = "tosa.reciprocal"(%2939) : (tensor<1xf32>) -> tensor<1xf32>
    %2941 = "tosa.mul"(%2940, %2938) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2942 = "tosa.sub"(%2936, %2941) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2943 = "tosa.mul"(%2942, %2942) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %2944 = "tosa.reduce_sum"(%2943) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %2945 = "tosa.reduce_sum"(%2944) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2946 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2947 = "tosa.reciprocal"(%2946) : (tensor<1xf32>) -> tensor<1xf32>
    %2948 = "tosa.mul"(%2947, %2945) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2949 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %2950 = "tosa.add"(%2948, %2949) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2951 = "tosa.rsqrt"(%2950) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2952 = "tosa.sub"(%2936, %2941) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2953 = "tosa.mul"(%2952, %2951) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %2954 = "tosa.reshape"(%2953) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %2955 = "tosa.reshape"(%arg320) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2956 = "tosa.reshape"(%2955) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2957 = "tosa.reshape"(%2956) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2958 = "tosa.reshape"(%arg321) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %2959 = "tosa.reshape"(%2958) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %2960 = "tosa.reshape"(%2959) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %2961 = "tosa.mul"(%2954, %2960) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2962 = "tosa.add"(%2961, %2957) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %2963 = "tosa.sigmoid"(%2962) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2964 = "tosa.mul"(%2962, %2963) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2965 = "tosa.identity"(%2964) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2966 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2967 = "tosa.transpose"(%2965, %2966) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %2968 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2969 = "tosa.transpose"(%arg322, %2968) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %2970 = "tosa.conv2d"(%2967, %2969, %arg323) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2971 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2972 = "tosa.transpose"(%2970, %2971) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2973 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2974 = "tosa.transpose"(%2884, %2973) : (tensor<1x2560x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x2560xf32>
    %2975 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2976 = "tosa.transpose"(%arg324, %2975) : (tensor<1280x2560x1x1xf32>, tensor<4xi32>) -> tensor<1280x1x1x2560xf32>
    %2977 = "tosa.conv2d"(%2974, %2976, %arg325) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x2560xf32>, tensor<1280x1x1x2560xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %2978 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2979 = "tosa.transpose"(%2977, %2978) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %2980 = "tosa.add"(%2979, %2972) : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2981 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x8x8xf32>}> : () -> tensor<1x1280x8x8xf32>
    %2982 = "tosa.reciprocal"(%2981) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2983 = "tosa.mul"(%2980, %2982) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %2984 = "tensor.empty"() : () -> tensor<1x2560x8x8xf32>
    %2985 = "tensor.insert_slice"(%2983, %2984) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1280, 8, 8>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x8x8xf32>, tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %2986 = "tensor.insert_slice"(%2338, %2985) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 1280, 0, 0>, static_sizes = array<i64: 1, 1280, 8, 8>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x8x8xf32>, tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %2987 = "tosa.reshape"(%2986) <{new_shape = array<i64: 1, 32, 80, 64>}> : (tensor<1x2560x8x8xf32>) -> tensor<1x32x80x64xf32>
    %2988 = "tosa.reduce_sum"(%2987) <{axis = 2 : i32}> : (tensor<1x32x80x64xf32>) -> tensor<1x32x1x64xf32>
    %2989 = "tosa.reduce_sum"(%2988) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2990 = "tosa.const"() <{value = dense<5.120000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2991 = "tosa.reciprocal"(%2990) : (tensor<1xf32>) -> tensor<1xf32>
    %2992 = "tosa.mul"(%2991, %2989) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %2993 = "tosa.sub"(%2987, %2992) : (tensor<1x32x80x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x64xf32>
    %2994 = "tosa.mul"(%2993, %2993) <{shift = 0 : i8}> : (tensor<1x32x80x64xf32>, tensor<1x32x80x64xf32>) -> tensor<1x32x80x64xf32>
    %2995 = "tosa.reduce_sum"(%2994) <{axis = 2 : i32}> : (tensor<1x32x80x64xf32>) -> tensor<1x32x1x64xf32>
    %2996 = "tosa.reduce_sum"(%2995) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %2997 = "tosa.const"() <{value = dense<5.120000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2998 = "tosa.reciprocal"(%2997) : (tensor<1xf32>) -> tensor<1xf32>
    %2999 = "tosa.mul"(%2998, %2996) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3000 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %3001 = "tosa.add"(%2999, %3000) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3002 = "tosa.rsqrt"(%3001) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3003 = "tosa.sub"(%2987, %2992) : (tensor<1x32x80x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x64xf32>
    %3004 = "tosa.mul"(%3003, %3002) <{shift = 0 : i8}> : (tensor<1x32x80x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x64xf32>
    %3005 = "tosa.reshape"(%3004) <{new_shape = array<i64: 1, 2560, 8, 8>}> : (tensor<1x32x80x64xf32>) -> tensor<1x2560x8x8xf32>
    %3006 = "tosa.reshape"(%arg326) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %3007 = "tosa.reshape"(%3006) <{new_shape = array<i64: 1, 2560, 1>}> : (tensor<1x2560xf32>) -> tensor<1x2560x1xf32>
    %3008 = "tosa.reshape"(%3007) <{new_shape = array<i64: 1, 2560, 1, 1>}> : (tensor<1x2560x1xf32>) -> tensor<1x2560x1x1xf32>
    %3009 = "tosa.reshape"(%arg327) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %3010 = "tosa.reshape"(%3009) <{new_shape = array<i64: 1, 2560, 1>}> : (tensor<1x2560xf32>) -> tensor<1x2560x1xf32>
    %3011 = "tosa.reshape"(%3010) <{new_shape = array<i64: 1, 2560, 1, 1>}> : (tensor<1x2560x1xf32>) -> tensor<1x2560x1x1xf32>
    %3012 = "tosa.mul"(%3005, %3011) <{shift = 0 : i8}> : (tensor<1x2560x8x8xf32>, tensor<1x2560x1x1xf32>) -> tensor<1x2560x8x8xf32>
    %3013 = "tosa.add"(%3012, %3008) : (tensor<1x2560x8x8xf32>, tensor<1x2560x1x1xf32>) -> tensor<1x2560x8x8xf32>
    %3014 = "tosa.sigmoid"(%3013) : (tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %3015 = "tosa.mul"(%3013, %3014) <{shift = 0 : i8}> : (tensor<1x2560x8x8xf32>, tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %3016 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3017 = "tosa.transpose"(%3015, %3016) : (tensor<1x2560x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x2560xf32>
    %3018 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3019 = "tosa.transpose"(%arg328, %3018) : (tensor<1280x2560x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x2560xf32>
    %3020 = "tosa.conv2d"(%3017, %3019, %arg329) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x2560xf32>, tensor<1280x3x3x2560xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %3021 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3022 = "tosa.transpose"(%3020, %3021) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %3023 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3024 = "tosa.mul"(%50, %3023) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3025 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3026 = "tosa.transpose"(%arg330, %3025) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3027 = "tosa.reshape"(%3024) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %3028 = "tosa.reshape"(%3026) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %3029 = "tosa.matmul"(%3027, %3028) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %3030 = "tosa.reshape"(%3029) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %3031 = "tosa.reshape"(%arg331) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3032 = "tosa.add"(%3031, %3030) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3033 = "tensor.extract_slice"(%3032) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3034 = "tensor.extract_slice"(%3033) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3035 = "tosa.reshape"(%3034) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3036 = "tosa.reshape"(%3035) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3037 = "tosa.add"(%3022, %3036) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %3038 = "tosa.reshape"(%3037) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %3039 = "tosa.reduce_sum"(%3038) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %3040 = "tosa.reduce_sum"(%3039) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %3041 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3042 = "tosa.reciprocal"(%3041) : (tensor<1xf32>) -> tensor<1xf32>
    %3043 = "tosa.mul"(%3042, %3040) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3044 = "tosa.sub"(%3038, %3043) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %3045 = "tosa.mul"(%3044, %3044) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3046 = "tosa.reduce_sum"(%3045) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %3047 = "tosa.reduce_sum"(%3046) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %3048 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3049 = "tosa.reciprocal"(%3048) : (tensor<1xf32>) -> tensor<1xf32>
    %3050 = "tosa.mul"(%3049, %3047) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3051 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %3052 = "tosa.add"(%3050, %3051) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3053 = "tosa.rsqrt"(%3052) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3054 = "tosa.sub"(%3038, %3043) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %3055 = "tosa.mul"(%3054, %3053) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %3056 = "tosa.reshape"(%3055) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %3057 = "tosa.reshape"(%arg332) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3058 = "tosa.reshape"(%3057) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3059 = "tosa.reshape"(%3058) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3060 = "tosa.reshape"(%arg333) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3061 = "tosa.reshape"(%3060) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3062 = "tosa.reshape"(%3061) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3063 = "tosa.mul"(%3056, %3062) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %3064 = "tosa.add"(%3063, %3059) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %3065 = "tosa.sigmoid"(%3064) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3066 = "tosa.mul"(%3064, %3065) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3067 = "tosa.identity"(%3066) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3068 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3069 = "tosa.transpose"(%3067, %3068) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %3070 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3071 = "tosa.transpose"(%arg334, %3070) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %3072 = "tosa.conv2d"(%3069, %3071, %arg335) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %3073 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3074 = "tosa.transpose"(%3072, %3073) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %3075 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3076 = "tosa.transpose"(%2986, %3075) : (tensor<1x2560x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x2560xf32>
    %3077 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3078 = "tosa.transpose"(%arg336, %3077) : (tensor<1280x2560x1x1xf32>, tensor<4xi32>) -> tensor<1280x1x1x2560xf32>
    %3079 = "tosa.conv2d"(%3076, %3078, %arg337) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x2560xf32>, tensor<1280x1x1x2560xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %3080 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3081 = "tosa.transpose"(%3079, %3080) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %3082 = "tosa.add"(%3081, %3074) : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3083 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x8x8xf32>}> : () -> tensor<1x1280x8x8xf32>
    %3084 = "tosa.reciprocal"(%3083) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3085 = "tosa.mul"(%3082, %3084) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3086 = "tensor.empty"() : () -> tensor<1x2560x8x8xf32>
    %3087 = "tensor.insert_slice"(%3085, %3086) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1280, 8, 8>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x8x8xf32>, tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %3088 = "tensor.insert_slice"(%2246, %3087) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 1280, 0, 0>, static_sizes = array<i64: 1, 1280, 8, 8>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x8x8xf32>, tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %3089 = "tosa.reshape"(%3088) <{new_shape = array<i64: 1, 32, 80, 64>}> : (tensor<1x2560x8x8xf32>) -> tensor<1x32x80x64xf32>
    %3090 = "tosa.reduce_sum"(%3089) <{axis = 2 : i32}> : (tensor<1x32x80x64xf32>) -> tensor<1x32x1x64xf32>
    %3091 = "tosa.reduce_sum"(%3090) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %3092 = "tosa.const"() <{value = dense<5.120000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3093 = "tosa.reciprocal"(%3092) : (tensor<1xf32>) -> tensor<1xf32>
    %3094 = "tosa.mul"(%3093, %3091) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3095 = "tosa.sub"(%3089, %3094) : (tensor<1x32x80x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x64xf32>
    %3096 = "tosa.mul"(%3095, %3095) <{shift = 0 : i8}> : (tensor<1x32x80x64xf32>, tensor<1x32x80x64xf32>) -> tensor<1x32x80x64xf32>
    %3097 = "tosa.reduce_sum"(%3096) <{axis = 2 : i32}> : (tensor<1x32x80x64xf32>) -> tensor<1x32x1x64xf32>
    %3098 = "tosa.reduce_sum"(%3097) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %3099 = "tosa.const"() <{value = dense<5.120000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3100 = "tosa.reciprocal"(%3099) : (tensor<1xf32>) -> tensor<1xf32>
    %3101 = "tosa.mul"(%3100, %3098) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3102 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %3103 = "tosa.add"(%3101, %3102) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3104 = "tosa.rsqrt"(%3103) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3105 = "tosa.sub"(%3089, %3094) : (tensor<1x32x80x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x64xf32>
    %3106 = "tosa.mul"(%3105, %3104) <{shift = 0 : i8}> : (tensor<1x32x80x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x64xf32>
    %3107 = "tosa.reshape"(%3106) <{new_shape = array<i64: 1, 2560, 8, 8>}> : (tensor<1x32x80x64xf32>) -> tensor<1x2560x8x8xf32>
    %3108 = "tosa.reshape"(%arg338) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %3109 = "tosa.reshape"(%3108) <{new_shape = array<i64: 1, 2560, 1>}> : (tensor<1x2560xf32>) -> tensor<1x2560x1xf32>
    %3110 = "tosa.reshape"(%3109) <{new_shape = array<i64: 1, 2560, 1, 1>}> : (tensor<1x2560x1xf32>) -> tensor<1x2560x1x1xf32>
    %3111 = "tosa.reshape"(%arg339) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %3112 = "tosa.reshape"(%3111) <{new_shape = array<i64: 1, 2560, 1>}> : (tensor<1x2560xf32>) -> tensor<1x2560x1xf32>
    %3113 = "tosa.reshape"(%3112) <{new_shape = array<i64: 1, 2560, 1, 1>}> : (tensor<1x2560x1xf32>) -> tensor<1x2560x1x1xf32>
    %3114 = "tosa.mul"(%3107, %3113) <{shift = 0 : i8}> : (tensor<1x2560x8x8xf32>, tensor<1x2560x1x1xf32>) -> tensor<1x2560x8x8xf32>
    %3115 = "tosa.add"(%3114, %3110) : (tensor<1x2560x8x8xf32>, tensor<1x2560x1x1xf32>) -> tensor<1x2560x8x8xf32>
    %3116 = "tosa.sigmoid"(%3115) : (tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %3117 = "tosa.mul"(%3115, %3116) <{shift = 0 : i8}> : (tensor<1x2560x8x8xf32>, tensor<1x2560x8x8xf32>) -> tensor<1x2560x8x8xf32>
    %3118 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3119 = "tosa.transpose"(%3117, %3118) : (tensor<1x2560x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x2560xf32>
    %3120 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3121 = "tosa.transpose"(%arg340, %3120) : (tensor<1280x2560x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x2560xf32>
    %3122 = "tosa.conv2d"(%3119, %3121, %arg341) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x2560xf32>, tensor<1280x3x3x2560xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %3123 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3124 = "tosa.transpose"(%3122, %3123) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %3125 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3126 = "tosa.mul"(%50, %3125) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3127 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3128 = "tosa.transpose"(%arg342, %3127) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3129 = "tosa.reshape"(%3126) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %3130 = "tosa.reshape"(%3128) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %3131 = "tosa.matmul"(%3129, %3130) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %3132 = "tosa.reshape"(%3131) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %3133 = "tosa.reshape"(%arg343) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3134 = "tosa.add"(%3133, %3132) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3135 = "tensor.extract_slice"(%3134) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3136 = "tensor.extract_slice"(%3135) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3137 = "tosa.reshape"(%3136) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3138 = "tosa.reshape"(%3137) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3139 = "tosa.add"(%3124, %3138) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %3140 = "tosa.reshape"(%3139) <{new_shape = array<i64: 1, 32, 40, 64>}> : (tensor<1x1280x8x8xf32>) -> tensor<1x32x40x64xf32>
    %3141 = "tosa.reduce_sum"(%3140) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %3142 = "tosa.reduce_sum"(%3141) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %3143 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3144 = "tosa.reciprocal"(%3143) : (tensor<1xf32>) -> tensor<1xf32>
    %3145 = "tosa.mul"(%3144, %3142) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3146 = "tosa.sub"(%3140, %3145) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %3147 = "tosa.mul"(%3146, %3146) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
    %3148 = "tosa.reduce_sum"(%3147) <{axis = 2 : i32}> : (tensor<1x32x40x64xf32>) -> tensor<1x32x1x64xf32>
    %3149 = "tosa.reduce_sum"(%3148) <{axis = 3 : i32}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
    %3150 = "tosa.const"() <{value = dense<2.560000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3151 = "tosa.reciprocal"(%3150) : (tensor<1xf32>) -> tensor<1xf32>
    %3152 = "tosa.mul"(%3151, %3149) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3153 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %3154 = "tosa.add"(%3152, %3153) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3155 = "tosa.rsqrt"(%3154) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3156 = "tosa.sub"(%3140, %3145) : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %3157 = "tosa.mul"(%3156, %3155) <{shift = 0 : i8}> : (tensor<1x32x40x64xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x64xf32>
    %3158 = "tosa.reshape"(%3157) <{new_shape = array<i64: 1, 1280, 8, 8>}> : (tensor<1x32x40x64xf32>) -> tensor<1x1280x8x8xf32>
    %3159 = "tosa.reshape"(%arg344) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3160 = "tosa.reshape"(%3159) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3161 = "tosa.reshape"(%3160) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3162 = "tosa.reshape"(%arg345) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3163 = "tosa.reshape"(%3162) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3164 = "tosa.reshape"(%3163) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3165 = "tosa.mul"(%3158, %3164) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %3166 = "tosa.add"(%3165, %3161) : (tensor<1x1280x8x8xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x8x8xf32>
    %3167 = "tosa.sigmoid"(%3166) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3168 = "tosa.mul"(%3166, %3167) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3169 = "tosa.identity"(%3168) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3170 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3171 = "tosa.transpose"(%3169, %3170) : (tensor<1x1280x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x1280xf32>
    %3172 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3173 = "tosa.transpose"(%arg346, %3172) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %3174 = "tosa.conv2d"(%3171, %3173, %arg347) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %3175 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3176 = "tosa.transpose"(%3174, %3175) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %3177 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3178 = "tosa.transpose"(%3088, %3177) : (tensor<1x2560x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x2560xf32>
    %3179 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3180 = "tosa.transpose"(%arg348, %3179) : (tensor<1280x2560x1x1xf32>, tensor<4xi32>) -> tensor<1280x1x1x2560xf32>
    %3181 = "tosa.conv2d"(%3178, %3180, %arg349) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x2560xf32>, tensor<1280x1x1x2560xf32>, tensor<1280xf32>) -> tensor<1x8x8x1280xf32>
    %3182 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3183 = "tosa.transpose"(%3181, %3182) : (tensor<1x8x8x1280xf32>, tensor<4xi32>) -> tensor<1x1280x8x8xf32>
    %3184 = "tosa.add"(%3183, %3176) : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3185 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x8x8xf32>}> : () -> tensor<1x1280x8x8xf32>
    %3186 = "tosa.reciprocal"(%3185) : (tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3187 = "tosa.mul"(%3184, %3186) <{shift = 0 : i8}> : (tensor<1x1280x8x8xf32>, tensor<1x1280x8x8xf32>) -> tensor<1x1280x8x8xf32>
    %3188 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>}> : () -> tensor<16xi64>
    %3189 = "tosa.cast"(%3188) : (tensor<16xi64>) -> tensor<16xf32>
    %3190 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
    %3191 = "tosa.mul"(%3189, %3190) <{shift = 0 : i8}> : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %3192 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
    %3193 = "tosa.add"(%3191, %3192) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %3194 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<16xf32>}> : () -> tensor<16xf32>
    %3195 = "tosa.mul"(%3193, %3194) <{shift = 0 : i8}> : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %3196 = "tosa.cast"(%3195) : (tensor<16xf32>) -> tensor<16xi64>
    %3197 = "tosa.reshape"(%3196) <{new_shape = array<i64: 16, 1>}> : (tensor<16xi64>) -> tensor<16x1xi64>
    %3198 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>}> : () -> tensor<16xi64>
    %3199 = "tosa.cast"(%3198) : (tensor<16xi64>) -> tensor<16xf32>
    %3200 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
    %3201 = "tosa.mul"(%3199, %3200) <{shift = 0 : i8}> : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %3202 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
    %3203 = "tosa.add"(%3201, %3202) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %3204 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<16xf32>}> : () -> tensor<16xf32>
    %3205 = "tosa.mul"(%3203, %3204) <{shift = 0 : i8}> : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %3206 = "tosa.cast"(%3205) : (tensor<16xf32>) -> tensor<16xi64>
    %3207 = "tensor.empty"() : () -> tensor<1x1280x16x16xf32>
    %3208 = "linalg.generic"(%3197, %3206, %3207) <{indexing_maps = [#map3, #map3, #map4], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: i64, %arg721: i64, %arg722: f32):
      %6629 = "arith.index_cast"(%arg720) : (i64) -> index
      %6630 = "arith.index_cast"(%arg721) : (i64) -> index
      %6631 = "linalg.index"() <{dim = 2 : i64}> : () -> index
      %6632 = "tensor.extract"(%3187, %6629, %6630, %6631) : (tensor<1x1280x8x8xf32>, index, index, index) -> f32
      "linalg.yield"(%6632) : (f32) -> ()
    }) : (tensor<16x1xi64>, tensor<16xi64>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3209 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3210 = "tosa.transpose"(%3208, %3209) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %3211 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3212 = "tosa.transpose"(%arg350, %3211) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %3213 = "tosa.conv2d"(%3210, %3212, %arg351) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %3214 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3215 = "tosa.transpose"(%3213, %3214) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %3216 = "tensor.empty"() : () -> tensor<1x2560x16x16xf32>
    %3217 = "tensor.insert_slice"(%3215, %3216) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1280, 16, 16>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x16x16xf32>, tensor<1x2560x16x16xf32>) -> tensor<1x2560x16x16xf32>
    %3218 = "tensor.insert_slice"(%2239, %3217) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 1280, 0, 0>, static_sizes = array<i64: 1, 1280, 16, 16>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x16x16xf32>, tensor<1x2560x16x16xf32>) -> tensor<1x2560x16x16xf32>
    %3219 = "tosa.reshape"(%3218) <{new_shape = array<i64: 1, 32, 80, 256>}> : (tensor<1x2560x16x16xf32>) -> tensor<1x32x80x256xf32>
    %3220 = "tosa.reduce_sum"(%3219) <{axis = 2 : i32}> : (tensor<1x32x80x256xf32>) -> tensor<1x32x1x256xf32>
    %3221 = "tosa.reduce_sum"(%3220) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3222 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3223 = "tosa.reciprocal"(%3222) : (tensor<1xf32>) -> tensor<1xf32>
    %3224 = "tosa.mul"(%3223, %3221) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3225 = "tosa.sub"(%3219, %3224) : (tensor<1x32x80x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x256xf32>
    %3226 = "tosa.mul"(%3225, %3225) <{shift = 0 : i8}> : (tensor<1x32x80x256xf32>, tensor<1x32x80x256xf32>) -> tensor<1x32x80x256xf32>
    %3227 = "tosa.reduce_sum"(%3226) <{axis = 2 : i32}> : (tensor<1x32x80x256xf32>) -> tensor<1x32x1x256xf32>
    %3228 = "tosa.reduce_sum"(%3227) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3229 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3230 = "tosa.reciprocal"(%3229) : (tensor<1xf32>) -> tensor<1xf32>
    %3231 = "tosa.mul"(%3230, %3228) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3232 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %3233 = "tosa.add"(%3231, %3232) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3234 = "tosa.rsqrt"(%3233) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3235 = "tosa.sub"(%3219, %3224) : (tensor<1x32x80x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x256xf32>
    %3236 = "tosa.mul"(%3235, %3234) <{shift = 0 : i8}> : (tensor<1x32x80x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x256xf32>
    %3237 = "tosa.reshape"(%3236) <{new_shape = array<i64: 1, 2560, 16, 16>}> : (tensor<1x32x80x256xf32>) -> tensor<1x2560x16x16xf32>
    %3238 = "tosa.reshape"(%arg352) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %3239 = "tosa.reshape"(%3238) <{new_shape = array<i64: 1, 2560, 1>}> : (tensor<1x2560xf32>) -> tensor<1x2560x1xf32>
    %3240 = "tosa.reshape"(%3239) <{new_shape = array<i64: 1, 2560, 1, 1>}> : (tensor<1x2560x1xf32>) -> tensor<1x2560x1x1xf32>
    %3241 = "tosa.reshape"(%arg353) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %3242 = "tosa.reshape"(%3241) <{new_shape = array<i64: 1, 2560, 1>}> : (tensor<1x2560xf32>) -> tensor<1x2560x1xf32>
    %3243 = "tosa.reshape"(%3242) <{new_shape = array<i64: 1, 2560, 1, 1>}> : (tensor<1x2560x1xf32>) -> tensor<1x2560x1x1xf32>
    %3244 = "tosa.mul"(%3237, %3243) <{shift = 0 : i8}> : (tensor<1x2560x16x16xf32>, tensor<1x2560x1x1xf32>) -> tensor<1x2560x16x16xf32>
    %3245 = "tosa.add"(%3244, %3240) : (tensor<1x2560x16x16xf32>, tensor<1x2560x1x1xf32>) -> tensor<1x2560x16x16xf32>
    %3246 = "tosa.sigmoid"(%3245) : (tensor<1x2560x16x16xf32>) -> tensor<1x2560x16x16xf32>
    %3247 = "tosa.mul"(%3245, %3246) <{shift = 0 : i8}> : (tensor<1x2560x16x16xf32>, tensor<1x2560x16x16xf32>) -> tensor<1x2560x16x16xf32>
    %3248 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3249 = "tosa.transpose"(%3247, %3248) : (tensor<1x2560x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x2560xf32>
    %3250 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3251 = "tosa.transpose"(%arg354, %3250) : (tensor<1280x2560x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x2560xf32>
    %3252 = "tosa.conv2d"(%3249, %3251, %arg355) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x2560xf32>, tensor<1280x3x3x2560xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %3253 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3254 = "tosa.transpose"(%3252, %3253) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %3255 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3256 = "tosa.mul"(%50, %3255) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3257 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3258 = "tosa.transpose"(%arg356, %3257) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3259 = "tosa.reshape"(%3256) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %3260 = "tosa.reshape"(%3258) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %3261 = "tosa.matmul"(%3259, %3260) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %3262 = "tosa.reshape"(%3261) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %3263 = "tosa.reshape"(%arg357) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3264 = "tosa.add"(%3263, %3262) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3265 = "tensor.extract_slice"(%3264) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3266 = "tensor.extract_slice"(%3265) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3267 = "tosa.reshape"(%3266) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3268 = "tosa.reshape"(%3267) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3269 = "tosa.add"(%3254, %3268) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %3270 = "tosa.reshape"(%3269) <{new_shape = array<i64: 1, 32, 40, 256>}> : (tensor<1x1280x16x16xf32>) -> tensor<1x32x40x256xf32>
    %3271 = "tosa.reduce_sum"(%3270) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %3272 = "tosa.reduce_sum"(%3271) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3273 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3274 = "tosa.reciprocal"(%3273) : (tensor<1xf32>) -> tensor<1xf32>
    %3275 = "tosa.mul"(%3274, %3272) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3276 = "tosa.sub"(%3270, %3275) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3277 = "tosa.mul"(%3276, %3276) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x40x256xf32>) -> tensor<1x32x40x256xf32>
    %3278 = "tosa.reduce_sum"(%3277) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %3279 = "tosa.reduce_sum"(%3278) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3280 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3281 = "tosa.reciprocal"(%3280) : (tensor<1xf32>) -> tensor<1xf32>
    %3282 = "tosa.mul"(%3281, %3279) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3283 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %3284 = "tosa.add"(%3282, %3283) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3285 = "tosa.rsqrt"(%3284) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3286 = "tosa.sub"(%3270, %3275) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3287 = "tosa.mul"(%3286, %3285) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3288 = "tosa.reshape"(%3287) <{new_shape = array<i64: 1, 1280, 16, 16>}> : (tensor<1x32x40x256xf32>) -> tensor<1x1280x16x16xf32>
    %3289 = "tosa.reshape"(%arg358) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3290 = "tosa.reshape"(%3289) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3291 = "tosa.reshape"(%3290) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3292 = "tosa.reshape"(%arg359) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3293 = "tosa.reshape"(%3292) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3294 = "tosa.reshape"(%3293) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3295 = "tosa.mul"(%3288, %3294) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %3296 = "tosa.add"(%3295, %3291) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %3297 = "tosa.sigmoid"(%3296) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3298 = "tosa.mul"(%3296, %3297) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3299 = "tosa.identity"(%3298) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3300 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3301 = "tosa.transpose"(%3299, %3300) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %3302 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3303 = "tosa.transpose"(%arg360, %3302) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %3304 = "tosa.conv2d"(%3301, %3303, %arg361) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %3305 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3306 = "tosa.transpose"(%3304, %3305) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %3307 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3308 = "tosa.transpose"(%3218, %3307) : (tensor<1x2560x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x2560xf32>
    %3309 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3310 = "tosa.transpose"(%arg362, %3309) : (tensor<1280x2560x1x1xf32>, tensor<4xi32>) -> tensor<1280x1x1x2560xf32>
    %3311 = "tosa.conv2d"(%3308, %3310, %arg363) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x2560xf32>, tensor<1280x1x1x2560xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %3312 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3313 = "tosa.transpose"(%3311, %3312) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %3314 = "tosa.add"(%3313, %3306) : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3315 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x16x16xf32>}> : () -> tensor<1x1280x16x16xf32>
    %3316 = "tosa.reciprocal"(%3315) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3317 = "tosa.mul"(%3314, %3316) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3318 = "tosa.reshape"(%3317) <{new_shape = array<i64: 1, 32, 40, 256>}> : (tensor<1x1280x16x16xf32>) -> tensor<1x32x40x256xf32>
    %3319 = "tosa.reduce_sum"(%3318) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %3320 = "tosa.reduce_sum"(%3319) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3321 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3322 = "tosa.reciprocal"(%3321) : (tensor<1xf32>) -> tensor<1xf32>
    %3323 = "tosa.mul"(%3322, %3320) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3324 = "tosa.sub"(%3318, %3323) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3325 = "tosa.mul"(%3324, %3324) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x40x256xf32>) -> tensor<1x32x40x256xf32>
    %3326 = "tosa.reduce_sum"(%3325) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %3327 = "tosa.reduce_sum"(%3326) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3328 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3329 = "tosa.reciprocal"(%3328) : (tensor<1xf32>) -> tensor<1xf32>
    %3330 = "tosa.mul"(%3329, %3327) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3331 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %3332 = "tosa.add"(%3330, %3331) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3333 = "tosa.rsqrt"(%3332) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3334 = "tosa.sub"(%3318, %3323) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3335 = "tosa.mul"(%3334, %3333) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3336 = "tosa.reshape"(%3335) <{new_shape = array<i64: 1, 1280, 16, 16>}> : (tensor<1x32x40x256xf32>) -> tensor<1x1280x16x16xf32>
    %3337 = "tosa.reshape"(%arg364) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3338 = "tosa.reshape"(%3337) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3339 = "tosa.reshape"(%3338) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3340 = "tosa.reshape"(%arg365) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3341 = "tosa.reshape"(%3340) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3342 = "tosa.reshape"(%3341) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3343 = "tosa.mul"(%3336, %3342) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %3344 = "tosa.add"(%3343, %3339) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %3345 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3346 = "tosa.transpose"(%3344, %3345) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %3347 = "tosa.reshape"(%3346) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x16x16x1280xf32>) -> tensor<1x256x1280xf32>
    %3348 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3349 = "tosa.transpose"(%arg366, %3348) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3350 = "tosa.reshape"(%3347) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3351 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %3352 = "linalg.matmul"(%3350, %3349, %3351) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3353 = "tosa.reshape"(%3352) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3354 = "tosa.reshape"(%arg367) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3355 = "tosa.add"(%3353, %3354) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3356 = "tosa.reduce_sum"(%3355) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3357 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3358 = "tosa.reciprocal"(%3357) : (tensor<1xf32>) -> tensor<1xf32>
    %3359 = "tosa.mul"(%3358, %3356) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3360 = "tosa.sub"(%3355, %3359) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3361 = "tosa.mul"(%3360, %3360) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3362 = "tosa.reduce_sum"(%3361) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3363 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3364 = "tosa.reciprocal"(%3363) : (tensor<1xf32>) -> tensor<1xf32>
    %3365 = "tosa.mul"(%3364, %3362) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3366 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %3367 = "tosa.add"(%3365, %3366) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3368 = "tosa.rsqrt"(%3367) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3369 = "tosa.sub"(%3355, %3359) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3370 = "tosa.mul"(%3369, %3368) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3371 = "tosa.reshape"(%arg368) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3372 = "tosa.mul"(%3370, %3371) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3373 = "tosa.reshape"(%arg369) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3374 = "tosa.add"(%3372, %3373) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3375 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3376 = "tosa.transpose"(%arg370, %3375) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3377 = "tosa.reshape"(%3374) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3378 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %3379 = "linalg.matmul"(%3377, %3376, %3378) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3380 = "tosa.reshape"(%3379) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3381 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3382 = "tosa.transpose"(%arg371, %3381) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3383 = "tosa.reshape"(%3374) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3384 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %3385 = "linalg.matmul"(%3383, %3382, %3384) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3386 = "tosa.reshape"(%3385) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3387 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3388 = "tosa.transpose"(%arg372, %3387) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3389 = "tosa.reshape"(%3374) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3390 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %3391 = "linalg.matmul"(%3389, %3388, %3390) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3392 = "tosa.reshape"(%3391) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3393 = "tosa.reshape"(%3380) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %3394 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3395 = "tosa.transpose"(%3393, %3394) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %3396 = "tosa.reshape"(%3386) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %3397 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3398 = "tosa.transpose"(%3396, %3397) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %3399 = "tosa.reshape"(%3392) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %3400 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3401 = "tosa.transpose"(%3399, %3400) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %3402 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x256xf32>}> : () -> tensor<256x256xf32>
    %3403 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3404 = "tosa.transpose"(%3398, %3403) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x20x64x256xf32>
    %3405 = "tosa.reshape"(%3395) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %3406 = "tosa.reshape"(%3404) <{new_shape = array<i64: 20, 64, 256>}> : (tensor<1x20x64x256xf32>) -> tensor<20x64x256xf32>
    %3407 = "tosa.matmul"(%3405, %3406) : (tensor<20x256x64xf32>, tensor<20x64x256xf32>) -> tensor<20x256x256xf32>
    %3408 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x256x256xf32>}> : () -> tensor<20x256x256xf32>
    %3409 = "tosa.mul"(%3407, %3408) <{shift = 0 : i8}> : (tensor<20x256x256xf32>, tensor<20x256x256xf32>) -> tensor<20x256x256xf32>
    %3410 = "tosa.add"(%3409, %3402) : (tensor<20x256x256xf32>, tensor<256x256xf32>) -> tensor<20x256x256xf32>
    %3411 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x256x256xf32>}> : () -> tensor<20x256x256xf32>
    %3412 = "linalg.softmax"(%3410, %3411) <{dimension = 3 : i64}> : (tensor<20x256x256xf32>, tensor<20x256x256xf32>) -> tensor<20x256x256xf32>
    %3413 = "tosa.reshape"(%3401) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %3414 = "tosa.matmul"(%3412, %3413) : (tensor<20x256x256xf32>, tensor<20x256x64xf32>) -> tensor<20x256x64xf32>
    %3415 = "tosa.reshape"(%3414) <{new_shape = array<i64: 1, 20, 256, 64>}> : (tensor<20x256x64xf32>) -> tensor<1x20x256x64xf32>
    %3416 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3417 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3418 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3419 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3420 = "tosa.transpose"(%3415, %3419) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x256x20x64xf32>
    %3421 = "tosa.reshape"(%3420) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x256x20x64xf32>) -> tensor<1x256x1280xf32>
    %3422 = "tosa.reshape"(%3421) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3423 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3424 = "tosa.transpose"(%arg373, %3423) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3425 = "tosa.reshape"(%3422) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3426 = "tosa.reshape"(%3424) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %3427 = "tosa.matmul"(%3425, %3426) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %3428 = "tosa.reshape"(%3427) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3429 = "tosa.reshape"(%arg374) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3430 = "tosa.add"(%3429, %3428) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3431 = "tosa.reshape"(%3430) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3432 = "tosa.identity"(%3431) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3433 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x1280xf32>}> : () -> tensor<1x256x1280xf32>
    %3434 = "tosa.reciprocal"(%3433) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3435 = "tosa.mul"(%3432, %3434) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3436 = "tosa.add"(%3435, %3355) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3437 = "tosa.reduce_sum"(%3436) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3438 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3439 = "tosa.reciprocal"(%3438) : (tensor<1xf32>) -> tensor<1xf32>
    %3440 = "tosa.mul"(%3439, %3437) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3441 = "tosa.sub"(%3436, %3440) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3442 = "tosa.mul"(%3441, %3441) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3443 = "tosa.reduce_sum"(%3442) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3444 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3445 = "tosa.reciprocal"(%3444) : (tensor<1xf32>) -> tensor<1xf32>
    %3446 = "tosa.mul"(%3445, %3443) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3447 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %3448 = "tosa.add"(%3446, %3447) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3449 = "tosa.rsqrt"(%3448) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3450 = "tosa.sub"(%3436, %3440) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3451 = "tosa.mul"(%3450, %3449) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3452 = "tosa.reshape"(%arg375) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3453 = "tosa.mul"(%3451, %3452) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3454 = "tosa.reshape"(%arg376) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3455 = "tosa.add"(%3453, %3454) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3456 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3457 = "tosa.transpose"(%arg377, %3456) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3458 = "tosa.reshape"(%3455) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3459 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %3460 = "linalg.matmul"(%3458, %3457, %3459) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3461 = "tosa.reshape"(%3460) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3462 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3463 = "tosa.transpose"(%arg378, %3462) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %3464 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3465 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %3466 = "linalg.matmul"(%3464, %3463, %3465) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %3467 = "tosa.reshape"(%3466) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %3468 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3469 = "tosa.transpose"(%arg380, %3468) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %3470 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3471 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %3472 = "linalg.matmul"(%3470, %3469, %3471) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %3473 = "tosa.reshape"(%3472) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %3474 = "tosa.reshape"(%3461) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %3475 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3476 = "tosa.transpose"(%3474, %3475) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %3477 = "tosa.reshape"(%3467) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %3478 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3479 = "tosa.transpose"(%3477, %3478) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %3480 = "tosa.reshape"(%3473) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %3481 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3482 = "tosa.transpose"(%3480, %3481) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %3483 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x12xf32>}> : () -> tensor<256x12xf32>
    %3484 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3485 = "tosa.transpose"(%3479, %3484) : (tensor<1x20x12x64xf32>, tensor<4xi32>) -> tensor<1x20x64x12xf32>
    %3486 = "tosa.reshape"(%3476) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %3487 = "tosa.reshape"(%3485) <{new_shape = array<i64: 20, 64, 12>}> : (tensor<1x20x64x12xf32>) -> tensor<20x64x12xf32>
    %3488 = "tosa.matmul"(%3486, %3487) : (tensor<20x256x64xf32>, tensor<20x64x12xf32>) -> tensor<20x256x12xf32>
    %3489 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x256x12xf32>}> : () -> tensor<20x256x12xf32>
    %3490 = "tosa.mul"(%3488, %3489) <{shift = 0 : i8}> : (tensor<20x256x12xf32>, tensor<20x256x12xf32>) -> tensor<20x256x12xf32>
    %3491 = "tosa.add"(%3490, %3483) : (tensor<20x256x12xf32>, tensor<256x12xf32>) -> tensor<20x256x12xf32>
    %3492 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x256x12xf32>}> : () -> tensor<20x256x12xf32>
    %3493 = "linalg.softmax"(%3491, %3492) <{dimension = 3 : i64}> : (tensor<20x256x12xf32>, tensor<20x256x12xf32>) -> tensor<20x256x12xf32>
    %3494 = "tosa.reshape"(%3482) <{new_shape = array<i64: 20, 12, 64>}> : (tensor<1x20x12x64xf32>) -> tensor<20x12x64xf32>
    %3495 = "tosa.matmul"(%3493, %3494) : (tensor<20x256x12xf32>, tensor<20x12x64xf32>) -> tensor<20x256x64xf32>
    %3496 = "tosa.reshape"(%3495) <{new_shape = array<i64: 1, 20, 256, 64>}> : (tensor<20x256x64xf32>) -> tensor<1x20x256x64xf32>
    %3497 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3498 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3499 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3500 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3501 = "tosa.transpose"(%3496, %3500) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x256x20x64xf32>
    %3502 = "tosa.reshape"(%3501) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x256x20x64xf32>) -> tensor<1x256x1280xf32>
    %3503 = "tosa.reshape"(%3502) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3504 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3505 = "tosa.transpose"(%arg382, %3504) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3506 = "tosa.reshape"(%3503) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3507 = "tosa.reshape"(%3505) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %3508 = "tosa.matmul"(%3506, %3507) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %3509 = "tosa.reshape"(%3508) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3510 = "tosa.reshape"(%arg383) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3511 = "tosa.add"(%3510, %3509) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3512 = "tosa.reshape"(%3511) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3513 = "tosa.identity"(%3512) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3514 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x1280xf32>}> : () -> tensor<1x256x1280xf32>
    %3515 = "tosa.reciprocal"(%3514) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3516 = "tosa.mul"(%3513, %3515) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3517 = "tosa.add"(%3516, %3436) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3518 = "tosa.reduce_sum"(%3517) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3519 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3520 = "tosa.reciprocal"(%3519) : (tensor<1xf32>) -> tensor<1xf32>
    %3521 = "tosa.mul"(%3520, %3518) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3522 = "tosa.sub"(%3517, %3521) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3523 = "tosa.mul"(%3522, %3522) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3524 = "tosa.reduce_sum"(%3523) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3525 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3526 = "tosa.reciprocal"(%3525) : (tensor<1xf32>) -> tensor<1xf32>
    %3527 = "tosa.mul"(%3526, %3524) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3528 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %3529 = "tosa.add"(%3527, %3528) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3530 = "tosa.rsqrt"(%3529) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3531 = "tosa.sub"(%3517, %3521) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3532 = "tosa.mul"(%3531, %3530) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3533 = "tosa.reshape"(%arg384) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3534 = "tosa.mul"(%3532, %3533) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3535 = "tosa.reshape"(%arg385) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3536 = "tosa.add"(%3534, %3535) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3537 = "tosa.reshape"(%3536) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3538 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3539 = "tosa.transpose"(%arg386, %3538) : (tensor<10240x1280xf32>, tensor<2xi32>) -> tensor<1280x10240xf32>
    %3540 = "tosa.reshape"(%3537) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3541 = "tosa.reshape"(%3539) <{new_shape = array<i64: 1, 1280, 10240>}> : (tensor<1280x10240xf32>) -> tensor<1x1280x10240xf32>
    %3542 = "tosa.matmul"(%3540, %3541) : (tensor<1x256x1280xf32>, tensor<1x1280x10240xf32>) -> tensor<1x256x10240xf32>
    %3543 = "tosa.reshape"(%3542) <{new_shape = array<i64: 256, 10240>}> : (tensor<1x256x10240xf32>) -> tensor<256x10240xf32>
    %3544 = "tosa.reshape"(%arg387) <{new_shape = array<i64: 1, 10240>}> : (tensor<10240xf32>) -> tensor<1x10240xf32>
    %3545 = "tosa.add"(%3544, %3543) : (tensor<1x10240xf32>, tensor<256x10240xf32>) -> tensor<256x10240xf32>
    %3546 = "tosa.reshape"(%3545) <{new_shape = array<i64: 1, 256, 10240>}> : (tensor<256x10240xf32>) -> tensor<1x256x10240xf32>
    %3547 = "tosa.slice"(%3546) <{size = array<i64: 0, 0, 5120>, start = array<i64: 0, 0, 0>}> : (tensor<1x256x10240xf32>) -> tensor<1x256x5120xf32>
    %3548 = "tosa.slice"(%3546) <{size = array<i64: 0, 0, 10240>, start = array<i64: 0, 0, 5120>}> : (tensor<1x256x10240xf32>) -> tensor<1x256x5120xf32>
    %3549 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %3550 = "tosa.mul"(%3548, %3549) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3551 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %3552 = "tosa.mul"(%3548, %3551) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3553 = "math.erf"(%3552) <{fastmath = #arith.fastmath<none>}> : (tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3554 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %3555 = "tosa.add"(%3553, %3554) : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3556 = "tosa.mul"(%3550, %3555) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3557 = "tosa.mul"(%3547, %3556) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3558 = "tosa.identity"(%3557) : (tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3559 = "tosa.reshape"(%3558) <{new_shape = array<i64: 256, 5120>}> : (tensor<1x256x5120xf32>) -> tensor<256x5120xf32>
    %3560 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3561 = "tosa.transpose"(%arg388, %3560) : (tensor<1280x5120xf32>, tensor<2xi32>) -> tensor<5120x1280xf32>
    %3562 = "tosa.reshape"(%3559) <{new_shape = array<i64: 1, 256, 5120>}> : (tensor<256x5120xf32>) -> tensor<1x256x5120xf32>
    %3563 = "tosa.reshape"(%3561) <{new_shape = array<i64: 1, 5120, 1280>}> : (tensor<5120x1280xf32>) -> tensor<1x5120x1280xf32>
    %3564 = "tosa.matmul"(%3562, %3563) : (tensor<1x256x5120xf32>, tensor<1x5120x1280xf32>) -> tensor<1x256x1280xf32>
    %3565 = "tosa.reshape"(%3564) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3566 = "tosa.reshape"(%arg389) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3567 = "tosa.add"(%3566, %3565) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3568 = "tosa.reshape"(%3567) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3569 = "tosa.add"(%3568, %3517) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3570 = "tosa.reshape"(%3569) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3571 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3572 = "tosa.transpose"(%arg390, %3571) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3573 = "tosa.reshape"(%3570) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3574 = "tosa.reshape"(%3572) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %3575 = "tosa.matmul"(%3573, %3574) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %3576 = "tosa.reshape"(%3575) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3577 = "tosa.reshape"(%arg391) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3578 = "tosa.add"(%3577, %3576) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3579 = "tosa.reshape"(%3578) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3580 = "tosa.reshape"(%3579) <{new_shape = array<i64: 1, 16, 16, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<1x16x16x1280xf32>
    %3581 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3582 = "tosa.transpose"(%3580, %3581) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %3583 = "tosa.identity"(%3582) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3584 = "tosa.add"(%3583, %3317) : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3585 = "tensor.empty"() : () -> tensor<1x2560x16x16xf32>
    %3586 = "tensor.insert_slice"(%3584, %3585) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1280, 16, 16>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x16x16xf32>, tensor<1x2560x16x16xf32>) -> tensor<1x2560x16x16xf32>
    %3587 = "tensor.insert_slice"(%1880, %3586) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 1280, 0, 0>, static_sizes = array<i64: 1, 1280, 16, 16>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x16x16xf32>, tensor<1x2560x16x16xf32>) -> tensor<1x2560x16x16xf32>
    %3588 = "tosa.reshape"(%3587) <{new_shape = array<i64: 1, 32, 80, 256>}> : (tensor<1x2560x16x16xf32>) -> tensor<1x32x80x256xf32>
    %3589 = "tosa.reduce_sum"(%3588) <{axis = 2 : i32}> : (tensor<1x32x80x256xf32>) -> tensor<1x32x1x256xf32>
    %3590 = "tosa.reduce_sum"(%3589) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3591 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3592 = "tosa.reciprocal"(%3591) : (tensor<1xf32>) -> tensor<1xf32>
    %3593 = "tosa.mul"(%3592, %3590) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3594 = "tosa.sub"(%3588, %3593) : (tensor<1x32x80x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x256xf32>
    %3595 = "tosa.mul"(%3594, %3594) <{shift = 0 : i8}> : (tensor<1x32x80x256xf32>, tensor<1x32x80x256xf32>) -> tensor<1x32x80x256xf32>
    %3596 = "tosa.reduce_sum"(%3595) <{axis = 2 : i32}> : (tensor<1x32x80x256xf32>) -> tensor<1x32x1x256xf32>
    %3597 = "tosa.reduce_sum"(%3596) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3598 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3599 = "tosa.reciprocal"(%3598) : (tensor<1xf32>) -> tensor<1xf32>
    %3600 = "tosa.mul"(%3599, %3597) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3601 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %3602 = "tosa.add"(%3600, %3601) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3603 = "tosa.rsqrt"(%3602) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3604 = "tosa.sub"(%3588, %3593) : (tensor<1x32x80x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x256xf32>
    %3605 = "tosa.mul"(%3604, %3603) <{shift = 0 : i8}> : (tensor<1x32x80x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x80x256xf32>
    %3606 = "tosa.reshape"(%3605) <{new_shape = array<i64: 1, 2560, 16, 16>}> : (tensor<1x32x80x256xf32>) -> tensor<1x2560x16x16xf32>
    %3607 = "tosa.reshape"(%arg392) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %3608 = "tosa.reshape"(%3607) <{new_shape = array<i64: 1, 2560, 1>}> : (tensor<1x2560xf32>) -> tensor<1x2560x1xf32>
    %3609 = "tosa.reshape"(%3608) <{new_shape = array<i64: 1, 2560, 1, 1>}> : (tensor<1x2560x1xf32>) -> tensor<1x2560x1x1xf32>
    %3610 = "tosa.reshape"(%arg393) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %3611 = "tosa.reshape"(%3610) <{new_shape = array<i64: 1, 2560, 1>}> : (tensor<1x2560xf32>) -> tensor<1x2560x1xf32>
    %3612 = "tosa.reshape"(%3611) <{new_shape = array<i64: 1, 2560, 1, 1>}> : (tensor<1x2560x1xf32>) -> tensor<1x2560x1x1xf32>
    %3613 = "tosa.mul"(%3606, %3612) <{shift = 0 : i8}> : (tensor<1x2560x16x16xf32>, tensor<1x2560x1x1xf32>) -> tensor<1x2560x16x16xf32>
    %3614 = "tosa.add"(%3613, %3609) : (tensor<1x2560x16x16xf32>, tensor<1x2560x1x1xf32>) -> tensor<1x2560x16x16xf32>
    %3615 = "tosa.sigmoid"(%3614) : (tensor<1x2560x16x16xf32>) -> tensor<1x2560x16x16xf32>
    %3616 = "tosa.mul"(%3614, %3615) <{shift = 0 : i8}> : (tensor<1x2560x16x16xf32>, tensor<1x2560x16x16xf32>) -> tensor<1x2560x16x16xf32>
    %3617 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3618 = "tosa.transpose"(%3616, %3617) : (tensor<1x2560x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x2560xf32>
    %3619 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3620 = "tosa.transpose"(%arg394, %3619) : (tensor<1280x2560x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x2560xf32>
    %3621 = "tosa.conv2d"(%3618, %3620, %arg395) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x2560xf32>, tensor<1280x3x3x2560xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %3622 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3623 = "tosa.transpose"(%3621, %3622) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %3624 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3625 = "tosa.mul"(%50, %3624) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3626 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3627 = "tosa.transpose"(%arg396, %3626) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3628 = "tosa.reshape"(%3625) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %3629 = "tosa.reshape"(%3627) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %3630 = "tosa.matmul"(%3628, %3629) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %3631 = "tosa.reshape"(%3630) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %3632 = "tosa.reshape"(%arg397) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3633 = "tosa.add"(%3632, %3631) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3634 = "tensor.extract_slice"(%3633) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3635 = "tensor.extract_slice"(%3634) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3636 = "tosa.reshape"(%3635) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3637 = "tosa.reshape"(%3636) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3638 = "tosa.add"(%3623, %3637) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %3639 = "tosa.reshape"(%3638) <{new_shape = array<i64: 1, 32, 40, 256>}> : (tensor<1x1280x16x16xf32>) -> tensor<1x32x40x256xf32>
    %3640 = "tosa.reduce_sum"(%3639) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %3641 = "tosa.reduce_sum"(%3640) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3642 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3643 = "tosa.reciprocal"(%3642) : (tensor<1xf32>) -> tensor<1xf32>
    %3644 = "tosa.mul"(%3643, %3641) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3645 = "tosa.sub"(%3639, %3644) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3646 = "tosa.mul"(%3645, %3645) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x40x256xf32>) -> tensor<1x32x40x256xf32>
    %3647 = "tosa.reduce_sum"(%3646) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %3648 = "tosa.reduce_sum"(%3647) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3649 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3650 = "tosa.reciprocal"(%3649) : (tensor<1xf32>) -> tensor<1xf32>
    %3651 = "tosa.mul"(%3650, %3648) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3652 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %3653 = "tosa.add"(%3651, %3652) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3654 = "tosa.rsqrt"(%3653) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3655 = "tosa.sub"(%3639, %3644) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3656 = "tosa.mul"(%3655, %3654) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3657 = "tosa.reshape"(%3656) <{new_shape = array<i64: 1, 1280, 16, 16>}> : (tensor<1x32x40x256xf32>) -> tensor<1x1280x16x16xf32>
    %3658 = "tosa.reshape"(%arg398) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3659 = "tosa.reshape"(%3658) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3660 = "tosa.reshape"(%3659) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3661 = "tosa.reshape"(%arg399) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3662 = "tosa.reshape"(%3661) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3663 = "tosa.reshape"(%3662) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3664 = "tosa.mul"(%3657, %3663) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %3665 = "tosa.add"(%3664, %3660) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %3666 = "tosa.sigmoid"(%3665) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3667 = "tosa.mul"(%3665, %3666) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3668 = "tosa.identity"(%3667) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3669 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3670 = "tosa.transpose"(%3668, %3669) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %3671 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3672 = "tosa.transpose"(%arg400, %3671) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %3673 = "tosa.conv2d"(%3670, %3672, %arg401) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %3674 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3675 = "tosa.transpose"(%3673, %3674) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %3676 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3677 = "tosa.transpose"(%3587, %3676) : (tensor<1x2560x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x2560xf32>
    %3678 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3679 = "tosa.transpose"(%arg402, %3678) : (tensor<1280x2560x1x1xf32>, tensor<4xi32>) -> tensor<1280x1x1x2560xf32>
    %3680 = "tosa.conv2d"(%3677, %3679, %arg403) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x2560xf32>, tensor<1280x1x1x2560xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %3681 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3682 = "tosa.transpose"(%3680, %3681) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %3683 = "tosa.add"(%3682, %3675) : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3684 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x16x16xf32>}> : () -> tensor<1x1280x16x16xf32>
    %3685 = "tosa.reciprocal"(%3684) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3686 = "tosa.mul"(%3683, %3685) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3687 = "tosa.reshape"(%3686) <{new_shape = array<i64: 1, 32, 40, 256>}> : (tensor<1x1280x16x16xf32>) -> tensor<1x32x40x256xf32>
    %3688 = "tosa.reduce_sum"(%3687) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %3689 = "tosa.reduce_sum"(%3688) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3690 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3691 = "tosa.reciprocal"(%3690) : (tensor<1xf32>) -> tensor<1xf32>
    %3692 = "tosa.mul"(%3691, %3689) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3693 = "tosa.sub"(%3687, %3692) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3694 = "tosa.mul"(%3693, %3693) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x40x256xf32>) -> tensor<1x32x40x256xf32>
    %3695 = "tosa.reduce_sum"(%3694) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %3696 = "tosa.reduce_sum"(%3695) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3697 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3698 = "tosa.reciprocal"(%3697) : (tensor<1xf32>) -> tensor<1xf32>
    %3699 = "tosa.mul"(%3698, %3696) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3700 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %3701 = "tosa.add"(%3699, %3700) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3702 = "tosa.rsqrt"(%3701) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3703 = "tosa.sub"(%3687, %3692) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3704 = "tosa.mul"(%3703, %3702) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %3705 = "tosa.reshape"(%3704) <{new_shape = array<i64: 1, 1280, 16, 16>}> : (tensor<1x32x40x256xf32>) -> tensor<1x1280x16x16xf32>
    %3706 = "tosa.reshape"(%arg404) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3707 = "tosa.reshape"(%3706) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3708 = "tosa.reshape"(%3707) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3709 = "tosa.reshape"(%arg405) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3710 = "tosa.reshape"(%3709) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %3711 = "tosa.reshape"(%3710) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %3712 = "tosa.mul"(%3705, %3711) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %3713 = "tosa.add"(%3712, %3708) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %3714 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3715 = "tosa.transpose"(%3713, %3714) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %3716 = "tosa.reshape"(%3715) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x16x16x1280xf32>) -> tensor<1x256x1280xf32>
    %3717 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3718 = "tosa.transpose"(%arg406, %3717) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3719 = "tosa.reshape"(%3716) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3720 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %3721 = "linalg.matmul"(%3719, %3718, %3720) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3722 = "tosa.reshape"(%3721) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3723 = "tosa.reshape"(%arg407) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3724 = "tosa.add"(%3722, %3723) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3725 = "tosa.reduce_sum"(%3724) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3726 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3727 = "tosa.reciprocal"(%3726) : (tensor<1xf32>) -> tensor<1xf32>
    %3728 = "tosa.mul"(%3727, %3725) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3729 = "tosa.sub"(%3724, %3728) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3730 = "tosa.mul"(%3729, %3729) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3731 = "tosa.reduce_sum"(%3730) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3732 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3733 = "tosa.reciprocal"(%3732) : (tensor<1xf32>) -> tensor<1xf32>
    %3734 = "tosa.mul"(%3733, %3731) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3735 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %3736 = "tosa.add"(%3734, %3735) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3737 = "tosa.rsqrt"(%3736) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3738 = "tosa.sub"(%3724, %3728) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3739 = "tosa.mul"(%3738, %3737) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3740 = "tosa.reshape"(%arg408) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3741 = "tosa.mul"(%3739, %3740) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3742 = "tosa.reshape"(%arg409) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3743 = "tosa.add"(%3741, %3742) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3744 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3745 = "tosa.transpose"(%arg410, %3744) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3746 = "tosa.reshape"(%3743) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3747 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %3748 = "linalg.matmul"(%3746, %3745, %3747) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3749 = "tosa.reshape"(%3748) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3750 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3751 = "tosa.transpose"(%arg411, %3750) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3752 = "tosa.reshape"(%3743) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3753 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %3754 = "linalg.matmul"(%3752, %3751, %3753) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3755 = "tosa.reshape"(%3754) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3756 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3757 = "tosa.transpose"(%arg412, %3756) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3758 = "tosa.reshape"(%3743) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3759 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %3760 = "linalg.matmul"(%3758, %3757, %3759) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3761 = "tosa.reshape"(%3760) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3762 = "tosa.reshape"(%3749) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %3763 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3764 = "tosa.transpose"(%3762, %3763) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %3765 = "tosa.reshape"(%3755) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %3766 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3767 = "tosa.transpose"(%3765, %3766) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %3768 = "tosa.reshape"(%3761) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %3769 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3770 = "tosa.transpose"(%3768, %3769) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %3771 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x256xf32>}> : () -> tensor<256x256xf32>
    %3772 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3773 = "tosa.transpose"(%3767, %3772) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x20x64x256xf32>
    %3774 = "tosa.reshape"(%3764) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %3775 = "tosa.reshape"(%3773) <{new_shape = array<i64: 20, 64, 256>}> : (tensor<1x20x64x256xf32>) -> tensor<20x64x256xf32>
    %3776 = "tosa.matmul"(%3774, %3775) : (tensor<20x256x64xf32>, tensor<20x64x256xf32>) -> tensor<20x256x256xf32>
    %3777 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x256x256xf32>}> : () -> tensor<20x256x256xf32>
    %3778 = "tosa.mul"(%3776, %3777) <{shift = 0 : i8}> : (tensor<20x256x256xf32>, tensor<20x256x256xf32>) -> tensor<20x256x256xf32>
    %3779 = "tosa.add"(%3778, %3771) : (tensor<20x256x256xf32>, tensor<256x256xf32>) -> tensor<20x256x256xf32>
    %3780 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x256x256xf32>}> : () -> tensor<20x256x256xf32>
    %3781 = "linalg.softmax"(%3779, %3780) <{dimension = 3 : i64}> : (tensor<20x256x256xf32>, tensor<20x256x256xf32>) -> tensor<20x256x256xf32>
    %3782 = "tosa.reshape"(%3770) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %3783 = "tosa.matmul"(%3781, %3782) : (tensor<20x256x256xf32>, tensor<20x256x64xf32>) -> tensor<20x256x64xf32>
    %3784 = "tosa.reshape"(%3783) <{new_shape = array<i64: 1, 20, 256, 64>}> : (tensor<20x256x64xf32>) -> tensor<1x20x256x64xf32>
    %3785 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3786 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3787 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3788 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3789 = "tosa.transpose"(%3784, %3788) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x256x20x64xf32>
    %3790 = "tosa.reshape"(%3789) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x256x20x64xf32>) -> tensor<1x256x1280xf32>
    %3791 = "tosa.reshape"(%3790) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3792 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3793 = "tosa.transpose"(%arg413, %3792) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3794 = "tosa.reshape"(%3791) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3795 = "tosa.reshape"(%3793) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %3796 = "tosa.matmul"(%3794, %3795) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %3797 = "tosa.reshape"(%3796) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3798 = "tosa.reshape"(%arg414) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3799 = "tosa.add"(%3798, %3797) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3800 = "tosa.reshape"(%3799) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3801 = "tosa.identity"(%3800) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3802 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x1280xf32>}> : () -> tensor<1x256x1280xf32>
    %3803 = "tosa.reciprocal"(%3802) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3804 = "tosa.mul"(%3801, %3803) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3805 = "tosa.add"(%3804, %3724) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3806 = "tosa.reduce_sum"(%3805) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3807 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3808 = "tosa.reciprocal"(%3807) : (tensor<1xf32>) -> tensor<1xf32>
    %3809 = "tosa.mul"(%3808, %3806) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3810 = "tosa.sub"(%3805, %3809) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3811 = "tosa.mul"(%3810, %3810) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3812 = "tosa.reduce_sum"(%3811) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3813 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3814 = "tosa.reciprocal"(%3813) : (tensor<1xf32>) -> tensor<1xf32>
    %3815 = "tosa.mul"(%3814, %3812) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3816 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %3817 = "tosa.add"(%3815, %3816) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3818 = "tosa.rsqrt"(%3817) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3819 = "tosa.sub"(%3805, %3809) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3820 = "tosa.mul"(%3819, %3818) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3821 = "tosa.reshape"(%arg415) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3822 = "tosa.mul"(%3820, %3821) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3823 = "tosa.reshape"(%arg416) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3824 = "tosa.add"(%3822, %3823) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3825 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3826 = "tosa.transpose"(%arg417, %3825) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3827 = "tosa.reshape"(%3824) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3828 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %3829 = "linalg.matmul"(%3827, %3826, %3828) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3830 = "tosa.reshape"(%3829) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3831 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3832 = "tosa.transpose"(%arg418, %3831) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %3833 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3834 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %3835 = "linalg.matmul"(%3833, %3832, %3834) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %3836 = "tosa.reshape"(%3835) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %3837 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3838 = "tosa.transpose"(%arg420, %3837) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %3839 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %3840 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %3841 = "linalg.matmul"(%3839, %3838, %3840) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %3842 = "tosa.reshape"(%3841) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %3843 = "tosa.reshape"(%3830) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %3844 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3845 = "tosa.transpose"(%3843, %3844) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %3846 = "tosa.reshape"(%3836) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %3847 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3848 = "tosa.transpose"(%3846, %3847) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %3849 = "tosa.reshape"(%3842) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %3850 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3851 = "tosa.transpose"(%3849, %3850) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %3852 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x12xf32>}> : () -> tensor<256x12xf32>
    %3853 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3854 = "tosa.transpose"(%3848, %3853) : (tensor<1x20x12x64xf32>, tensor<4xi32>) -> tensor<1x20x64x12xf32>
    %3855 = "tosa.reshape"(%3845) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %3856 = "tosa.reshape"(%3854) <{new_shape = array<i64: 20, 64, 12>}> : (tensor<1x20x64x12xf32>) -> tensor<20x64x12xf32>
    %3857 = "tosa.matmul"(%3855, %3856) : (tensor<20x256x64xf32>, tensor<20x64x12xf32>) -> tensor<20x256x12xf32>
    %3858 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x256x12xf32>}> : () -> tensor<20x256x12xf32>
    %3859 = "tosa.mul"(%3857, %3858) <{shift = 0 : i8}> : (tensor<20x256x12xf32>, tensor<20x256x12xf32>) -> tensor<20x256x12xf32>
    %3860 = "tosa.add"(%3859, %3852) : (tensor<20x256x12xf32>, tensor<256x12xf32>) -> tensor<20x256x12xf32>
    %3861 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x256x12xf32>}> : () -> tensor<20x256x12xf32>
    %3862 = "linalg.softmax"(%3860, %3861) <{dimension = 3 : i64}> : (tensor<20x256x12xf32>, tensor<20x256x12xf32>) -> tensor<20x256x12xf32>
    %3863 = "tosa.reshape"(%3851) <{new_shape = array<i64: 20, 12, 64>}> : (tensor<1x20x12x64xf32>) -> tensor<20x12x64xf32>
    %3864 = "tosa.matmul"(%3862, %3863) : (tensor<20x256x12xf32>, tensor<20x12x64xf32>) -> tensor<20x256x64xf32>
    %3865 = "tosa.reshape"(%3864) <{new_shape = array<i64: 1, 20, 256, 64>}> : (tensor<20x256x64xf32>) -> tensor<1x20x256x64xf32>
    %3866 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3867 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3868 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %3869 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3870 = "tosa.transpose"(%3865, %3869) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x256x20x64xf32>
    %3871 = "tosa.reshape"(%3870) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x256x20x64xf32>) -> tensor<1x256x1280xf32>
    %3872 = "tosa.reshape"(%3871) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3873 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3874 = "tosa.transpose"(%arg422, %3873) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3875 = "tosa.reshape"(%3872) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3876 = "tosa.reshape"(%3874) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %3877 = "tosa.matmul"(%3875, %3876) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %3878 = "tosa.reshape"(%3877) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3879 = "tosa.reshape"(%arg423) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3880 = "tosa.add"(%3879, %3878) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3881 = "tosa.reshape"(%3880) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3882 = "tosa.identity"(%3881) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3883 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x1280xf32>}> : () -> tensor<1x256x1280xf32>
    %3884 = "tosa.reciprocal"(%3883) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3885 = "tosa.mul"(%3882, %3884) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3886 = "tosa.add"(%3885, %3805) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3887 = "tosa.reduce_sum"(%3886) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3888 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3889 = "tosa.reciprocal"(%3888) : (tensor<1xf32>) -> tensor<1xf32>
    %3890 = "tosa.mul"(%3889, %3887) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3891 = "tosa.sub"(%3886, %3890) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3892 = "tosa.mul"(%3891, %3891) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3893 = "tosa.reduce_sum"(%3892) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %3894 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3895 = "tosa.reciprocal"(%3894) : (tensor<1xf32>) -> tensor<1xf32>
    %3896 = "tosa.mul"(%3895, %3893) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3897 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %3898 = "tosa.add"(%3896, %3897) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3899 = "tosa.rsqrt"(%3898) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %3900 = "tosa.sub"(%3886, %3890) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3901 = "tosa.mul"(%3900, %3899) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %3902 = "tosa.reshape"(%arg424) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3903 = "tosa.mul"(%3901, %3902) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3904 = "tosa.reshape"(%arg425) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %3905 = "tosa.add"(%3903, %3904) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %3906 = "tosa.reshape"(%3905) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3907 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3908 = "tosa.transpose"(%arg426, %3907) : (tensor<10240x1280xf32>, tensor<2xi32>) -> tensor<1280x10240xf32>
    %3909 = "tosa.reshape"(%3906) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3910 = "tosa.reshape"(%3908) <{new_shape = array<i64: 1, 1280, 10240>}> : (tensor<1280x10240xf32>) -> tensor<1x1280x10240xf32>
    %3911 = "tosa.matmul"(%3909, %3910) : (tensor<1x256x1280xf32>, tensor<1x1280x10240xf32>) -> tensor<1x256x10240xf32>
    %3912 = "tosa.reshape"(%3911) <{new_shape = array<i64: 256, 10240>}> : (tensor<1x256x10240xf32>) -> tensor<256x10240xf32>
    %3913 = "tosa.reshape"(%arg427) <{new_shape = array<i64: 1, 10240>}> : (tensor<10240xf32>) -> tensor<1x10240xf32>
    %3914 = "tosa.add"(%3913, %3912) : (tensor<1x10240xf32>, tensor<256x10240xf32>) -> tensor<256x10240xf32>
    %3915 = "tosa.reshape"(%3914) <{new_shape = array<i64: 1, 256, 10240>}> : (tensor<256x10240xf32>) -> tensor<1x256x10240xf32>
    %3916 = "tosa.slice"(%3915) <{size = array<i64: 0, 0, 5120>, start = array<i64: 0, 0, 0>}> : (tensor<1x256x10240xf32>) -> tensor<1x256x5120xf32>
    %3917 = "tosa.slice"(%3915) <{size = array<i64: 0, 0, 10240>, start = array<i64: 0, 0, 5120>}> : (tensor<1x256x10240xf32>) -> tensor<1x256x5120xf32>
    %3918 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %3919 = "tosa.mul"(%3917, %3918) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3920 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %3921 = "tosa.mul"(%3917, %3920) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3922 = "math.erf"(%3921) <{fastmath = #arith.fastmath<none>}> : (tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3923 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %3924 = "tosa.add"(%3922, %3923) : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3925 = "tosa.mul"(%3919, %3924) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3926 = "tosa.mul"(%3916, %3925) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3927 = "tosa.identity"(%3926) : (tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %3928 = "tosa.reshape"(%3927) <{new_shape = array<i64: 256, 5120>}> : (tensor<1x256x5120xf32>) -> tensor<256x5120xf32>
    %3929 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3930 = "tosa.transpose"(%arg428, %3929) : (tensor<1280x5120xf32>, tensor<2xi32>) -> tensor<5120x1280xf32>
    %3931 = "tosa.reshape"(%3928) <{new_shape = array<i64: 1, 256, 5120>}> : (tensor<256x5120xf32>) -> tensor<1x256x5120xf32>
    %3932 = "tosa.reshape"(%3930) <{new_shape = array<i64: 1, 5120, 1280>}> : (tensor<5120x1280xf32>) -> tensor<1x5120x1280xf32>
    %3933 = "tosa.matmul"(%3931, %3932) : (tensor<1x256x5120xf32>, tensor<1x5120x1280xf32>) -> tensor<1x256x1280xf32>
    %3934 = "tosa.reshape"(%3933) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3935 = "tosa.reshape"(%arg429) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3936 = "tosa.add"(%3935, %3934) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3937 = "tosa.reshape"(%3936) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3938 = "tosa.add"(%3937, %3886) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %3939 = "tosa.reshape"(%3938) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3940 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3941 = "tosa.transpose"(%arg430, %3940) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3942 = "tosa.reshape"(%3939) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3943 = "tosa.reshape"(%3941) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %3944 = "tosa.matmul"(%3942, %3943) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %3945 = "tosa.reshape"(%3944) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %3946 = "tosa.reshape"(%arg431) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %3947 = "tosa.add"(%3946, %3945) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %3948 = "tosa.reshape"(%3947) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %3949 = "tosa.reshape"(%3948) <{new_shape = array<i64: 1, 16, 16, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<1x16x16x1280xf32>
    %3950 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3951 = "tosa.transpose"(%3949, %3950) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %3952 = "tosa.identity"(%3951) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3953 = "tosa.add"(%3952, %3686) : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %3954 = "tensor.empty"() : () -> tensor<1x1920x16x16xf32>
    %3955 = "tensor.insert_slice"(%3953, %3954) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1280, 16, 16>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x16x16xf32>, tensor<1x1920x16x16xf32>) -> tensor<1x1920x16x16xf32>
    %3956 = "tensor.insert_slice"(%1514, %3955) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 1280, 0, 0>, static_sizes = array<i64: 1, 640, 16, 16>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x640x16x16xf32>, tensor<1x1920x16x16xf32>) -> tensor<1x1920x16x16xf32>
    %3957 = "tosa.reshape"(%3956) <{new_shape = array<i64: 1, 32, 60, 256>}> : (tensor<1x1920x16x16xf32>) -> tensor<1x32x60x256xf32>
    %3958 = "tosa.reduce_sum"(%3957) <{axis = 2 : i32}> : (tensor<1x32x60x256xf32>) -> tensor<1x32x1x256xf32>
    %3959 = "tosa.reduce_sum"(%3958) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3960 = "tosa.const"() <{value = dense<1.536000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3961 = "tosa.reciprocal"(%3960) : (tensor<1xf32>) -> tensor<1xf32>
    %3962 = "tosa.mul"(%3961, %3959) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3963 = "tosa.sub"(%3957, %3962) : (tensor<1x32x60x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x60x256xf32>
    %3964 = "tosa.mul"(%3963, %3963) <{shift = 0 : i8}> : (tensor<1x32x60x256xf32>, tensor<1x32x60x256xf32>) -> tensor<1x32x60x256xf32>
    %3965 = "tosa.reduce_sum"(%3964) <{axis = 2 : i32}> : (tensor<1x32x60x256xf32>) -> tensor<1x32x1x256xf32>
    %3966 = "tosa.reduce_sum"(%3965) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %3967 = "tosa.const"() <{value = dense<1.536000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3968 = "tosa.reciprocal"(%3967) : (tensor<1xf32>) -> tensor<1xf32>
    %3969 = "tosa.mul"(%3968, %3966) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3970 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %3971 = "tosa.add"(%3969, %3970) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3972 = "tosa.rsqrt"(%3971) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %3973 = "tosa.sub"(%3957, %3962) : (tensor<1x32x60x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x60x256xf32>
    %3974 = "tosa.mul"(%3973, %3972) <{shift = 0 : i8}> : (tensor<1x32x60x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x60x256xf32>
    %3975 = "tosa.reshape"(%3974) <{new_shape = array<i64: 1, 1920, 16, 16>}> : (tensor<1x32x60x256xf32>) -> tensor<1x1920x16x16xf32>
    %3976 = "tosa.reshape"(%arg432) <{new_shape = array<i64: 1, 1920>}> : (tensor<1920xf32>) -> tensor<1x1920xf32>
    %3977 = "tosa.reshape"(%3976) <{new_shape = array<i64: 1, 1920, 1>}> : (tensor<1x1920xf32>) -> tensor<1x1920x1xf32>
    %3978 = "tosa.reshape"(%3977) <{new_shape = array<i64: 1, 1920, 1, 1>}> : (tensor<1x1920x1xf32>) -> tensor<1x1920x1x1xf32>
    %3979 = "tosa.reshape"(%arg433) <{new_shape = array<i64: 1, 1920>}> : (tensor<1920xf32>) -> tensor<1x1920xf32>
    %3980 = "tosa.reshape"(%3979) <{new_shape = array<i64: 1, 1920, 1>}> : (tensor<1x1920xf32>) -> tensor<1x1920x1xf32>
    %3981 = "tosa.reshape"(%3980) <{new_shape = array<i64: 1, 1920, 1, 1>}> : (tensor<1x1920x1xf32>) -> tensor<1x1920x1x1xf32>
    %3982 = "tosa.mul"(%3975, %3981) <{shift = 0 : i8}> : (tensor<1x1920x16x16xf32>, tensor<1x1920x1x1xf32>) -> tensor<1x1920x16x16xf32>
    %3983 = "tosa.add"(%3982, %3978) : (tensor<1x1920x16x16xf32>, tensor<1x1920x1x1xf32>) -> tensor<1x1920x16x16xf32>
    %3984 = "tosa.sigmoid"(%3983) : (tensor<1x1920x16x16xf32>) -> tensor<1x1920x16x16xf32>
    %3985 = "tosa.mul"(%3983, %3984) <{shift = 0 : i8}> : (tensor<1x1920x16x16xf32>, tensor<1x1920x16x16xf32>) -> tensor<1x1920x16x16xf32>
    %3986 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3987 = "tosa.transpose"(%3985, %3986) : (tensor<1x1920x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1920xf32>
    %3988 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3989 = "tosa.transpose"(%arg434, %3988) : (tensor<1280x1920x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1920xf32>
    %3990 = "tosa.conv2d"(%3987, %3989, %arg435) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x1920xf32>, tensor<1280x3x3x1920xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %3991 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3992 = "tosa.transpose"(%3990, %3991) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %3993 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3994 = "tosa.mul"(%50, %3993) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %3995 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3996 = "tosa.transpose"(%arg436, %3995) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %3997 = "tosa.reshape"(%3994) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %3998 = "tosa.reshape"(%3996) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %3999 = "tosa.matmul"(%3997, %3998) : (tensor<1x1x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x1x1280xf32>
    %4000 = "tosa.reshape"(%3999) <{new_shape = array<i64: 1, 1280>}> : (tensor<1x1x1280xf32>) -> tensor<1x1280xf32>
    %4001 = "tosa.reshape"(%arg437) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %4002 = "tosa.add"(%4001, %4000) : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %4003 = "tensor.extract_slice"(%4002) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %4004 = "tensor.extract_slice"(%4003) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1280>, static_strides = array<i64: 1, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %4005 = "tosa.reshape"(%4004) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %4006 = "tosa.reshape"(%4005) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %4007 = "tosa.add"(%3992, %4006) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %4008 = "tosa.reshape"(%4007) <{new_shape = array<i64: 1, 32, 40, 256>}> : (tensor<1x1280x16x16xf32>) -> tensor<1x32x40x256xf32>
    %4009 = "tosa.reduce_sum"(%4008) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %4010 = "tosa.reduce_sum"(%4009) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %4011 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4012 = "tosa.reciprocal"(%4011) : (tensor<1xf32>) -> tensor<1xf32>
    %4013 = "tosa.mul"(%4012, %4010) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4014 = "tosa.sub"(%4008, %4013) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %4015 = "tosa.mul"(%4014, %4014) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x40x256xf32>) -> tensor<1x32x40x256xf32>
    %4016 = "tosa.reduce_sum"(%4015) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %4017 = "tosa.reduce_sum"(%4016) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %4018 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4019 = "tosa.reciprocal"(%4018) : (tensor<1xf32>) -> tensor<1xf32>
    %4020 = "tosa.mul"(%4019, %4017) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4021 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %4022 = "tosa.add"(%4020, %4021) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4023 = "tosa.rsqrt"(%4022) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4024 = "tosa.sub"(%4008, %4013) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %4025 = "tosa.mul"(%4024, %4023) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %4026 = "tosa.reshape"(%4025) <{new_shape = array<i64: 1, 1280, 16, 16>}> : (tensor<1x32x40x256xf32>) -> tensor<1x1280x16x16xf32>
    %4027 = "tosa.reshape"(%arg438) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %4028 = "tosa.reshape"(%4027) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %4029 = "tosa.reshape"(%4028) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %4030 = "tosa.reshape"(%arg439) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %4031 = "tosa.reshape"(%4030) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %4032 = "tosa.reshape"(%4031) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %4033 = "tosa.mul"(%4026, %4032) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %4034 = "tosa.add"(%4033, %4029) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %4035 = "tosa.sigmoid"(%4034) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %4036 = "tosa.mul"(%4034, %4035) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %4037 = "tosa.identity"(%4036) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %4038 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4039 = "tosa.transpose"(%4037, %4038) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %4040 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4041 = "tosa.transpose"(%arg440, %4040) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %4042 = "tosa.conv2d"(%4039, %4041, %arg441) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %4043 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4044 = "tosa.transpose"(%4042, %4043) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %4045 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4046 = "tosa.transpose"(%3956, %4045) : (tensor<1x1920x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1920xf32>
    %4047 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4048 = "tosa.transpose"(%arg442, %4047) : (tensor<1280x1920x1x1xf32>, tensor<4xi32>) -> tensor<1280x1x1x1920xf32>
    %4049 = "tosa.conv2d"(%4046, %4048, %arg443) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x16x16x1920xf32>, tensor<1280x1x1x1920xf32>, tensor<1280xf32>) -> tensor<1x16x16x1280xf32>
    %4050 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4051 = "tosa.transpose"(%4049, %4050) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %4052 = "tosa.add"(%4051, %4044) : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %4053 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1280x16x16xf32>}> : () -> tensor<1x1280x16x16xf32>
    %4054 = "tosa.reciprocal"(%4053) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %4055 = "tosa.mul"(%4052, %4054) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %4056 = "tosa.reshape"(%4055) <{new_shape = array<i64: 1, 32, 40, 256>}> : (tensor<1x1280x16x16xf32>) -> tensor<1x32x40x256xf32>
    %4057 = "tosa.reduce_sum"(%4056) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %4058 = "tosa.reduce_sum"(%4057) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %4059 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4060 = "tosa.reciprocal"(%4059) : (tensor<1xf32>) -> tensor<1xf32>
    %4061 = "tosa.mul"(%4060, %4058) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4062 = "tosa.sub"(%4056, %4061) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %4063 = "tosa.mul"(%4062, %4062) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x40x256xf32>) -> tensor<1x32x40x256xf32>
    %4064 = "tosa.reduce_sum"(%4063) <{axis = 2 : i32}> : (tensor<1x32x40x256xf32>) -> tensor<1x32x1x256xf32>
    %4065 = "tosa.reduce_sum"(%4064) <{axis = 3 : i32}> : (tensor<1x32x1x256xf32>) -> tensor<1x32x1x1xf32>
    %4066 = "tosa.const"() <{value = dense<1.024000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4067 = "tosa.reciprocal"(%4066) : (tensor<1xf32>) -> tensor<1xf32>
    %4068 = "tosa.mul"(%4067, %4065) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4069 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %4070 = "tosa.add"(%4068, %4069) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4071 = "tosa.rsqrt"(%4070) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4072 = "tosa.sub"(%4056, %4061) : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %4073 = "tosa.mul"(%4072, %4071) <{shift = 0 : i8}> : (tensor<1x32x40x256xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x256xf32>
    %4074 = "tosa.reshape"(%4073) <{new_shape = array<i64: 1, 1280, 16, 16>}> : (tensor<1x32x40x256xf32>) -> tensor<1x1280x16x16xf32>
    %4075 = "tosa.reshape"(%arg444) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %4076 = "tosa.reshape"(%4075) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %4077 = "tosa.reshape"(%4076) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %4078 = "tosa.reshape"(%arg445) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %4079 = "tosa.reshape"(%4078) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %4080 = "tosa.reshape"(%4079) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %4081 = "tosa.mul"(%4074, %4080) <{shift = 0 : i8}> : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %4082 = "tosa.add"(%4081, %4077) : (tensor<1x1280x16x16xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x16x16xf32>
    %4083 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4084 = "tosa.transpose"(%4082, %4083) : (tensor<1x1280x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
    %4085 = "tosa.reshape"(%4084) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x16x16x1280xf32>) -> tensor<1x256x1280xf32>
    %4086 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4087 = "tosa.transpose"(%arg446, %4086) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %4088 = "tosa.reshape"(%4085) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4089 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %4090 = "linalg.matmul"(%4088, %4087, %4089) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4091 = "tosa.reshape"(%4090) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4092 = "tosa.reshape"(%arg447) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %4093 = "tosa.add"(%4091, %4092) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %4094 = "tosa.reduce_sum"(%4093) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %4095 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4096 = "tosa.reciprocal"(%4095) : (tensor<1xf32>) -> tensor<1xf32>
    %4097 = "tosa.mul"(%4096, %4094) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4098 = "tosa.sub"(%4093, %4097) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4099 = "tosa.mul"(%4098, %4098) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4100 = "tosa.reduce_sum"(%4099) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %4101 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4102 = "tosa.reciprocal"(%4101) : (tensor<1xf32>) -> tensor<1xf32>
    %4103 = "tosa.mul"(%4102, %4100) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4104 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %4105 = "tosa.add"(%4103, %4104) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4106 = "tosa.rsqrt"(%4105) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4107 = "tosa.sub"(%4093, %4097) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4108 = "tosa.mul"(%4107, %4106) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4109 = "tosa.reshape"(%arg448) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %4110 = "tosa.mul"(%4108, %4109) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %4111 = "tosa.reshape"(%arg449) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %4112 = "tosa.add"(%4110, %4111) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %4113 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4114 = "tosa.transpose"(%arg450, %4113) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %4115 = "tosa.reshape"(%4112) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4116 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %4117 = "linalg.matmul"(%4115, %4114, %4116) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4118 = "tosa.reshape"(%4117) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4119 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4120 = "tosa.transpose"(%arg451, %4119) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %4121 = "tosa.reshape"(%4112) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4122 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %4123 = "linalg.matmul"(%4121, %4120, %4122) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4124 = "tosa.reshape"(%4123) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4125 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4126 = "tosa.transpose"(%arg452, %4125) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %4127 = "tosa.reshape"(%4112) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4128 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %4129 = "linalg.matmul"(%4127, %4126, %4128) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4130 = "tosa.reshape"(%4129) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4131 = "tosa.reshape"(%4118) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %4132 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4133 = "tosa.transpose"(%4131, %4132) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %4134 = "tosa.reshape"(%4124) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %4135 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4136 = "tosa.transpose"(%4134, %4135) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %4137 = "tosa.reshape"(%4130) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %4138 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4139 = "tosa.transpose"(%4137, %4138) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %4140 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x256xf32>}> : () -> tensor<256x256xf32>
    %4141 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4142 = "tosa.transpose"(%4136, %4141) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x20x64x256xf32>
    %4143 = "tosa.reshape"(%4133) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %4144 = "tosa.reshape"(%4142) <{new_shape = array<i64: 20, 64, 256>}> : (tensor<1x20x64x256xf32>) -> tensor<20x64x256xf32>
    %4145 = "tosa.matmul"(%4143, %4144) : (tensor<20x256x64xf32>, tensor<20x64x256xf32>) -> tensor<20x256x256xf32>
    %4146 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x256x256xf32>}> : () -> tensor<20x256x256xf32>
    %4147 = "tosa.mul"(%4145, %4146) <{shift = 0 : i8}> : (tensor<20x256x256xf32>, tensor<20x256x256xf32>) -> tensor<20x256x256xf32>
    %4148 = "tosa.add"(%4147, %4140) : (tensor<20x256x256xf32>, tensor<256x256xf32>) -> tensor<20x256x256xf32>
    %4149 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x256x256xf32>}> : () -> tensor<20x256x256xf32>
    %4150 = "linalg.softmax"(%4148, %4149) <{dimension = 3 : i64}> : (tensor<20x256x256xf32>, tensor<20x256x256xf32>) -> tensor<20x256x256xf32>
    %4151 = "tosa.reshape"(%4139) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %4152 = "tosa.matmul"(%4150, %4151) : (tensor<20x256x256xf32>, tensor<20x256x64xf32>) -> tensor<20x256x64xf32>
    %4153 = "tosa.reshape"(%4152) <{new_shape = array<i64: 1, 20, 256, 64>}> : (tensor<20x256x64xf32>) -> tensor<1x20x256x64xf32>
    %4154 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4155 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4156 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4157 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4158 = "tosa.transpose"(%4153, %4157) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x256x20x64xf32>
    %4159 = "tosa.reshape"(%4158) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x256x20x64xf32>) -> tensor<1x256x1280xf32>
    %4160 = "tosa.reshape"(%4159) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4161 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4162 = "tosa.transpose"(%arg453, %4161) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %4163 = "tosa.reshape"(%4160) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4164 = "tosa.reshape"(%4162) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %4165 = "tosa.matmul"(%4163, %4164) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %4166 = "tosa.reshape"(%4165) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4167 = "tosa.reshape"(%arg454) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %4168 = "tosa.add"(%4167, %4166) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4169 = "tosa.reshape"(%4168) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4170 = "tosa.identity"(%4169) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4171 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x1280xf32>}> : () -> tensor<1x256x1280xf32>
    %4172 = "tosa.reciprocal"(%4171) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4173 = "tosa.mul"(%4170, %4172) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4174 = "tosa.add"(%4173, %4093) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4175 = "tosa.reduce_sum"(%4174) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %4176 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4177 = "tosa.reciprocal"(%4176) : (tensor<1xf32>) -> tensor<1xf32>
    %4178 = "tosa.mul"(%4177, %4175) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4179 = "tosa.sub"(%4174, %4178) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4180 = "tosa.mul"(%4179, %4179) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4181 = "tosa.reduce_sum"(%4180) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %4182 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4183 = "tosa.reciprocal"(%4182) : (tensor<1xf32>) -> tensor<1xf32>
    %4184 = "tosa.mul"(%4183, %4181) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4185 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %4186 = "tosa.add"(%4184, %4185) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4187 = "tosa.rsqrt"(%4186) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4188 = "tosa.sub"(%4174, %4178) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4189 = "tosa.mul"(%4188, %4187) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4190 = "tosa.reshape"(%arg455) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %4191 = "tosa.mul"(%4189, %4190) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %4192 = "tosa.reshape"(%arg456) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %4193 = "tosa.add"(%4191, %4192) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %4194 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4195 = "tosa.transpose"(%arg457, %4194) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %4196 = "tosa.reshape"(%4193) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4197 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x1280xf32>}> : () -> tensor<256x1280xf32>
    %4198 = "linalg.matmul"(%4196, %4195, %4197) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<256x1280xf32>, tensor<1280x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4199 = "tosa.reshape"(%4198) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4200 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4201 = "tosa.transpose"(%arg458, %4200) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %4202 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %4203 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %4204 = "linalg.matmul"(%4202, %4201, %4203) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %4205 = "tosa.reshape"(%4204) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %4206 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4207 = "tosa.transpose"(%arg460, %4206) : (tensor<1280x1024xf32>, tensor<2xi32>) -> tensor<1024x1280xf32>
    %4208 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %4209 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x1280xf32>}> : () -> tensor<12x1280xf32>
    %4210 = "linalg.matmul"(%4208, %4207, %4209) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x1280xf32>, tensor<12x1280xf32>) -> tensor<12x1280xf32>
    %4211 = "tosa.reshape"(%4210) <{new_shape = array<i64: 1, 12, 1280>}> : (tensor<12x1280xf32>) -> tensor<1x12x1280xf32>
    %4212 = "tosa.reshape"(%4199) <{new_shape = array<i64: 1, 256, 20, 64>}> : (tensor<1x256x1280xf32>) -> tensor<1x256x20x64xf32>
    %4213 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4214 = "tosa.transpose"(%4212, %4213) : (tensor<1x256x20x64xf32>, tensor<4xi32>) -> tensor<1x20x256x64xf32>
    %4215 = "tosa.reshape"(%4205) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %4216 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4217 = "tosa.transpose"(%4215, %4216) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %4218 = "tosa.reshape"(%4211) <{new_shape = array<i64: 1, 12, 20, 64>}> : (tensor<1x12x1280xf32>) -> tensor<1x12x20x64xf32>
    %4219 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4220 = "tosa.transpose"(%4218, %4219) : (tensor<1x12x20x64xf32>, tensor<4xi32>) -> tensor<1x20x12x64xf32>
    %4221 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<256x12xf32>}> : () -> tensor<256x12xf32>
    %4222 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4223 = "tosa.transpose"(%4217, %4222) : (tensor<1x20x12x64xf32>, tensor<4xi32>) -> tensor<1x20x64x12xf32>
    %4224 = "tosa.reshape"(%4214) <{new_shape = array<i64: 20, 256, 64>}> : (tensor<1x20x256x64xf32>) -> tensor<20x256x64xf32>
    %4225 = "tosa.reshape"(%4223) <{new_shape = array<i64: 20, 64, 12>}> : (tensor<1x20x64x12xf32>) -> tensor<20x64x12xf32>
    %4226 = "tosa.matmul"(%4224, %4225) : (tensor<20x256x64xf32>, tensor<20x64x12xf32>) -> tensor<20x256x12xf32>
    %4227 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<20x256x12xf32>}> : () -> tensor<20x256x12xf32>
    %4228 = "tosa.mul"(%4226, %4227) <{shift = 0 : i8}> : (tensor<20x256x12xf32>, tensor<20x256x12xf32>) -> tensor<20x256x12xf32>
    %4229 = "tosa.add"(%4228, %4221) : (tensor<20x256x12xf32>, tensor<256x12xf32>) -> tensor<20x256x12xf32>
    %4230 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<20x256x12xf32>}> : () -> tensor<20x256x12xf32>
    %4231 = "linalg.softmax"(%4229, %4230) <{dimension = 3 : i64}> : (tensor<20x256x12xf32>, tensor<20x256x12xf32>) -> tensor<20x256x12xf32>
    %4232 = "tosa.reshape"(%4220) <{new_shape = array<i64: 20, 12, 64>}> : (tensor<1x20x12x64xf32>) -> tensor<20x12x64xf32>
    %4233 = "tosa.matmul"(%4231, %4232) : (tensor<20x256x12xf32>, tensor<20x12x64xf32>) -> tensor<20x256x64xf32>
    %4234 = "tosa.reshape"(%4233) <{new_shape = array<i64: 1, 20, 256, 64>}> : (tensor<20x256x64xf32>) -> tensor<1x20x256x64xf32>
    %4235 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4236 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4237 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4238 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4239 = "tosa.transpose"(%4234, %4238) : (tensor<1x20x256x64xf32>, tensor<4xi32>) -> tensor<1x256x20x64xf32>
    %4240 = "tosa.reshape"(%4239) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<1x256x20x64xf32>) -> tensor<1x256x1280xf32>
    %4241 = "tosa.reshape"(%4240) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4242 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4243 = "tosa.transpose"(%arg462, %4242) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %4244 = "tosa.reshape"(%4241) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4245 = "tosa.reshape"(%4243) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %4246 = "tosa.matmul"(%4244, %4245) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %4247 = "tosa.reshape"(%4246) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4248 = "tosa.reshape"(%arg463) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %4249 = "tosa.add"(%4248, %4247) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4250 = "tosa.reshape"(%4249) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4251 = "tosa.identity"(%4250) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4252 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x1280xf32>}> : () -> tensor<1x256x1280xf32>
    %4253 = "tosa.reciprocal"(%4252) : (tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4254 = "tosa.mul"(%4251, %4253) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4255 = "tosa.add"(%4254, %4174) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4256 = "tosa.reduce_sum"(%4255) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %4257 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4258 = "tosa.reciprocal"(%4257) : (tensor<1xf32>) -> tensor<1xf32>
    %4259 = "tosa.mul"(%4258, %4256) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4260 = "tosa.sub"(%4255, %4259) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4261 = "tosa.mul"(%4260, %4260) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4262 = "tosa.reduce_sum"(%4261) <{axis = 2 : i32}> : (tensor<1x256x1280xf32>) -> tensor<1x256x1xf32>
    %4263 = "tosa.const"() <{value = dense<1.280000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4264 = "tosa.reciprocal"(%4263) : (tensor<1xf32>) -> tensor<1xf32>
    %4265 = "tosa.mul"(%4264, %4262) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4266 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x256x1xf32>}> : () -> tensor<1x256x1xf32>
    %4267 = "tosa.add"(%4265, %4266) : (tensor<1x256x1xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4268 = "tosa.rsqrt"(%4267) : (tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
    %4269 = "tosa.sub"(%4255, %4259) : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4270 = "tosa.mul"(%4269, %4268) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x256x1xf32>) -> tensor<1x256x1280xf32>
    %4271 = "tosa.reshape"(%arg464) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %4272 = "tosa.mul"(%4270, %4271) <{shift = 0 : i8}> : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %4273 = "tosa.reshape"(%arg465) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1x1280xf32>
    %4274 = "tosa.add"(%4272, %4273) : (tensor<1x256x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x256x1280xf32>
    %4275 = "tosa.reshape"(%4274) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4276 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4277 = "tosa.transpose"(%arg466, %4276) : (tensor<10240x1280xf32>, tensor<2xi32>) -> tensor<1280x10240xf32>
    %4278 = "tosa.reshape"(%4275) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4279 = "tosa.reshape"(%4277) <{new_shape = array<i64: 1, 1280, 10240>}> : (tensor<1280x10240xf32>) -> tensor<1x1280x10240xf32>
    %4280 = "tosa.matmul"(%4278, %4279) : (tensor<1x256x1280xf32>, tensor<1x1280x10240xf32>) -> tensor<1x256x10240xf32>
    %4281 = "tosa.reshape"(%4280) <{new_shape = array<i64: 256, 10240>}> : (tensor<1x256x10240xf32>) -> tensor<256x10240xf32>
    %4282 = "tosa.reshape"(%arg467) <{new_shape = array<i64: 1, 10240>}> : (tensor<10240xf32>) -> tensor<1x10240xf32>
    %4283 = "tosa.add"(%4282, %4281) : (tensor<1x10240xf32>, tensor<256x10240xf32>) -> tensor<256x10240xf32>
    %4284 = "tosa.reshape"(%4283) <{new_shape = array<i64: 1, 256, 10240>}> : (tensor<256x10240xf32>) -> tensor<1x256x10240xf32>
    %4285 = "tosa.slice"(%4284) <{size = array<i64: 0, 0, 5120>, start = array<i64: 0, 0, 0>}> : (tensor<1x256x10240xf32>) -> tensor<1x256x5120xf32>
    %4286 = "tosa.slice"(%4284) <{size = array<i64: 0, 0, 10240>, start = array<i64: 0, 0, 5120>}> : (tensor<1x256x10240xf32>) -> tensor<1x256x5120xf32>
    %4287 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %4288 = "tosa.mul"(%4286, %4287) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %4289 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %4290 = "tosa.mul"(%4286, %4289) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %4291 = "math.erf"(%4290) <{fastmath = #arith.fastmath<none>}> : (tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %4292 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x256x5120xf32>}> : () -> tensor<1x256x5120xf32>
    %4293 = "tosa.add"(%4291, %4292) : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %4294 = "tosa.mul"(%4288, %4293) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %4295 = "tosa.mul"(%4285, %4294) <{shift = 0 : i8}> : (tensor<1x256x5120xf32>, tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %4296 = "tosa.identity"(%4295) : (tensor<1x256x5120xf32>) -> tensor<1x256x5120xf32>
    %4297 = "tosa.reshape"(%4296) <{new_shape = array<i64: 256, 5120>}> : (tensor<1x256x5120xf32>) -> tensor<256x5120xf32>
    %4298 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4299 = "tosa.transpose"(%arg468, %4298) : (tensor<1280x5120xf32>, tensor<2xi32>) -> tensor<5120x1280xf32>
    %4300 = "tosa.reshape"(%4297) <{new_shape = array<i64: 1, 256, 5120>}> : (tensor<256x5120xf32>) -> tensor<1x256x5120xf32>
    %4301 = "tosa.reshape"(%4299) <{new_shape = array<i64: 1, 5120, 1280>}> : (tensor<5120x1280xf32>) -> tensor<1x5120x1280xf32>
    %4302 = "tosa.matmul"(%4300, %4301) : (tensor<1x256x5120xf32>, tensor<1x5120x1280xf32>) -> tensor<1x256x1280xf32>
    %4303 = "tosa.reshape"(%4302) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4304 = "tosa.reshape"(%arg469) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %4305 = "tosa.add"(%4304, %4303) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4306 = "tosa.reshape"(%4305) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4307 = "tosa.add"(%4306, %4255) : (tensor<1x256x1280xf32>, tensor<1x256x1280xf32>) -> tensor<1x256x1280xf32>
    %4308 = "tosa.reshape"(%4307) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4309 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4310 = "tosa.transpose"(%arg470, %4309) : (tensor<1280x1280xf32>, tensor<2xi32>) -> tensor<1280x1280xf32>
    %4311 = "tosa.reshape"(%4308) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4312 = "tosa.reshape"(%4310) <{new_shape = array<i64: 1, 1280, 1280>}> : (tensor<1280x1280xf32>) -> tensor<1x1280x1280xf32>
    %4313 = "tosa.matmul"(%4311, %4312) : (tensor<1x256x1280xf32>, tensor<1x1280x1280xf32>) -> tensor<1x256x1280xf32>
    %4314 = "tosa.reshape"(%4313) <{new_shape = array<i64: 256, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<256x1280xf32>
    %4315 = "tosa.reshape"(%arg471) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %4316 = "tosa.add"(%4315, %4314) : (tensor<1x1280xf32>, tensor<256x1280xf32>) -> tensor<256x1280xf32>
    %4317 = "tosa.reshape"(%4316) <{new_shape = array<i64: 1, 256, 1280>}> : (tensor<256x1280xf32>) -> tensor<1x256x1280xf32>
    %4318 = "tosa.reshape"(%4317) <{new_shape = array<i64: 1, 16, 16, 1280>}> : (tensor<1x256x1280xf32>) -> tensor<1x16x16x1280xf32>
    %4319 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4320 = "tosa.transpose"(%4318, %4319) : (tensor<1x16x16x1280xf32>, tensor<4xi32>) -> tensor<1x1280x16x16xf32>
    %4321 = "tosa.identity"(%4320) : (tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %4322 = "tosa.add"(%4321, %4055) : (tensor<1x1280x16x16xf32>, tensor<1x1280x16x16xf32>) -> tensor<1x1280x16x16xf32>
    %4323 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi64>}> : () -> tensor<32xi64>
    %4324 = "tosa.cast"(%4323) : (tensor<32xi64>) -> tensor<32xf32>
    %4325 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<32xf32>}> : () -> tensor<32xf32>
    %4326 = "tosa.mul"(%4324, %4325) <{shift = 0 : i8}> : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %4327 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<32xf32>}> : () -> tensor<32xf32>
    %4328 = "tosa.add"(%4326, %4327) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %4329 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<32xf32>}> : () -> tensor<32xf32>
    %4330 = "tosa.mul"(%4328, %4329) <{shift = 0 : i8}> : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %4331 = "tosa.cast"(%4330) : (tensor<32xf32>) -> tensor<32xi64>
    %4332 = "tosa.reshape"(%4331) <{new_shape = array<i64: 32, 1>}> : (tensor<32xi64>) -> tensor<32x1xi64>
    %4333 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi64>}> : () -> tensor<32xi64>
    %4334 = "tosa.cast"(%4333) : (tensor<32xi64>) -> tensor<32xf32>
    %4335 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<32xf32>}> : () -> tensor<32xf32>
    %4336 = "tosa.mul"(%4334, %4335) <{shift = 0 : i8}> : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %4337 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<32xf32>}> : () -> tensor<32xf32>
    %4338 = "tosa.add"(%4336, %4337) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %4339 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<32xf32>}> : () -> tensor<32xf32>
    %4340 = "tosa.mul"(%4338, %4339) <{shift = 0 : i8}> : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %4341 = "tosa.cast"(%4340) : (tensor<32xf32>) -> tensor<32xi64>
    %4342 = "tensor.empty"() : () -> tensor<1x1280x32x32xf32>
    %4343 = "linalg.generic"(%4332, %4341, %4342) <{indexing_maps = [#map3, #map3, #map4], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: i64, %arg721: i64, %arg722: f32):
      %6629 = "arith.index_cast"(%arg720) : (i64) -> index
      %6630 = "arith.index_cast"(%arg721) : (i64) -> index
      %6631 = "linalg.index"() <{dim = 2 : i64}> : () -> index
      %6632 = "tensor.extract"(%4322, %6629, %6630, %6631) : (tensor<1x1280x16x16xf32>, index, index, index) -> f32
      "linalg.yield"(%6632) : (f32) -> ()
    }) : (tensor<32x1xi64>, tensor<32xi64>, tensor<1x1280x32x32xf32>) -> tensor<1x1280x32x32xf32>
    %4344 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4345 = "tosa.transpose"(%4343, %4344) : (tensor<1x1280x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x1280xf32>
    %4346 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4347 = "tosa.transpose"(%arg472, %4346) : (tensor<1280x1280x3x3xf32>, tensor<4xi32>) -> tensor<1280x3x3x1280xf32>
    %4348 = "tosa.conv2d"(%4345, %4347, %arg473) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x1280xf32>, tensor<1280x3x3x1280xf32>, tensor<1280xf32>) -> tensor<1x32x32x1280xf32>
    %4349 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4350 = "tosa.transpose"(%4348, %4349) : (tensor<1x32x32x1280xf32>, tensor<4xi32>) -> tensor<1x1280x32x32xf32>
    %4351 = "tensor.empty"() : () -> tensor<1x1920x32x32xf32>
    %4352 = "tensor.insert_slice"(%4350, %4351) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1280, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x32x32xf32>, tensor<1x1920x32x32xf32>) -> tensor<1x1920x32x32xf32>
    %4353 = "tensor.insert_slice"(%1507, %4352) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 1280, 0, 0>, static_sizes = array<i64: 1, 640, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x640x32x32xf32>, tensor<1x1920x32x32xf32>) -> tensor<1x1920x32x32xf32>
    %4354 = "tosa.reshape"(%4353) <{new_shape = array<i64: 1, 32, 60, 1024>}> : (tensor<1x1920x32x32xf32>) -> tensor<1x32x60x1024xf32>
    %4355 = "tosa.reduce_sum"(%4354) <{axis = 2 : i32}> : (tensor<1x32x60x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4356 = "tosa.reduce_sum"(%4355) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4357 = "tosa.const"() <{value = dense<6.144000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4358 = "tosa.reciprocal"(%4357) : (tensor<1xf32>) -> tensor<1xf32>
    %4359 = "tosa.mul"(%4358, %4356) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4360 = "tosa.sub"(%4354, %4359) : (tensor<1x32x60x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x60x1024xf32>
    %4361 = "tosa.mul"(%4360, %4360) <{shift = 0 : i8}> : (tensor<1x32x60x1024xf32>, tensor<1x32x60x1024xf32>) -> tensor<1x32x60x1024xf32>
    %4362 = "tosa.reduce_sum"(%4361) <{axis = 2 : i32}> : (tensor<1x32x60x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4363 = "tosa.reduce_sum"(%4362) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4364 = "tosa.const"() <{value = dense<6.144000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4365 = "tosa.reciprocal"(%4364) : (tensor<1xf32>) -> tensor<1xf32>
    %4366 = "tosa.mul"(%4365, %4363) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4367 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %4368 = "tosa.add"(%4366, %4367) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4369 = "tosa.rsqrt"(%4368) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4370 = "tosa.sub"(%4354, %4359) : (tensor<1x32x60x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x60x1024xf32>
    %4371 = "tosa.mul"(%4370, %4369) <{shift = 0 : i8}> : (tensor<1x32x60x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x60x1024xf32>
    %4372 = "tosa.reshape"(%4371) <{new_shape = array<i64: 1, 1920, 32, 32>}> : (tensor<1x32x60x1024xf32>) -> tensor<1x1920x32x32xf32>
    %4373 = "tosa.reshape"(%arg474) <{new_shape = array<i64: 1, 1920>}> : (tensor<1920xf32>) -> tensor<1x1920xf32>
    %4374 = "tosa.reshape"(%4373) <{new_shape = array<i64: 1, 1920, 1>}> : (tensor<1x1920xf32>) -> tensor<1x1920x1xf32>
    %4375 = "tosa.reshape"(%4374) <{new_shape = array<i64: 1, 1920, 1, 1>}> : (tensor<1x1920x1xf32>) -> tensor<1x1920x1x1xf32>
    %4376 = "tosa.reshape"(%arg475) <{new_shape = array<i64: 1, 1920>}> : (tensor<1920xf32>) -> tensor<1x1920xf32>
    %4377 = "tosa.reshape"(%4376) <{new_shape = array<i64: 1, 1920, 1>}> : (tensor<1x1920xf32>) -> tensor<1x1920x1xf32>
    %4378 = "tosa.reshape"(%4377) <{new_shape = array<i64: 1, 1920, 1, 1>}> : (tensor<1x1920x1xf32>) -> tensor<1x1920x1x1xf32>
    %4379 = "tosa.mul"(%4372, %4378) <{shift = 0 : i8}> : (tensor<1x1920x32x32xf32>, tensor<1x1920x1x1xf32>) -> tensor<1x1920x32x32xf32>
    %4380 = "tosa.add"(%4379, %4375) : (tensor<1x1920x32x32xf32>, tensor<1x1920x1x1xf32>) -> tensor<1x1920x32x32xf32>
    %4381 = "tosa.sigmoid"(%4380) : (tensor<1x1920x32x32xf32>) -> tensor<1x1920x32x32xf32>
    %4382 = "tosa.mul"(%4380, %4381) <{shift = 0 : i8}> : (tensor<1x1920x32x32xf32>, tensor<1x1920x32x32xf32>) -> tensor<1x1920x32x32xf32>
    %4383 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4384 = "tosa.transpose"(%4382, %4383) : (tensor<1x1920x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x1920xf32>
    %4385 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4386 = "tosa.transpose"(%arg476, %4385) : (tensor<640x1920x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x1920xf32>
    %4387 = "tosa.conv2d"(%4384, %4386, %arg477) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x1920xf32>, tensor<640x3x3x1920xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %4388 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4389 = "tosa.transpose"(%4387, %4388) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %4390 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %4391 = "tosa.mul"(%50, %4390) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %4392 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4393 = "tosa.transpose"(%arg478, %4392) : (tensor<640x1280xf32>, tensor<2xi32>) -> tensor<1280x640xf32>
    %4394 = "tosa.reshape"(%4391) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %4395 = "tosa.reshape"(%4393) <{new_shape = array<i64: 1, 1280, 640>}> : (tensor<1280x640xf32>) -> tensor<1x1280x640xf32>
    %4396 = "tosa.matmul"(%4394, %4395) : (tensor<1x1x1280xf32>, tensor<1x1280x640xf32>) -> tensor<1x1x640xf32>
    %4397 = "tosa.reshape"(%4396) <{new_shape = array<i64: 1, 640>}> : (tensor<1x1x640xf32>) -> tensor<1x640xf32>
    %4398 = "tosa.reshape"(%arg479) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4399 = "tosa.add"(%4398, %4397) : (tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4400 = "tensor.extract_slice"(%4399) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 640>, static_strides = array<i64: 1, 1>}> : (tensor<1x640xf32>) -> tensor<1x640xf32>
    %4401 = "tensor.extract_slice"(%4400) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 640>, static_strides = array<i64: 1, 1>}> : (tensor<1x640xf32>) -> tensor<1x640xf32>
    %4402 = "tosa.reshape"(%4401) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %4403 = "tosa.reshape"(%4402) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %4404 = "tosa.add"(%4389, %4403) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %4405 = "tosa.reshape"(%4404) <{new_shape = array<i64: 1, 32, 20, 1024>}> : (tensor<1x640x32x32xf32>) -> tensor<1x32x20x1024xf32>
    %4406 = "tosa.reduce_sum"(%4405) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4407 = "tosa.reduce_sum"(%4406) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4408 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4409 = "tosa.reciprocal"(%4408) : (tensor<1xf32>) -> tensor<1xf32>
    %4410 = "tosa.mul"(%4409, %4407) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4411 = "tosa.sub"(%4405, %4410) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4412 = "tosa.mul"(%4411, %4411) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x20x1024xf32>) -> tensor<1x32x20x1024xf32>
    %4413 = "tosa.reduce_sum"(%4412) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4414 = "tosa.reduce_sum"(%4413) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4415 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4416 = "tosa.reciprocal"(%4415) : (tensor<1xf32>) -> tensor<1xf32>
    %4417 = "tosa.mul"(%4416, %4414) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4418 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %4419 = "tosa.add"(%4417, %4418) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4420 = "tosa.rsqrt"(%4419) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4421 = "tosa.sub"(%4405, %4410) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4422 = "tosa.mul"(%4421, %4420) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4423 = "tosa.reshape"(%4422) <{new_shape = array<i64: 1, 640, 32, 32>}> : (tensor<1x32x20x1024xf32>) -> tensor<1x640x32x32xf32>
    %4424 = "tosa.reshape"(%arg480) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4425 = "tosa.reshape"(%4424) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %4426 = "tosa.reshape"(%4425) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %4427 = "tosa.reshape"(%arg481) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4428 = "tosa.reshape"(%4427) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %4429 = "tosa.reshape"(%4428) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %4430 = "tosa.mul"(%4423, %4429) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %4431 = "tosa.add"(%4430, %4426) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %4432 = "tosa.sigmoid"(%4431) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4433 = "tosa.mul"(%4431, %4432) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4434 = "tosa.identity"(%4433) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4435 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4436 = "tosa.transpose"(%4434, %4435) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %4437 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4438 = "tosa.transpose"(%arg482, %4437) : (tensor<640x640x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x640xf32>
    %4439 = "tosa.conv2d"(%4436, %4438, %arg483) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x640xf32>, tensor<640x3x3x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %4440 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4441 = "tosa.transpose"(%4439, %4440) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %4442 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4443 = "tosa.transpose"(%4353, %4442) : (tensor<1x1920x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x1920xf32>
    %4444 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4445 = "tosa.transpose"(%arg484, %4444) : (tensor<640x1920x1x1xf32>, tensor<4xi32>) -> tensor<640x1x1x1920xf32>
    %4446 = "tosa.conv2d"(%4443, %4445, %arg485) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x1920xf32>, tensor<640x1x1x1920xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %4447 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4448 = "tosa.transpose"(%4446, %4447) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %4449 = "tosa.add"(%4448, %4441) : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4450 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x640x32x32xf32>}> : () -> tensor<1x640x32x32xf32>
    %4451 = "tosa.reciprocal"(%4450) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4452 = "tosa.mul"(%4449, %4451) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4453 = "tosa.reshape"(%4452) <{new_shape = array<i64: 1, 32, 20, 1024>}> : (tensor<1x640x32x32xf32>) -> tensor<1x32x20x1024xf32>
    %4454 = "tosa.reduce_sum"(%4453) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4455 = "tosa.reduce_sum"(%4454) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4456 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4457 = "tosa.reciprocal"(%4456) : (tensor<1xf32>) -> tensor<1xf32>
    %4458 = "tosa.mul"(%4457, %4455) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4459 = "tosa.sub"(%4453, %4458) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4460 = "tosa.mul"(%4459, %4459) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x20x1024xf32>) -> tensor<1x32x20x1024xf32>
    %4461 = "tosa.reduce_sum"(%4460) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4462 = "tosa.reduce_sum"(%4461) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4463 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4464 = "tosa.reciprocal"(%4463) : (tensor<1xf32>) -> tensor<1xf32>
    %4465 = "tosa.mul"(%4464, %4462) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4466 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %4467 = "tosa.add"(%4465, %4466) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4468 = "tosa.rsqrt"(%4467) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4469 = "tosa.sub"(%4453, %4458) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4470 = "tosa.mul"(%4469, %4468) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4471 = "tosa.reshape"(%4470) <{new_shape = array<i64: 1, 640, 32, 32>}> : (tensor<1x32x20x1024xf32>) -> tensor<1x640x32x32xf32>
    %4472 = "tosa.reshape"(%arg486) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4473 = "tosa.reshape"(%4472) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %4474 = "tosa.reshape"(%4473) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %4475 = "tosa.reshape"(%arg487) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4476 = "tosa.reshape"(%4475) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %4477 = "tosa.reshape"(%4476) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %4478 = "tosa.mul"(%4471, %4477) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %4479 = "tosa.add"(%4478, %4474) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %4480 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4481 = "tosa.transpose"(%4479, %4480) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %4482 = "tosa.reshape"(%4481) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x32x32x640xf32>) -> tensor<1x1024x640xf32>
    %4483 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4484 = "tosa.transpose"(%arg488, %4483) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4485 = "tosa.reshape"(%4482) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4486 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %4487 = "linalg.matmul"(%4485, %4484, %4486) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4488 = "tosa.reshape"(%4487) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4489 = "tosa.reshape"(%arg489) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4490 = "tosa.add"(%4488, %4489) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4491 = "tosa.reduce_sum"(%4490) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %4492 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4493 = "tosa.reciprocal"(%4492) : (tensor<1xf32>) -> tensor<1xf32>
    %4494 = "tosa.mul"(%4493, %4491) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4495 = "tosa.sub"(%4490, %4494) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4496 = "tosa.mul"(%4495, %4495) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4497 = "tosa.reduce_sum"(%4496) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %4498 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4499 = "tosa.reciprocal"(%4498) : (tensor<1xf32>) -> tensor<1xf32>
    %4500 = "tosa.mul"(%4499, %4497) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4501 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %4502 = "tosa.add"(%4500, %4501) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4503 = "tosa.rsqrt"(%4502) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4504 = "tosa.sub"(%4490, %4494) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4505 = "tosa.mul"(%4504, %4503) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4506 = "tosa.reshape"(%arg490) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4507 = "tosa.mul"(%4505, %4506) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4508 = "tosa.reshape"(%arg491) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4509 = "tosa.add"(%4507, %4508) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4510 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4511 = "tosa.transpose"(%arg492, %4510) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4512 = "tosa.reshape"(%4509) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4513 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %4514 = "linalg.matmul"(%4512, %4511, %4513) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4515 = "tosa.reshape"(%4514) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4516 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4517 = "tosa.transpose"(%arg493, %4516) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4518 = "tosa.reshape"(%4509) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4519 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %4520 = "linalg.matmul"(%4518, %4517, %4519) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4521 = "tosa.reshape"(%4520) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4522 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4523 = "tosa.transpose"(%arg494, %4522) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4524 = "tosa.reshape"(%4509) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4525 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %4526 = "linalg.matmul"(%4524, %4523, %4525) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4527 = "tosa.reshape"(%4526) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4528 = "tosa.reshape"(%4515) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %4529 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4530 = "tosa.transpose"(%4528, %4529) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %4531 = "tosa.reshape"(%4521) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %4532 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4533 = "tosa.transpose"(%4531, %4532) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %4534 = "tosa.reshape"(%4527) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %4535 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4536 = "tosa.transpose"(%4534, %4535) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %4537 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x1024xf32>}> : () -> tensor<1024x1024xf32>
    %4538 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4539 = "tosa.transpose"(%4533, %4538) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x10x64x1024xf32>
    %4540 = "tosa.reshape"(%4530) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %4541 = "tosa.reshape"(%4539) <{new_shape = array<i64: 10, 64, 1024>}> : (tensor<1x10x64x1024xf32>) -> tensor<10x64x1024xf32>
    %4542 = "tosa.matmul"(%4540, %4541) : (tensor<10x1024x64xf32>, tensor<10x64x1024xf32>) -> tensor<10x1024x1024xf32>
    %4543 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<10x1024x1024xf32>}> : () -> tensor<10x1024x1024xf32>
    %4544 = "tosa.mul"(%4542, %4543) <{shift = 0 : i8}> : (tensor<10x1024x1024xf32>, tensor<10x1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %4545 = "tosa.add"(%4544, %4537) : (tensor<10x1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %4546 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<10x1024x1024xf32>}> : () -> tensor<10x1024x1024xf32>
    %4547 = "linalg.softmax"(%4545, %4546) <{dimension = 3 : i64}> : (tensor<10x1024x1024xf32>, tensor<10x1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %4548 = "tosa.reshape"(%4536) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %4549 = "tosa.matmul"(%4547, %4548) : (tensor<10x1024x1024xf32>, tensor<10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %4550 = "tosa.reshape"(%4549) <{new_shape = array<i64: 1, 10, 1024, 64>}> : (tensor<10x1024x64xf32>) -> tensor<1x10x1024x64xf32>
    %4551 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4552 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4553 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4554 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4555 = "tosa.transpose"(%4550, %4554) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x1024x10x64xf32>
    %4556 = "tosa.reshape"(%4555) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x1024x10x64xf32>) -> tensor<1x1024x640xf32>
    %4557 = "tosa.reshape"(%4556) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4558 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4559 = "tosa.transpose"(%arg495, %4558) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4560 = "tosa.reshape"(%4557) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4561 = "tosa.reshape"(%4559) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %4562 = "tosa.matmul"(%4560, %4561) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %4563 = "tosa.reshape"(%4562) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4564 = "tosa.reshape"(%arg496) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4565 = "tosa.add"(%4564, %4563) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4566 = "tosa.reshape"(%4565) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4567 = "tosa.identity"(%4566) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4568 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x640xf32>}> : () -> tensor<1x1024x640xf32>
    %4569 = "tosa.reciprocal"(%4568) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4570 = "tosa.mul"(%4567, %4569) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4571 = "tosa.add"(%4570, %4490) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4572 = "tosa.reduce_sum"(%4571) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %4573 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4574 = "tosa.reciprocal"(%4573) : (tensor<1xf32>) -> tensor<1xf32>
    %4575 = "tosa.mul"(%4574, %4572) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4576 = "tosa.sub"(%4571, %4575) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4577 = "tosa.mul"(%4576, %4576) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4578 = "tosa.reduce_sum"(%4577) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %4579 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4580 = "tosa.reciprocal"(%4579) : (tensor<1xf32>) -> tensor<1xf32>
    %4581 = "tosa.mul"(%4580, %4578) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4582 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %4583 = "tosa.add"(%4581, %4582) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4584 = "tosa.rsqrt"(%4583) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4585 = "tosa.sub"(%4571, %4575) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4586 = "tosa.mul"(%4585, %4584) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4587 = "tosa.reshape"(%arg497) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4588 = "tosa.mul"(%4586, %4587) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4589 = "tosa.reshape"(%arg498) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4590 = "tosa.add"(%4588, %4589) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4591 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4592 = "tosa.transpose"(%arg499, %4591) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4593 = "tosa.reshape"(%4590) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4594 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %4595 = "linalg.matmul"(%4593, %4592, %4594) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4596 = "tosa.reshape"(%4595) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4597 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4598 = "tosa.transpose"(%arg500, %4597) : (tensor<640x1024xf32>, tensor<2xi32>) -> tensor<1024x640xf32>
    %4599 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %4600 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x640xf32>}> : () -> tensor<12x640xf32>
    %4601 = "linalg.matmul"(%4599, %4598, %4600) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x640xf32>, tensor<12x640xf32>) -> tensor<12x640xf32>
    %4602 = "tosa.reshape"(%4601) <{new_shape = array<i64: 1, 12, 640>}> : (tensor<12x640xf32>) -> tensor<1x12x640xf32>
    %4603 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4604 = "tosa.transpose"(%arg502, %4603) : (tensor<640x1024xf32>, tensor<2xi32>) -> tensor<1024x640xf32>
    %4605 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %4606 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x640xf32>}> : () -> tensor<12x640xf32>
    %4607 = "linalg.matmul"(%4605, %4604, %4606) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x640xf32>, tensor<12x640xf32>) -> tensor<12x640xf32>
    %4608 = "tosa.reshape"(%4607) <{new_shape = array<i64: 1, 12, 640>}> : (tensor<12x640xf32>) -> tensor<1x12x640xf32>
    %4609 = "tosa.reshape"(%4596) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %4610 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4611 = "tosa.transpose"(%4609, %4610) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %4612 = "tosa.reshape"(%4602) <{new_shape = array<i64: 1, 12, 10, 64>}> : (tensor<1x12x640xf32>) -> tensor<1x12x10x64xf32>
    %4613 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4614 = "tosa.transpose"(%4612, %4613) : (tensor<1x12x10x64xf32>, tensor<4xi32>) -> tensor<1x10x12x64xf32>
    %4615 = "tosa.reshape"(%4608) <{new_shape = array<i64: 1, 12, 10, 64>}> : (tensor<1x12x640xf32>) -> tensor<1x12x10x64xf32>
    %4616 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4617 = "tosa.transpose"(%4615, %4616) : (tensor<1x12x10x64xf32>, tensor<4xi32>) -> tensor<1x10x12x64xf32>
    %4618 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x12xf32>}> : () -> tensor<1024x12xf32>
    %4619 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4620 = "tosa.transpose"(%4614, %4619) : (tensor<1x10x12x64xf32>, tensor<4xi32>) -> tensor<1x10x64x12xf32>
    %4621 = "tosa.reshape"(%4611) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %4622 = "tosa.reshape"(%4620) <{new_shape = array<i64: 10, 64, 12>}> : (tensor<1x10x64x12xf32>) -> tensor<10x64x12xf32>
    %4623 = "tosa.matmul"(%4621, %4622) : (tensor<10x1024x64xf32>, tensor<10x64x12xf32>) -> tensor<10x1024x12xf32>
    %4624 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<10x1024x12xf32>}> : () -> tensor<10x1024x12xf32>
    %4625 = "tosa.mul"(%4623, %4624) <{shift = 0 : i8}> : (tensor<10x1024x12xf32>, tensor<10x1024x12xf32>) -> tensor<10x1024x12xf32>
    %4626 = "tosa.add"(%4625, %4618) : (tensor<10x1024x12xf32>, tensor<1024x12xf32>) -> tensor<10x1024x12xf32>
    %4627 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<10x1024x12xf32>}> : () -> tensor<10x1024x12xf32>
    %4628 = "linalg.softmax"(%4626, %4627) <{dimension = 3 : i64}> : (tensor<10x1024x12xf32>, tensor<10x1024x12xf32>) -> tensor<10x1024x12xf32>
    %4629 = "tosa.reshape"(%4617) <{new_shape = array<i64: 10, 12, 64>}> : (tensor<1x10x12x64xf32>) -> tensor<10x12x64xf32>
    %4630 = "tosa.matmul"(%4628, %4629) : (tensor<10x1024x12xf32>, tensor<10x12x64xf32>) -> tensor<10x1024x64xf32>
    %4631 = "tosa.reshape"(%4630) <{new_shape = array<i64: 1, 10, 1024, 64>}> : (tensor<10x1024x64xf32>) -> tensor<1x10x1024x64xf32>
    %4632 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4633 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4634 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4635 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4636 = "tosa.transpose"(%4631, %4635) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x1024x10x64xf32>
    %4637 = "tosa.reshape"(%4636) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x1024x10x64xf32>) -> tensor<1x1024x640xf32>
    %4638 = "tosa.reshape"(%4637) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4639 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4640 = "tosa.transpose"(%arg504, %4639) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4641 = "tosa.reshape"(%4638) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4642 = "tosa.reshape"(%4640) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %4643 = "tosa.matmul"(%4641, %4642) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %4644 = "tosa.reshape"(%4643) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4645 = "tosa.reshape"(%arg505) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4646 = "tosa.add"(%4645, %4644) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4647 = "tosa.reshape"(%4646) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4648 = "tosa.identity"(%4647) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4649 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x640xf32>}> : () -> tensor<1x1024x640xf32>
    %4650 = "tosa.reciprocal"(%4649) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4651 = "tosa.mul"(%4648, %4650) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4652 = "tosa.add"(%4651, %4571) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4653 = "tosa.reduce_sum"(%4652) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %4654 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4655 = "tosa.reciprocal"(%4654) : (tensor<1xf32>) -> tensor<1xf32>
    %4656 = "tosa.mul"(%4655, %4653) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4657 = "tosa.sub"(%4652, %4656) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4658 = "tosa.mul"(%4657, %4657) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4659 = "tosa.reduce_sum"(%4658) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %4660 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4661 = "tosa.reciprocal"(%4660) : (tensor<1xf32>) -> tensor<1xf32>
    %4662 = "tosa.mul"(%4661, %4659) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4663 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %4664 = "tosa.add"(%4662, %4663) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4665 = "tosa.rsqrt"(%4664) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4666 = "tosa.sub"(%4652, %4656) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4667 = "tosa.mul"(%4666, %4665) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4668 = "tosa.reshape"(%arg506) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4669 = "tosa.mul"(%4667, %4668) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4670 = "tosa.reshape"(%arg507) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4671 = "tosa.add"(%4669, %4670) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4672 = "tosa.reshape"(%4671) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4673 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4674 = "tosa.transpose"(%arg508, %4673) : (tensor<5120x640xf32>, tensor<2xi32>) -> tensor<640x5120xf32>
    %4675 = "tosa.reshape"(%4672) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4676 = "tosa.reshape"(%4674) <{new_shape = array<i64: 1, 640, 5120>}> : (tensor<640x5120xf32>) -> tensor<1x640x5120xf32>
    %4677 = "tosa.matmul"(%4675, %4676) : (tensor<1x1024x640xf32>, tensor<1x640x5120xf32>) -> tensor<1x1024x5120xf32>
    %4678 = "tosa.reshape"(%4677) <{new_shape = array<i64: 1024, 5120>}> : (tensor<1x1024x5120xf32>) -> tensor<1024x5120xf32>
    %4679 = "tosa.reshape"(%arg509) <{new_shape = array<i64: 1, 5120>}> : (tensor<5120xf32>) -> tensor<1x5120xf32>
    %4680 = "tosa.add"(%4679, %4678) : (tensor<1x5120xf32>, tensor<1024x5120xf32>) -> tensor<1024x5120xf32>
    %4681 = "tosa.reshape"(%4680) <{new_shape = array<i64: 1, 1024, 5120>}> : (tensor<1024x5120xf32>) -> tensor<1x1024x5120xf32>
    %4682 = "tosa.slice"(%4681) <{size = array<i64: 0, 0, 2560>, start = array<i64: 0, 0, 0>}> : (tensor<1x1024x5120xf32>) -> tensor<1x1024x2560xf32>
    %4683 = "tosa.slice"(%4681) <{size = array<i64: 0, 0, 5120>, start = array<i64: 0, 0, 2560>}> : (tensor<1x1024x5120xf32>) -> tensor<1x1024x2560xf32>
    %4684 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %4685 = "tosa.mul"(%4683, %4684) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %4686 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %4687 = "tosa.mul"(%4683, %4686) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %4688 = "math.erf"(%4687) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %4689 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %4690 = "tosa.add"(%4688, %4689) : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %4691 = "tosa.mul"(%4685, %4690) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %4692 = "tosa.mul"(%4682, %4691) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %4693 = "tosa.identity"(%4692) : (tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %4694 = "tosa.reshape"(%4693) <{new_shape = array<i64: 1024, 2560>}> : (tensor<1x1024x2560xf32>) -> tensor<1024x2560xf32>
    %4695 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4696 = "tosa.transpose"(%arg510, %4695) : (tensor<640x2560xf32>, tensor<2xi32>) -> tensor<2560x640xf32>
    %4697 = "tosa.reshape"(%4694) <{new_shape = array<i64: 1, 1024, 2560>}> : (tensor<1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %4698 = "tosa.reshape"(%4696) <{new_shape = array<i64: 1, 2560, 640>}> : (tensor<2560x640xf32>) -> tensor<1x2560x640xf32>
    %4699 = "tosa.matmul"(%4697, %4698) : (tensor<1x1024x2560xf32>, tensor<1x2560x640xf32>) -> tensor<1x1024x640xf32>
    %4700 = "tosa.reshape"(%4699) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4701 = "tosa.reshape"(%arg511) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4702 = "tosa.add"(%4701, %4700) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4703 = "tosa.reshape"(%4702) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4704 = "tosa.add"(%4703, %4652) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4705 = "tosa.reshape"(%4704) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4706 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4707 = "tosa.transpose"(%arg512, %4706) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4708 = "tosa.reshape"(%4705) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4709 = "tosa.reshape"(%4707) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %4710 = "tosa.matmul"(%4708, %4709) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %4711 = "tosa.reshape"(%4710) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4712 = "tosa.reshape"(%arg513) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4713 = "tosa.add"(%4712, %4711) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4714 = "tosa.reshape"(%4713) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4715 = "tosa.reshape"(%4714) <{new_shape = array<i64: 1, 32, 32, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1x32x32x640xf32>
    %4716 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4717 = "tosa.transpose"(%4715, %4716) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %4718 = "tosa.identity"(%4717) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4719 = "tosa.add"(%4718, %4452) : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4720 = "tensor.empty"() : () -> tensor<1x1280x32x32xf32>
    %4721 = "tensor.insert_slice"(%4719, %4720) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 640, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x640x32x32xf32>, tensor<1x1280x32x32xf32>) -> tensor<1x1280x32x32xf32>
    %4722 = "tensor.insert_slice"(%1148, %4721) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 640, 0, 0>, static_sizes = array<i64: 1, 640, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x640x32x32xf32>, tensor<1x1280x32x32xf32>) -> tensor<1x1280x32x32xf32>
    %4723 = "tosa.reshape"(%4722) <{new_shape = array<i64: 1, 32, 40, 1024>}> : (tensor<1x1280x32x32xf32>) -> tensor<1x32x40x1024xf32>
    %4724 = "tosa.reduce_sum"(%4723) <{axis = 2 : i32}> : (tensor<1x32x40x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4725 = "tosa.reduce_sum"(%4724) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4726 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4727 = "tosa.reciprocal"(%4726) : (tensor<1xf32>) -> tensor<1xf32>
    %4728 = "tosa.mul"(%4727, %4725) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4729 = "tosa.sub"(%4723, %4728) : (tensor<1x32x40x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x1024xf32>
    %4730 = "tosa.mul"(%4729, %4729) <{shift = 0 : i8}> : (tensor<1x32x40x1024xf32>, tensor<1x32x40x1024xf32>) -> tensor<1x32x40x1024xf32>
    %4731 = "tosa.reduce_sum"(%4730) <{axis = 2 : i32}> : (tensor<1x32x40x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4732 = "tosa.reduce_sum"(%4731) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4733 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4734 = "tosa.reciprocal"(%4733) : (tensor<1xf32>) -> tensor<1xf32>
    %4735 = "tosa.mul"(%4734, %4732) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4736 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %4737 = "tosa.add"(%4735, %4736) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4738 = "tosa.rsqrt"(%4737) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4739 = "tosa.sub"(%4723, %4728) : (tensor<1x32x40x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x1024xf32>
    %4740 = "tosa.mul"(%4739, %4738) <{shift = 0 : i8}> : (tensor<1x32x40x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x40x1024xf32>
    %4741 = "tosa.reshape"(%4740) <{new_shape = array<i64: 1, 1280, 32, 32>}> : (tensor<1x32x40x1024xf32>) -> tensor<1x1280x32x32xf32>
    %4742 = "tosa.reshape"(%arg514) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %4743 = "tosa.reshape"(%4742) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %4744 = "tosa.reshape"(%4743) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %4745 = "tosa.reshape"(%arg515) <{new_shape = array<i64: 1, 1280>}> : (tensor<1280xf32>) -> tensor<1x1280xf32>
    %4746 = "tosa.reshape"(%4745) <{new_shape = array<i64: 1, 1280, 1>}> : (tensor<1x1280xf32>) -> tensor<1x1280x1xf32>
    %4747 = "tosa.reshape"(%4746) <{new_shape = array<i64: 1, 1280, 1, 1>}> : (tensor<1x1280x1xf32>) -> tensor<1x1280x1x1xf32>
    %4748 = "tosa.mul"(%4741, %4747) <{shift = 0 : i8}> : (tensor<1x1280x32x32xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x32x32xf32>
    %4749 = "tosa.add"(%4748, %4744) : (tensor<1x1280x32x32xf32>, tensor<1x1280x1x1xf32>) -> tensor<1x1280x32x32xf32>
    %4750 = "tosa.sigmoid"(%4749) : (tensor<1x1280x32x32xf32>) -> tensor<1x1280x32x32xf32>
    %4751 = "tosa.mul"(%4749, %4750) <{shift = 0 : i8}> : (tensor<1x1280x32x32xf32>, tensor<1x1280x32x32xf32>) -> tensor<1x1280x32x32xf32>
    %4752 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4753 = "tosa.transpose"(%4751, %4752) : (tensor<1x1280x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x1280xf32>
    %4754 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4755 = "tosa.transpose"(%arg516, %4754) : (tensor<640x1280x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x1280xf32>
    %4756 = "tosa.conv2d"(%4753, %4755, %arg517) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x1280xf32>, tensor<640x3x3x1280xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %4757 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4758 = "tosa.transpose"(%4756, %4757) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %4759 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %4760 = "tosa.mul"(%50, %4759) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %4761 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4762 = "tosa.transpose"(%arg518, %4761) : (tensor<640x1280xf32>, tensor<2xi32>) -> tensor<1280x640xf32>
    %4763 = "tosa.reshape"(%4760) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %4764 = "tosa.reshape"(%4762) <{new_shape = array<i64: 1, 1280, 640>}> : (tensor<1280x640xf32>) -> tensor<1x1280x640xf32>
    %4765 = "tosa.matmul"(%4763, %4764) : (tensor<1x1x1280xf32>, tensor<1x1280x640xf32>) -> tensor<1x1x640xf32>
    %4766 = "tosa.reshape"(%4765) <{new_shape = array<i64: 1, 640>}> : (tensor<1x1x640xf32>) -> tensor<1x640xf32>
    %4767 = "tosa.reshape"(%arg519) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4768 = "tosa.add"(%4767, %4766) : (tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %4769 = "tensor.extract_slice"(%4768) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 640>, static_strides = array<i64: 1, 1>}> : (tensor<1x640xf32>) -> tensor<1x640xf32>
    %4770 = "tensor.extract_slice"(%4769) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 640>, static_strides = array<i64: 1, 1>}> : (tensor<1x640xf32>) -> tensor<1x640xf32>
    %4771 = "tosa.reshape"(%4770) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %4772 = "tosa.reshape"(%4771) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %4773 = "tosa.add"(%4758, %4772) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %4774 = "tosa.reshape"(%4773) <{new_shape = array<i64: 1, 32, 20, 1024>}> : (tensor<1x640x32x32xf32>) -> tensor<1x32x20x1024xf32>
    %4775 = "tosa.reduce_sum"(%4774) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4776 = "tosa.reduce_sum"(%4775) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4777 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4778 = "tosa.reciprocal"(%4777) : (tensor<1xf32>) -> tensor<1xf32>
    %4779 = "tosa.mul"(%4778, %4776) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4780 = "tosa.sub"(%4774, %4779) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4781 = "tosa.mul"(%4780, %4780) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x20x1024xf32>) -> tensor<1x32x20x1024xf32>
    %4782 = "tosa.reduce_sum"(%4781) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4783 = "tosa.reduce_sum"(%4782) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4784 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4785 = "tosa.reciprocal"(%4784) : (tensor<1xf32>) -> tensor<1xf32>
    %4786 = "tosa.mul"(%4785, %4783) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4787 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %4788 = "tosa.add"(%4786, %4787) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4789 = "tosa.rsqrt"(%4788) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4790 = "tosa.sub"(%4774, %4779) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4791 = "tosa.mul"(%4790, %4789) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4792 = "tosa.reshape"(%4791) <{new_shape = array<i64: 1, 640, 32, 32>}> : (tensor<1x32x20x1024xf32>) -> tensor<1x640x32x32xf32>
    %4793 = "tosa.reshape"(%arg520) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4794 = "tosa.reshape"(%4793) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %4795 = "tosa.reshape"(%4794) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %4796 = "tosa.reshape"(%arg521) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4797 = "tosa.reshape"(%4796) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %4798 = "tosa.reshape"(%4797) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %4799 = "tosa.mul"(%4792, %4798) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %4800 = "tosa.add"(%4799, %4795) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %4801 = "tosa.sigmoid"(%4800) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4802 = "tosa.mul"(%4800, %4801) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4803 = "tosa.identity"(%4802) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4804 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4805 = "tosa.transpose"(%4803, %4804) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %4806 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4807 = "tosa.transpose"(%arg522, %4806) : (tensor<640x640x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x640xf32>
    %4808 = "tosa.conv2d"(%4805, %4807, %arg523) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x640xf32>, tensor<640x3x3x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %4809 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4810 = "tosa.transpose"(%4808, %4809) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %4811 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4812 = "tosa.transpose"(%4722, %4811) : (tensor<1x1280x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x1280xf32>
    %4813 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4814 = "tosa.transpose"(%arg524, %4813) : (tensor<640x1280x1x1xf32>, tensor<4xi32>) -> tensor<640x1x1x1280xf32>
    %4815 = "tosa.conv2d"(%4812, %4814, %arg525) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x1280xf32>, tensor<640x1x1x1280xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %4816 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4817 = "tosa.transpose"(%4815, %4816) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %4818 = "tosa.add"(%4817, %4810) : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4819 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x640x32x32xf32>}> : () -> tensor<1x640x32x32xf32>
    %4820 = "tosa.reciprocal"(%4819) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4821 = "tosa.mul"(%4818, %4820) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %4822 = "tosa.reshape"(%4821) <{new_shape = array<i64: 1, 32, 20, 1024>}> : (tensor<1x640x32x32xf32>) -> tensor<1x32x20x1024xf32>
    %4823 = "tosa.reduce_sum"(%4822) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4824 = "tosa.reduce_sum"(%4823) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4825 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4826 = "tosa.reciprocal"(%4825) : (tensor<1xf32>) -> tensor<1xf32>
    %4827 = "tosa.mul"(%4826, %4824) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4828 = "tosa.sub"(%4822, %4827) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4829 = "tosa.mul"(%4828, %4828) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x20x1024xf32>) -> tensor<1x32x20x1024xf32>
    %4830 = "tosa.reduce_sum"(%4829) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %4831 = "tosa.reduce_sum"(%4830) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %4832 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4833 = "tosa.reciprocal"(%4832) : (tensor<1xf32>) -> tensor<1xf32>
    %4834 = "tosa.mul"(%4833, %4831) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4835 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %4836 = "tosa.add"(%4834, %4835) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4837 = "tosa.rsqrt"(%4836) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %4838 = "tosa.sub"(%4822, %4827) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4839 = "tosa.mul"(%4838, %4837) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %4840 = "tosa.reshape"(%4839) <{new_shape = array<i64: 1, 640, 32, 32>}> : (tensor<1x32x20x1024xf32>) -> tensor<1x640x32x32xf32>
    %4841 = "tosa.reshape"(%arg526) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4842 = "tosa.reshape"(%4841) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %4843 = "tosa.reshape"(%4842) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %4844 = "tosa.reshape"(%arg527) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4845 = "tosa.reshape"(%4844) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %4846 = "tosa.reshape"(%4845) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %4847 = "tosa.mul"(%4840, %4846) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %4848 = "tosa.add"(%4847, %4843) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %4849 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4850 = "tosa.transpose"(%4848, %4849) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %4851 = "tosa.reshape"(%4850) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x32x32x640xf32>) -> tensor<1x1024x640xf32>
    %4852 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4853 = "tosa.transpose"(%arg528, %4852) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4854 = "tosa.reshape"(%4851) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4855 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %4856 = "linalg.matmul"(%4854, %4853, %4855) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4857 = "tosa.reshape"(%4856) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4858 = "tosa.reshape"(%arg529) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4859 = "tosa.add"(%4857, %4858) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4860 = "tosa.reduce_sum"(%4859) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %4861 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4862 = "tosa.reciprocal"(%4861) : (tensor<1xf32>) -> tensor<1xf32>
    %4863 = "tosa.mul"(%4862, %4860) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4864 = "tosa.sub"(%4859, %4863) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4865 = "tosa.mul"(%4864, %4864) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4866 = "tosa.reduce_sum"(%4865) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %4867 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4868 = "tosa.reciprocal"(%4867) : (tensor<1xf32>) -> tensor<1xf32>
    %4869 = "tosa.mul"(%4868, %4866) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4870 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %4871 = "tosa.add"(%4869, %4870) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4872 = "tosa.rsqrt"(%4871) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4873 = "tosa.sub"(%4859, %4863) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4874 = "tosa.mul"(%4873, %4872) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4875 = "tosa.reshape"(%arg530) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4876 = "tosa.mul"(%4874, %4875) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4877 = "tosa.reshape"(%arg531) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4878 = "tosa.add"(%4876, %4877) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4879 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4880 = "tosa.transpose"(%arg532, %4879) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4881 = "tosa.reshape"(%4878) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4882 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %4883 = "linalg.matmul"(%4881, %4880, %4882) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4884 = "tosa.reshape"(%4883) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4885 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4886 = "tosa.transpose"(%arg533, %4885) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4887 = "tosa.reshape"(%4878) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4888 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %4889 = "linalg.matmul"(%4887, %4886, %4888) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4890 = "tosa.reshape"(%4889) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4891 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4892 = "tosa.transpose"(%arg534, %4891) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4893 = "tosa.reshape"(%4878) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4894 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %4895 = "linalg.matmul"(%4893, %4892, %4894) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4896 = "tosa.reshape"(%4895) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4897 = "tosa.reshape"(%4884) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %4898 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4899 = "tosa.transpose"(%4897, %4898) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %4900 = "tosa.reshape"(%4890) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %4901 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4902 = "tosa.transpose"(%4900, %4901) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %4903 = "tosa.reshape"(%4896) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %4904 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4905 = "tosa.transpose"(%4903, %4904) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %4906 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x1024xf32>}> : () -> tensor<1024x1024xf32>
    %4907 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4908 = "tosa.transpose"(%4902, %4907) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x10x64x1024xf32>
    %4909 = "tosa.reshape"(%4899) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %4910 = "tosa.reshape"(%4908) <{new_shape = array<i64: 10, 64, 1024>}> : (tensor<1x10x64x1024xf32>) -> tensor<10x64x1024xf32>
    %4911 = "tosa.matmul"(%4909, %4910) : (tensor<10x1024x64xf32>, tensor<10x64x1024xf32>) -> tensor<10x1024x1024xf32>
    %4912 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<10x1024x1024xf32>}> : () -> tensor<10x1024x1024xf32>
    %4913 = "tosa.mul"(%4911, %4912) <{shift = 0 : i8}> : (tensor<10x1024x1024xf32>, tensor<10x1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %4914 = "tosa.add"(%4913, %4906) : (tensor<10x1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %4915 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<10x1024x1024xf32>}> : () -> tensor<10x1024x1024xf32>
    %4916 = "linalg.softmax"(%4914, %4915) <{dimension = 3 : i64}> : (tensor<10x1024x1024xf32>, tensor<10x1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %4917 = "tosa.reshape"(%4905) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %4918 = "tosa.matmul"(%4916, %4917) : (tensor<10x1024x1024xf32>, tensor<10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %4919 = "tosa.reshape"(%4918) <{new_shape = array<i64: 1, 10, 1024, 64>}> : (tensor<10x1024x64xf32>) -> tensor<1x10x1024x64xf32>
    %4920 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4921 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4922 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %4923 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4924 = "tosa.transpose"(%4919, %4923) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x1024x10x64xf32>
    %4925 = "tosa.reshape"(%4924) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x1024x10x64xf32>) -> tensor<1x1024x640xf32>
    %4926 = "tosa.reshape"(%4925) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4927 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4928 = "tosa.transpose"(%arg535, %4927) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4929 = "tosa.reshape"(%4926) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4930 = "tosa.reshape"(%4928) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %4931 = "tosa.matmul"(%4929, %4930) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %4932 = "tosa.reshape"(%4931) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4933 = "tosa.reshape"(%arg536) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %4934 = "tosa.add"(%4933, %4932) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4935 = "tosa.reshape"(%4934) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4936 = "tosa.identity"(%4935) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4937 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x640xf32>}> : () -> tensor<1x1024x640xf32>
    %4938 = "tosa.reciprocal"(%4937) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4939 = "tosa.mul"(%4936, %4938) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4940 = "tosa.add"(%4939, %4859) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4941 = "tosa.reduce_sum"(%4940) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %4942 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4943 = "tosa.reciprocal"(%4942) : (tensor<1xf32>) -> tensor<1xf32>
    %4944 = "tosa.mul"(%4943, %4941) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4945 = "tosa.sub"(%4940, %4944) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4946 = "tosa.mul"(%4945, %4945) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %4947 = "tosa.reduce_sum"(%4946) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %4948 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4949 = "tosa.reciprocal"(%4948) : (tensor<1xf32>) -> tensor<1xf32>
    %4950 = "tosa.mul"(%4949, %4947) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4951 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %4952 = "tosa.add"(%4950, %4951) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4953 = "tosa.rsqrt"(%4952) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %4954 = "tosa.sub"(%4940, %4944) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4955 = "tosa.mul"(%4954, %4953) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %4956 = "tosa.reshape"(%arg537) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4957 = "tosa.mul"(%4955, %4956) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4958 = "tosa.reshape"(%arg538) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %4959 = "tosa.add"(%4957, %4958) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %4960 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4961 = "tosa.transpose"(%arg539, %4960) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %4962 = "tosa.reshape"(%4959) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %4963 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %4964 = "linalg.matmul"(%4962, %4961, %4963) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %4965 = "tosa.reshape"(%4964) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %4966 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4967 = "tosa.transpose"(%arg540, %4966) : (tensor<640x1024xf32>, tensor<2xi32>) -> tensor<1024x640xf32>
    %4968 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %4969 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x640xf32>}> : () -> tensor<12x640xf32>
    %4970 = "linalg.matmul"(%4968, %4967, %4969) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x640xf32>, tensor<12x640xf32>) -> tensor<12x640xf32>
    %4971 = "tosa.reshape"(%4970) <{new_shape = array<i64: 1, 12, 640>}> : (tensor<12x640xf32>) -> tensor<1x12x640xf32>
    %4972 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %4973 = "tosa.transpose"(%arg542, %4972) : (tensor<640x1024xf32>, tensor<2xi32>) -> tensor<1024x640xf32>
    %4974 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %4975 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x640xf32>}> : () -> tensor<12x640xf32>
    %4976 = "linalg.matmul"(%4974, %4973, %4975) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x640xf32>, tensor<12x640xf32>) -> tensor<12x640xf32>
    %4977 = "tosa.reshape"(%4976) <{new_shape = array<i64: 1, 12, 640>}> : (tensor<12x640xf32>) -> tensor<1x12x640xf32>
    %4978 = "tosa.reshape"(%4965) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %4979 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4980 = "tosa.transpose"(%4978, %4979) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %4981 = "tosa.reshape"(%4971) <{new_shape = array<i64: 1, 12, 10, 64>}> : (tensor<1x12x640xf32>) -> tensor<1x12x10x64xf32>
    %4982 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4983 = "tosa.transpose"(%4981, %4982) : (tensor<1x12x10x64xf32>, tensor<4xi32>) -> tensor<1x10x12x64xf32>
    %4984 = "tosa.reshape"(%4977) <{new_shape = array<i64: 1, 12, 10, 64>}> : (tensor<1x12x640xf32>) -> tensor<1x12x10x64xf32>
    %4985 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4986 = "tosa.transpose"(%4984, %4985) : (tensor<1x12x10x64xf32>, tensor<4xi32>) -> tensor<1x10x12x64xf32>
    %4987 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x12xf32>}> : () -> tensor<1024x12xf32>
    %4988 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4989 = "tosa.transpose"(%4983, %4988) : (tensor<1x10x12x64xf32>, tensor<4xi32>) -> tensor<1x10x64x12xf32>
    %4990 = "tosa.reshape"(%4980) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %4991 = "tosa.reshape"(%4989) <{new_shape = array<i64: 10, 64, 12>}> : (tensor<1x10x64x12xf32>) -> tensor<10x64x12xf32>
    %4992 = "tosa.matmul"(%4990, %4991) : (tensor<10x1024x64xf32>, tensor<10x64x12xf32>) -> tensor<10x1024x12xf32>
    %4993 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<10x1024x12xf32>}> : () -> tensor<10x1024x12xf32>
    %4994 = "tosa.mul"(%4992, %4993) <{shift = 0 : i8}> : (tensor<10x1024x12xf32>, tensor<10x1024x12xf32>) -> tensor<10x1024x12xf32>
    %4995 = "tosa.add"(%4994, %4987) : (tensor<10x1024x12xf32>, tensor<1024x12xf32>) -> tensor<10x1024x12xf32>
    %4996 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<10x1024x12xf32>}> : () -> tensor<10x1024x12xf32>
    %4997 = "linalg.softmax"(%4995, %4996) <{dimension = 3 : i64}> : (tensor<10x1024x12xf32>, tensor<10x1024x12xf32>) -> tensor<10x1024x12xf32>
    %4998 = "tosa.reshape"(%4986) <{new_shape = array<i64: 10, 12, 64>}> : (tensor<1x10x12x64xf32>) -> tensor<10x12x64xf32>
    %4999 = "tosa.matmul"(%4997, %4998) : (tensor<10x1024x12xf32>, tensor<10x12x64xf32>) -> tensor<10x1024x64xf32>
    %5000 = "tosa.reshape"(%4999) <{new_shape = array<i64: 1, 10, 1024, 64>}> : (tensor<10x1024x64xf32>) -> tensor<1x10x1024x64xf32>
    %5001 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5002 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5003 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5004 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5005 = "tosa.transpose"(%5000, %5004) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x1024x10x64xf32>
    %5006 = "tosa.reshape"(%5005) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x1024x10x64xf32>) -> tensor<1x1024x640xf32>
    %5007 = "tosa.reshape"(%5006) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5008 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5009 = "tosa.transpose"(%arg544, %5008) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %5010 = "tosa.reshape"(%5007) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5011 = "tosa.reshape"(%5009) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %5012 = "tosa.matmul"(%5010, %5011) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %5013 = "tosa.reshape"(%5012) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5014 = "tosa.reshape"(%arg545) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5015 = "tosa.add"(%5014, %5013) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5016 = "tosa.reshape"(%5015) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5017 = "tosa.identity"(%5016) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5018 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x640xf32>}> : () -> tensor<1x1024x640xf32>
    %5019 = "tosa.reciprocal"(%5018) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5020 = "tosa.mul"(%5017, %5019) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5021 = "tosa.add"(%5020, %4940) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5022 = "tosa.reduce_sum"(%5021) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %5023 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5024 = "tosa.reciprocal"(%5023) : (tensor<1xf32>) -> tensor<1xf32>
    %5025 = "tosa.mul"(%5024, %5022) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5026 = "tosa.sub"(%5021, %5025) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5027 = "tosa.mul"(%5026, %5026) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5028 = "tosa.reduce_sum"(%5027) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %5029 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5030 = "tosa.reciprocal"(%5029) : (tensor<1xf32>) -> tensor<1xf32>
    %5031 = "tosa.mul"(%5030, %5028) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5032 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %5033 = "tosa.add"(%5031, %5032) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5034 = "tosa.rsqrt"(%5033) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5035 = "tosa.sub"(%5021, %5025) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5036 = "tosa.mul"(%5035, %5034) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5037 = "tosa.reshape"(%arg546) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %5038 = "tosa.mul"(%5036, %5037) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %5039 = "tosa.reshape"(%arg547) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %5040 = "tosa.add"(%5038, %5039) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %5041 = "tosa.reshape"(%5040) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5042 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5043 = "tosa.transpose"(%arg548, %5042) : (tensor<5120x640xf32>, tensor<2xi32>) -> tensor<640x5120xf32>
    %5044 = "tosa.reshape"(%5041) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5045 = "tosa.reshape"(%5043) <{new_shape = array<i64: 1, 640, 5120>}> : (tensor<640x5120xf32>) -> tensor<1x640x5120xf32>
    %5046 = "tosa.matmul"(%5044, %5045) : (tensor<1x1024x640xf32>, tensor<1x640x5120xf32>) -> tensor<1x1024x5120xf32>
    %5047 = "tosa.reshape"(%5046) <{new_shape = array<i64: 1024, 5120>}> : (tensor<1x1024x5120xf32>) -> tensor<1024x5120xf32>
    %5048 = "tosa.reshape"(%arg549) <{new_shape = array<i64: 1, 5120>}> : (tensor<5120xf32>) -> tensor<1x5120xf32>
    %5049 = "tosa.add"(%5048, %5047) : (tensor<1x5120xf32>, tensor<1024x5120xf32>) -> tensor<1024x5120xf32>
    %5050 = "tosa.reshape"(%5049) <{new_shape = array<i64: 1, 1024, 5120>}> : (tensor<1024x5120xf32>) -> tensor<1x1024x5120xf32>
    %5051 = "tosa.slice"(%5050) <{size = array<i64: 0, 0, 2560>, start = array<i64: 0, 0, 0>}> : (tensor<1x1024x5120xf32>) -> tensor<1x1024x2560xf32>
    %5052 = "tosa.slice"(%5050) <{size = array<i64: 0, 0, 5120>, start = array<i64: 0, 0, 2560>}> : (tensor<1x1024x5120xf32>) -> tensor<1x1024x2560xf32>
    %5053 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %5054 = "tosa.mul"(%5052, %5053) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5055 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %5056 = "tosa.mul"(%5052, %5055) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5057 = "math.erf"(%5056) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5058 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %5059 = "tosa.add"(%5057, %5058) : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5060 = "tosa.mul"(%5054, %5059) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5061 = "tosa.mul"(%5051, %5060) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5062 = "tosa.identity"(%5061) : (tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5063 = "tosa.reshape"(%5062) <{new_shape = array<i64: 1024, 2560>}> : (tensor<1x1024x2560xf32>) -> tensor<1024x2560xf32>
    %5064 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5065 = "tosa.transpose"(%arg550, %5064) : (tensor<640x2560xf32>, tensor<2xi32>) -> tensor<2560x640xf32>
    %5066 = "tosa.reshape"(%5063) <{new_shape = array<i64: 1, 1024, 2560>}> : (tensor<1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5067 = "tosa.reshape"(%5065) <{new_shape = array<i64: 1, 2560, 640>}> : (tensor<2560x640xf32>) -> tensor<1x2560x640xf32>
    %5068 = "tosa.matmul"(%5066, %5067) : (tensor<1x1024x2560xf32>, tensor<1x2560x640xf32>) -> tensor<1x1024x640xf32>
    %5069 = "tosa.reshape"(%5068) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5070 = "tosa.reshape"(%arg551) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5071 = "tosa.add"(%5070, %5069) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5072 = "tosa.reshape"(%5071) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5073 = "tosa.add"(%5072, %5021) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5074 = "tosa.reshape"(%5073) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5075 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5076 = "tosa.transpose"(%arg552, %5075) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %5077 = "tosa.reshape"(%5074) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5078 = "tosa.reshape"(%5076) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %5079 = "tosa.matmul"(%5077, %5078) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %5080 = "tosa.reshape"(%5079) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5081 = "tosa.reshape"(%arg553) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5082 = "tosa.add"(%5081, %5080) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5083 = "tosa.reshape"(%5082) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5084 = "tosa.reshape"(%5083) <{new_shape = array<i64: 1, 32, 32, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1x32x32x640xf32>
    %5085 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5086 = "tosa.transpose"(%5084, %5085) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %5087 = "tosa.identity"(%5086) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %5088 = "tosa.add"(%5087, %4821) : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %5089 = "tensor.empty"() : () -> tensor<1x960x32x32xf32>
    %5090 = "tensor.insert_slice"(%5088, %5089) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 640, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x640x32x32xf32>, tensor<1x960x32x32xf32>) -> tensor<1x960x32x32xf32>
    %5091 = "tensor.insert_slice"(%782, %5090) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 640, 0, 0>, static_sizes = array<i64: 1, 320, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x320x32x32xf32>, tensor<1x960x32x32xf32>) -> tensor<1x960x32x32xf32>
    %5092 = "tosa.reshape"(%5091) <{new_shape = array<i64: 1, 32, 30, 1024>}> : (tensor<1x960x32x32xf32>) -> tensor<1x32x30x1024xf32>
    %5093 = "tosa.reduce_sum"(%5092) <{axis = 2 : i32}> : (tensor<1x32x30x1024xf32>) -> tensor<1x32x1x1024xf32>
    %5094 = "tosa.reduce_sum"(%5093) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %5095 = "tosa.const"() <{value = dense<3.072000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5096 = "tosa.reciprocal"(%5095) : (tensor<1xf32>) -> tensor<1xf32>
    %5097 = "tosa.mul"(%5096, %5094) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5098 = "tosa.sub"(%5092, %5097) : (tensor<1x32x30x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x30x1024xf32>
    %5099 = "tosa.mul"(%5098, %5098) <{shift = 0 : i8}> : (tensor<1x32x30x1024xf32>, tensor<1x32x30x1024xf32>) -> tensor<1x32x30x1024xf32>
    %5100 = "tosa.reduce_sum"(%5099) <{axis = 2 : i32}> : (tensor<1x32x30x1024xf32>) -> tensor<1x32x1x1024xf32>
    %5101 = "tosa.reduce_sum"(%5100) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %5102 = "tosa.const"() <{value = dense<3.072000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5103 = "tosa.reciprocal"(%5102) : (tensor<1xf32>) -> tensor<1xf32>
    %5104 = "tosa.mul"(%5103, %5101) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5105 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %5106 = "tosa.add"(%5104, %5105) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5107 = "tosa.rsqrt"(%5106) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5108 = "tosa.sub"(%5092, %5097) : (tensor<1x32x30x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x30x1024xf32>
    %5109 = "tosa.mul"(%5108, %5107) <{shift = 0 : i8}> : (tensor<1x32x30x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x30x1024xf32>
    %5110 = "tosa.reshape"(%5109) <{new_shape = array<i64: 1, 960, 32, 32>}> : (tensor<1x32x30x1024xf32>) -> tensor<1x960x32x32xf32>
    %5111 = "tosa.reshape"(%arg554) <{new_shape = array<i64: 1, 960>}> : (tensor<960xf32>) -> tensor<1x960xf32>
    %5112 = "tosa.reshape"(%5111) <{new_shape = array<i64: 1, 960, 1>}> : (tensor<1x960xf32>) -> tensor<1x960x1xf32>
    %5113 = "tosa.reshape"(%5112) <{new_shape = array<i64: 1, 960, 1, 1>}> : (tensor<1x960x1xf32>) -> tensor<1x960x1x1xf32>
    %5114 = "tosa.reshape"(%arg555) <{new_shape = array<i64: 1, 960>}> : (tensor<960xf32>) -> tensor<1x960xf32>
    %5115 = "tosa.reshape"(%5114) <{new_shape = array<i64: 1, 960, 1>}> : (tensor<1x960xf32>) -> tensor<1x960x1xf32>
    %5116 = "tosa.reshape"(%5115) <{new_shape = array<i64: 1, 960, 1, 1>}> : (tensor<1x960x1xf32>) -> tensor<1x960x1x1xf32>
    %5117 = "tosa.mul"(%5110, %5116) <{shift = 0 : i8}> : (tensor<1x960x32x32xf32>, tensor<1x960x1x1xf32>) -> tensor<1x960x32x32xf32>
    %5118 = "tosa.add"(%5117, %5113) : (tensor<1x960x32x32xf32>, tensor<1x960x1x1xf32>) -> tensor<1x960x32x32xf32>
    %5119 = "tosa.sigmoid"(%5118) : (tensor<1x960x32x32xf32>) -> tensor<1x960x32x32xf32>
    %5120 = "tosa.mul"(%5118, %5119) <{shift = 0 : i8}> : (tensor<1x960x32x32xf32>, tensor<1x960x32x32xf32>) -> tensor<1x960x32x32xf32>
    %5121 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5122 = "tosa.transpose"(%5120, %5121) : (tensor<1x960x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x960xf32>
    %5123 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5124 = "tosa.transpose"(%arg556, %5123) : (tensor<640x960x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x960xf32>
    %5125 = "tosa.conv2d"(%5122, %5124, %arg557) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x960xf32>, tensor<640x3x3x960xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %5126 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5127 = "tosa.transpose"(%5125, %5126) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %5128 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %5129 = "tosa.mul"(%50, %5128) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %5130 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5131 = "tosa.transpose"(%arg558, %5130) : (tensor<640x1280xf32>, tensor<2xi32>) -> tensor<1280x640xf32>
    %5132 = "tosa.reshape"(%5129) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %5133 = "tosa.reshape"(%5131) <{new_shape = array<i64: 1, 1280, 640>}> : (tensor<1280x640xf32>) -> tensor<1x1280x640xf32>
    %5134 = "tosa.matmul"(%5132, %5133) : (tensor<1x1x1280xf32>, tensor<1x1280x640xf32>) -> tensor<1x1x640xf32>
    %5135 = "tosa.reshape"(%5134) <{new_shape = array<i64: 1, 640>}> : (tensor<1x1x640xf32>) -> tensor<1x640xf32>
    %5136 = "tosa.reshape"(%arg559) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5137 = "tosa.add"(%5136, %5135) : (tensor<1x640xf32>, tensor<1x640xf32>) -> tensor<1x640xf32>
    %5138 = "tensor.extract_slice"(%5137) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 640>, static_strides = array<i64: 1, 1>}> : (tensor<1x640xf32>) -> tensor<1x640xf32>
    %5139 = "tensor.extract_slice"(%5138) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 640>, static_strides = array<i64: 1, 1>}> : (tensor<1x640xf32>) -> tensor<1x640xf32>
    %5140 = "tosa.reshape"(%5139) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %5141 = "tosa.reshape"(%5140) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %5142 = "tosa.add"(%5127, %5141) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %5143 = "tosa.reshape"(%5142) <{new_shape = array<i64: 1, 32, 20, 1024>}> : (tensor<1x640x32x32xf32>) -> tensor<1x32x20x1024xf32>
    %5144 = "tosa.reduce_sum"(%5143) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %5145 = "tosa.reduce_sum"(%5144) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %5146 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5147 = "tosa.reciprocal"(%5146) : (tensor<1xf32>) -> tensor<1xf32>
    %5148 = "tosa.mul"(%5147, %5145) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5149 = "tosa.sub"(%5143, %5148) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %5150 = "tosa.mul"(%5149, %5149) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x20x1024xf32>) -> tensor<1x32x20x1024xf32>
    %5151 = "tosa.reduce_sum"(%5150) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %5152 = "tosa.reduce_sum"(%5151) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %5153 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5154 = "tosa.reciprocal"(%5153) : (tensor<1xf32>) -> tensor<1xf32>
    %5155 = "tosa.mul"(%5154, %5152) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5156 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %5157 = "tosa.add"(%5155, %5156) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5158 = "tosa.rsqrt"(%5157) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5159 = "tosa.sub"(%5143, %5148) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %5160 = "tosa.mul"(%5159, %5158) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %5161 = "tosa.reshape"(%5160) <{new_shape = array<i64: 1, 640, 32, 32>}> : (tensor<1x32x20x1024xf32>) -> tensor<1x640x32x32xf32>
    %5162 = "tosa.reshape"(%arg560) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5163 = "tosa.reshape"(%5162) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %5164 = "tosa.reshape"(%5163) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %5165 = "tosa.reshape"(%arg561) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5166 = "tosa.reshape"(%5165) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %5167 = "tosa.reshape"(%5166) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %5168 = "tosa.mul"(%5161, %5167) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %5169 = "tosa.add"(%5168, %5164) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %5170 = "tosa.sigmoid"(%5169) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %5171 = "tosa.mul"(%5169, %5170) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %5172 = "tosa.identity"(%5171) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %5173 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5174 = "tosa.transpose"(%5172, %5173) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %5175 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5176 = "tosa.transpose"(%arg562, %5175) : (tensor<640x640x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x640xf32>
    %5177 = "tosa.conv2d"(%5174, %5176, %arg563) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x640xf32>, tensor<640x3x3x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %5178 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5179 = "tosa.transpose"(%5177, %5178) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %5180 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5181 = "tosa.transpose"(%5091, %5180) : (tensor<1x960x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x960xf32>
    %5182 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5183 = "tosa.transpose"(%arg564, %5182) : (tensor<640x960x1x1xf32>, tensor<4xi32>) -> tensor<640x1x1x960xf32>
    %5184 = "tosa.conv2d"(%5181, %5183, %arg565) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x32x32x960xf32>, tensor<640x1x1x960xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %5185 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5186 = "tosa.transpose"(%5184, %5185) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %5187 = "tosa.add"(%5186, %5179) : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %5188 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x640x32x32xf32>}> : () -> tensor<1x640x32x32xf32>
    %5189 = "tosa.reciprocal"(%5188) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %5190 = "tosa.mul"(%5187, %5189) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %5191 = "tosa.reshape"(%5190) <{new_shape = array<i64: 1, 32, 20, 1024>}> : (tensor<1x640x32x32xf32>) -> tensor<1x32x20x1024xf32>
    %5192 = "tosa.reduce_sum"(%5191) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %5193 = "tosa.reduce_sum"(%5192) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %5194 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5195 = "tosa.reciprocal"(%5194) : (tensor<1xf32>) -> tensor<1xf32>
    %5196 = "tosa.mul"(%5195, %5193) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5197 = "tosa.sub"(%5191, %5196) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %5198 = "tosa.mul"(%5197, %5197) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x20x1024xf32>) -> tensor<1x32x20x1024xf32>
    %5199 = "tosa.reduce_sum"(%5198) <{axis = 2 : i32}> : (tensor<1x32x20x1024xf32>) -> tensor<1x32x1x1024xf32>
    %5200 = "tosa.reduce_sum"(%5199) <{axis = 3 : i32}> : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
    %5201 = "tosa.const"() <{value = dense<2.048000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5202 = "tosa.reciprocal"(%5201) : (tensor<1xf32>) -> tensor<1xf32>
    %5203 = "tosa.mul"(%5202, %5200) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5204 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %5205 = "tosa.add"(%5203, %5204) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5206 = "tosa.rsqrt"(%5205) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5207 = "tosa.sub"(%5191, %5196) : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %5208 = "tosa.mul"(%5207, %5206) <{shift = 0 : i8}> : (tensor<1x32x20x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x1024xf32>
    %5209 = "tosa.reshape"(%5208) <{new_shape = array<i64: 1, 640, 32, 32>}> : (tensor<1x32x20x1024xf32>) -> tensor<1x640x32x32xf32>
    %5210 = "tosa.reshape"(%arg566) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5211 = "tosa.reshape"(%5210) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %5212 = "tosa.reshape"(%5211) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %5213 = "tosa.reshape"(%arg567) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5214 = "tosa.reshape"(%5213) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %5215 = "tosa.reshape"(%5214) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %5216 = "tosa.mul"(%5209, %5215) <{shift = 0 : i8}> : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %5217 = "tosa.add"(%5216, %5212) : (tensor<1x640x32x32xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x32x32xf32>
    %5218 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5219 = "tosa.transpose"(%5217, %5218) : (tensor<1x640x32x32xf32>, tensor<4xi32>) -> tensor<1x32x32x640xf32>
    %5220 = "tosa.reshape"(%5219) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x32x32x640xf32>) -> tensor<1x1024x640xf32>
    %5221 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5222 = "tosa.transpose"(%arg568, %5221) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %5223 = "tosa.reshape"(%5220) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5224 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %5225 = "linalg.matmul"(%5223, %5222, %5224) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5226 = "tosa.reshape"(%5225) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5227 = "tosa.reshape"(%arg569) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %5228 = "tosa.add"(%5226, %5227) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %5229 = "tosa.reduce_sum"(%5228) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %5230 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5231 = "tosa.reciprocal"(%5230) : (tensor<1xf32>) -> tensor<1xf32>
    %5232 = "tosa.mul"(%5231, %5229) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5233 = "tosa.sub"(%5228, %5232) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5234 = "tosa.mul"(%5233, %5233) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5235 = "tosa.reduce_sum"(%5234) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %5236 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5237 = "tosa.reciprocal"(%5236) : (tensor<1xf32>) -> tensor<1xf32>
    %5238 = "tosa.mul"(%5237, %5235) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5239 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %5240 = "tosa.add"(%5238, %5239) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5241 = "tosa.rsqrt"(%5240) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5242 = "tosa.sub"(%5228, %5232) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5243 = "tosa.mul"(%5242, %5241) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5244 = "tosa.reshape"(%arg570) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %5245 = "tosa.mul"(%5243, %5244) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %5246 = "tosa.reshape"(%arg571) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %5247 = "tosa.add"(%5245, %5246) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %5248 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5249 = "tosa.transpose"(%arg572, %5248) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %5250 = "tosa.reshape"(%5247) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5251 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %5252 = "linalg.matmul"(%5250, %5249, %5251) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5253 = "tosa.reshape"(%5252) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5254 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5255 = "tosa.transpose"(%arg573, %5254) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %5256 = "tosa.reshape"(%5247) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5257 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %5258 = "linalg.matmul"(%5256, %5255, %5257) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5259 = "tosa.reshape"(%5258) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5260 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5261 = "tosa.transpose"(%arg574, %5260) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %5262 = "tosa.reshape"(%5247) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5263 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %5264 = "linalg.matmul"(%5262, %5261, %5263) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5265 = "tosa.reshape"(%5264) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5266 = "tosa.reshape"(%5253) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %5267 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5268 = "tosa.transpose"(%5266, %5267) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %5269 = "tosa.reshape"(%5259) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %5270 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5271 = "tosa.transpose"(%5269, %5270) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %5272 = "tosa.reshape"(%5265) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %5273 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5274 = "tosa.transpose"(%5272, %5273) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %5275 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x1024xf32>}> : () -> tensor<1024x1024xf32>
    %5276 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5277 = "tosa.transpose"(%5271, %5276) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x10x64x1024xf32>
    %5278 = "tosa.reshape"(%5268) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %5279 = "tosa.reshape"(%5277) <{new_shape = array<i64: 10, 64, 1024>}> : (tensor<1x10x64x1024xf32>) -> tensor<10x64x1024xf32>
    %5280 = "tosa.matmul"(%5278, %5279) : (tensor<10x1024x64xf32>, tensor<10x64x1024xf32>) -> tensor<10x1024x1024xf32>
    %5281 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<10x1024x1024xf32>}> : () -> tensor<10x1024x1024xf32>
    %5282 = "tosa.mul"(%5280, %5281) <{shift = 0 : i8}> : (tensor<10x1024x1024xf32>, tensor<10x1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %5283 = "tosa.add"(%5282, %5275) : (tensor<10x1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %5284 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<10x1024x1024xf32>}> : () -> tensor<10x1024x1024xf32>
    %5285 = "linalg.softmax"(%5283, %5284) <{dimension = 3 : i64}> : (tensor<10x1024x1024xf32>, tensor<10x1024x1024xf32>) -> tensor<10x1024x1024xf32>
    %5286 = "tosa.reshape"(%5274) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %5287 = "tosa.matmul"(%5285, %5286) : (tensor<10x1024x1024xf32>, tensor<10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %5288 = "tosa.reshape"(%5287) <{new_shape = array<i64: 1, 10, 1024, 64>}> : (tensor<10x1024x64xf32>) -> tensor<1x10x1024x64xf32>
    %5289 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5290 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5291 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5292 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5293 = "tosa.transpose"(%5288, %5292) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x1024x10x64xf32>
    %5294 = "tosa.reshape"(%5293) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x1024x10x64xf32>) -> tensor<1x1024x640xf32>
    %5295 = "tosa.reshape"(%5294) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5296 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5297 = "tosa.transpose"(%arg575, %5296) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %5298 = "tosa.reshape"(%5295) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5299 = "tosa.reshape"(%5297) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %5300 = "tosa.matmul"(%5298, %5299) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %5301 = "tosa.reshape"(%5300) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5302 = "tosa.reshape"(%arg576) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5303 = "tosa.add"(%5302, %5301) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5304 = "tosa.reshape"(%5303) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5305 = "tosa.identity"(%5304) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5306 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x640xf32>}> : () -> tensor<1x1024x640xf32>
    %5307 = "tosa.reciprocal"(%5306) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5308 = "tosa.mul"(%5305, %5307) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5309 = "tosa.add"(%5308, %5228) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5310 = "tosa.reduce_sum"(%5309) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %5311 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5312 = "tosa.reciprocal"(%5311) : (tensor<1xf32>) -> tensor<1xf32>
    %5313 = "tosa.mul"(%5312, %5310) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5314 = "tosa.sub"(%5309, %5313) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5315 = "tosa.mul"(%5314, %5314) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5316 = "tosa.reduce_sum"(%5315) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %5317 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5318 = "tosa.reciprocal"(%5317) : (tensor<1xf32>) -> tensor<1xf32>
    %5319 = "tosa.mul"(%5318, %5316) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5320 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %5321 = "tosa.add"(%5319, %5320) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5322 = "tosa.rsqrt"(%5321) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5323 = "tosa.sub"(%5309, %5313) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5324 = "tosa.mul"(%5323, %5322) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5325 = "tosa.reshape"(%arg577) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %5326 = "tosa.mul"(%5324, %5325) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %5327 = "tosa.reshape"(%arg578) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %5328 = "tosa.add"(%5326, %5327) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %5329 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5330 = "tosa.transpose"(%arg579, %5329) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %5331 = "tosa.reshape"(%5328) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5332 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x640xf32>}> : () -> tensor<1024x640xf32>
    %5333 = "linalg.matmul"(%5331, %5330, %5332) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<1024x640xf32>, tensor<640x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5334 = "tosa.reshape"(%5333) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5335 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5336 = "tosa.transpose"(%arg580, %5335) : (tensor<640x1024xf32>, tensor<2xi32>) -> tensor<1024x640xf32>
    %5337 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %5338 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x640xf32>}> : () -> tensor<12x640xf32>
    %5339 = "linalg.matmul"(%5337, %5336, %5338) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x640xf32>, tensor<12x640xf32>) -> tensor<12x640xf32>
    %5340 = "tosa.reshape"(%5339) <{new_shape = array<i64: 1, 12, 640>}> : (tensor<12x640xf32>) -> tensor<1x12x640xf32>
    %5341 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5342 = "tosa.transpose"(%arg582, %5341) : (tensor<640x1024xf32>, tensor<2xi32>) -> tensor<1024x640xf32>
    %5343 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %5344 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x640xf32>}> : () -> tensor<12x640xf32>
    %5345 = "linalg.matmul"(%5343, %5342, %5344) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x640xf32>, tensor<12x640xf32>) -> tensor<12x640xf32>
    %5346 = "tosa.reshape"(%5345) <{new_shape = array<i64: 1, 12, 640>}> : (tensor<12x640xf32>) -> tensor<1x12x640xf32>
    %5347 = "tosa.reshape"(%5334) <{new_shape = array<i64: 1, 1024, 10, 64>}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x10x64xf32>
    %5348 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5349 = "tosa.transpose"(%5347, %5348) : (tensor<1x1024x10x64xf32>, tensor<4xi32>) -> tensor<1x10x1024x64xf32>
    %5350 = "tosa.reshape"(%5340) <{new_shape = array<i64: 1, 12, 10, 64>}> : (tensor<1x12x640xf32>) -> tensor<1x12x10x64xf32>
    %5351 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5352 = "tosa.transpose"(%5350, %5351) : (tensor<1x12x10x64xf32>, tensor<4xi32>) -> tensor<1x10x12x64xf32>
    %5353 = "tosa.reshape"(%5346) <{new_shape = array<i64: 1, 12, 10, 64>}> : (tensor<1x12x640xf32>) -> tensor<1x12x10x64xf32>
    %5354 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5355 = "tosa.transpose"(%5353, %5354) : (tensor<1x12x10x64xf32>, tensor<4xi32>) -> tensor<1x10x12x64xf32>
    %5356 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1024x12xf32>}> : () -> tensor<1024x12xf32>
    %5357 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5358 = "tosa.transpose"(%5352, %5357) : (tensor<1x10x12x64xf32>, tensor<4xi32>) -> tensor<1x10x64x12xf32>
    %5359 = "tosa.reshape"(%5349) <{new_shape = array<i64: 10, 1024, 64>}> : (tensor<1x10x1024x64xf32>) -> tensor<10x1024x64xf32>
    %5360 = "tosa.reshape"(%5358) <{new_shape = array<i64: 10, 64, 12>}> : (tensor<1x10x64x12xf32>) -> tensor<10x64x12xf32>
    %5361 = "tosa.matmul"(%5359, %5360) : (tensor<10x1024x64xf32>, tensor<10x64x12xf32>) -> tensor<10x1024x12xf32>
    %5362 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<10x1024x12xf32>}> : () -> tensor<10x1024x12xf32>
    %5363 = "tosa.mul"(%5361, %5362) <{shift = 0 : i8}> : (tensor<10x1024x12xf32>, tensor<10x1024x12xf32>) -> tensor<10x1024x12xf32>
    %5364 = "tosa.add"(%5363, %5356) : (tensor<10x1024x12xf32>, tensor<1024x12xf32>) -> tensor<10x1024x12xf32>
    %5365 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<10x1024x12xf32>}> : () -> tensor<10x1024x12xf32>
    %5366 = "linalg.softmax"(%5364, %5365) <{dimension = 3 : i64}> : (tensor<10x1024x12xf32>, tensor<10x1024x12xf32>) -> tensor<10x1024x12xf32>
    %5367 = "tosa.reshape"(%5355) <{new_shape = array<i64: 10, 12, 64>}> : (tensor<1x10x12x64xf32>) -> tensor<10x12x64xf32>
    %5368 = "tosa.matmul"(%5366, %5367) : (tensor<10x1024x12xf32>, tensor<10x12x64xf32>) -> tensor<10x1024x64xf32>
    %5369 = "tosa.reshape"(%5368) <{new_shape = array<i64: 1, 10, 1024, 64>}> : (tensor<10x1024x64xf32>) -> tensor<1x10x1024x64xf32>
    %5370 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5371 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5372 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5373 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5374 = "tosa.transpose"(%5369, %5373) : (tensor<1x10x1024x64xf32>, tensor<4xi32>) -> tensor<1x1024x10x64xf32>
    %5375 = "tosa.reshape"(%5374) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1x1024x10x64xf32>) -> tensor<1x1024x640xf32>
    %5376 = "tosa.reshape"(%5375) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5377 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5378 = "tosa.transpose"(%arg584, %5377) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %5379 = "tosa.reshape"(%5376) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5380 = "tosa.reshape"(%5378) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %5381 = "tosa.matmul"(%5379, %5380) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %5382 = "tosa.reshape"(%5381) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5383 = "tosa.reshape"(%arg585) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5384 = "tosa.add"(%5383, %5382) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5385 = "tosa.reshape"(%5384) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5386 = "tosa.identity"(%5385) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5387 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x640xf32>}> : () -> tensor<1x1024x640xf32>
    %5388 = "tosa.reciprocal"(%5387) : (tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5389 = "tosa.mul"(%5386, %5388) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5390 = "tosa.add"(%5389, %5309) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5391 = "tosa.reduce_sum"(%5390) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %5392 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5393 = "tosa.reciprocal"(%5392) : (tensor<1xf32>) -> tensor<1xf32>
    %5394 = "tosa.mul"(%5393, %5391) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5395 = "tosa.sub"(%5390, %5394) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5396 = "tosa.mul"(%5395, %5395) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5397 = "tosa.reduce_sum"(%5396) <{axis = 2 : i32}> : (tensor<1x1024x640xf32>) -> tensor<1x1024x1xf32>
    %5398 = "tosa.const"() <{value = dense<6.400000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5399 = "tosa.reciprocal"(%5398) : (tensor<1xf32>) -> tensor<1xf32>
    %5400 = "tosa.mul"(%5399, %5397) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5401 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1024x1xf32>}> : () -> tensor<1x1024x1xf32>
    %5402 = "tosa.add"(%5400, %5401) : (tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5403 = "tosa.rsqrt"(%5402) : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %5404 = "tosa.sub"(%5390, %5394) : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5405 = "tosa.mul"(%5404, %5403) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x640xf32>
    %5406 = "tosa.reshape"(%arg586) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %5407 = "tosa.mul"(%5405, %5406) <{shift = 0 : i8}> : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %5408 = "tosa.reshape"(%arg587) <{new_shape = array<i64: 1, 1, 640>}> : (tensor<640xf32>) -> tensor<1x1x640xf32>
    %5409 = "tosa.add"(%5407, %5408) : (tensor<1x1024x640xf32>, tensor<1x1x640xf32>) -> tensor<1x1024x640xf32>
    %5410 = "tosa.reshape"(%5409) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5411 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5412 = "tosa.transpose"(%arg588, %5411) : (tensor<5120x640xf32>, tensor<2xi32>) -> tensor<640x5120xf32>
    %5413 = "tosa.reshape"(%5410) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5414 = "tosa.reshape"(%5412) <{new_shape = array<i64: 1, 640, 5120>}> : (tensor<640x5120xf32>) -> tensor<1x640x5120xf32>
    %5415 = "tosa.matmul"(%5413, %5414) : (tensor<1x1024x640xf32>, tensor<1x640x5120xf32>) -> tensor<1x1024x5120xf32>
    %5416 = "tosa.reshape"(%5415) <{new_shape = array<i64: 1024, 5120>}> : (tensor<1x1024x5120xf32>) -> tensor<1024x5120xf32>
    %5417 = "tosa.reshape"(%arg589) <{new_shape = array<i64: 1, 5120>}> : (tensor<5120xf32>) -> tensor<1x5120xf32>
    %5418 = "tosa.add"(%5417, %5416) : (tensor<1x5120xf32>, tensor<1024x5120xf32>) -> tensor<1024x5120xf32>
    %5419 = "tosa.reshape"(%5418) <{new_shape = array<i64: 1, 1024, 5120>}> : (tensor<1024x5120xf32>) -> tensor<1x1024x5120xf32>
    %5420 = "tosa.slice"(%5419) <{size = array<i64: 0, 0, 2560>, start = array<i64: 0, 0, 0>}> : (tensor<1x1024x5120xf32>) -> tensor<1x1024x2560xf32>
    %5421 = "tosa.slice"(%5419) <{size = array<i64: 0, 0, 5120>, start = array<i64: 0, 0, 2560>}> : (tensor<1x1024x5120xf32>) -> tensor<1x1024x2560xf32>
    %5422 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %5423 = "tosa.mul"(%5421, %5422) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5424 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %5425 = "tosa.mul"(%5421, %5424) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5426 = "math.erf"(%5425) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5427 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1024x2560xf32>}> : () -> tensor<1x1024x2560xf32>
    %5428 = "tosa.add"(%5426, %5427) : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5429 = "tosa.mul"(%5423, %5428) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5430 = "tosa.mul"(%5420, %5429) <{shift = 0 : i8}> : (tensor<1x1024x2560xf32>, tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5431 = "tosa.identity"(%5430) : (tensor<1x1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5432 = "tosa.reshape"(%5431) <{new_shape = array<i64: 1024, 2560>}> : (tensor<1x1024x2560xf32>) -> tensor<1024x2560xf32>
    %5433 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5434 = "tosa.transpose"(%arg590, %5433) : (tensor<640x2560xf32>, tensor<2xi32>) -> tensor<2560x640xf32>
    %5435 = "tosa.reshape"(%5432) <{new_shape = array<i64: 1, 1024, 2560>}> : (tensor<1024x2560xf32>) -> tensor<1x1024x2560xf32>
    %5436 = "tosa.reshape"(%5434) <{new_shape = array<i64: 1, 2560, 640>}> : (tensor<2560x640xf32>) -> tensor<1x2560x640xf32>
    %5437 = "tosa.matmul"(%5435, %5436) : (tensor<1x1024x2560xf32>, tensor<1x2560x640xf32>) -> tensor<1x1024x640xf32>
    %5438 = "tosa.reshape"(%5437) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5439 = "tosa.reshape"(%arg591) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5440 = "tosa.add"(%5439, %5438) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5441 = "tosa.reshape"(%5440) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5442 = "tosa.add"(%5441, %5390) : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xf32>
    %5443 = "tosa.reshape"(%5442) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5444 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5445 = "tosa.transpose"(%arg592, %5444) : (tensor<640x640xf32>, tensor<2xi32>) -> tensor<640x640xf32>
    %5446 = "tosa.reshape"(%5443) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5447 = "tosa.reshape"(%5445) <{new_shape = array<i64: 1, 640, 640>}> : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %5448 = "tosa.matmul"(%5446, %5447) : (tensor<1x1024x640xf32>, tensor<1x640x640xf32>) -> tensor<1x1024x640xf32>
    %5449 = "tosa.reshape"(%5448) <{new_shape = array<i64: 1024, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
    %5450 = "tosa.reshape"(%arg593) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5451 = "tosa.add"(%5450, %5449) : (tensor<1x640xf32>, tensor<1024x640xf32>) -> tensor<1024x640xf32>
    %5452 = "tosa.reshape"(%5451) <{new_shape = array<i64: 1, 1024, 640>}> : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
    %5453 = "tosa.reshape"(%5452) <{new_shape = array<i64: 1, 32, 32, 640>}> : (tensor<1x1024x640xf32>) -> tensor<1x32x32x640xf32>
    %5454 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5455 = "tosa.transpose"(%5453, %5454) : (tensor<1x32x32x640xf32>, tensor<4xi32>) -> tensor<1x640x32x32xf32>
    %5456 = "tosa.identity"(%5455) : (tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %5457 = "tosa.add"(%5456, %5190) : (tensor<1x640x32x32xf32>, tensor<1x640x32x32xf32>) -> tensor<1x640x32x32xf32>
    %5458 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>}> : () -> tensor<64xi64>
    %5459 = "tosa.cast"(%5458) : (tensor<64xi64>) -> tensor<64xf32>
    %5460 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %5461 = "tosa.mul"(%5459, %5460) <{shift = 0 : i8}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %5462 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %5463 = "tosa.add"(%5461, %5462) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %5464 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<64xf32>}> : () -> tensor<64xf32>
    %5465 = "tosa.mul"(%5463, %5464) <{shift = 0 : i8}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %5466 = "tosa.cast"(%5465) : (tensor<64xf32>) -> tensor<64xi64>
    %5467 = "tosa.reshape"(%5466) <{new_shape = array<i64: 64, 1>}> : (tensor<64xi64>) -> tensor<64x1xi64>
    %5468 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>}> : () -> tensor<64xi64>
    %5469 = "tosa.cast"(%5468) : (tensor<64xi64>) -> tensor<64xf32>
    %5470 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %5471 = "tosa.mul"(%5469, %5470) <{shift = 0 : i8}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %5472 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %5473 = "tosa.add"(%5471, %5472) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %5474 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<64xf32>}> : () -> tensor<64xf32>
    %5475 = "tosa.mul"(%5473, %5474) <{shift = 0 : i8}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %5476 = "tosa.cast"(%5475) : (tensor<64xf32>) -> tensor<64xi64>
    %5477 = "tensor.empty"() : () -> tensor<1x640x64x64xf32>
    %5478 = "linalg.generic"(%5467, %5476, %5477) <{indexing_maps = [#map3, #map3, #map4], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: i64, %arg721: i64, %arg722: f32):
      %6629 = "arith.index_cast"(%arg720) : (i64) -> index
      %6630 = "arith.index_cast"(%arg721) : (i64) -> index
      %6631 = "linalg.index"() <{dim = 2 : i64}> : () -> index
      %6632 = "tensor.extract"(%5457, %6629, %6630, %6631) : (tensor<1x640x32x32xf32>, index, index, index) -> f32
      "linalg.yield"(%6632) : (f32) -> ()
    }) : (tensor<64x1xi64>, tensor<64xi64>, tensor<1x640x64x64xf32>) -> tensor<1x640x64x64xf32>
    %5479 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5480 = "tosa.transpose"(%5478, %5479) : (tensor<1x640x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x640xf32>
    %5481 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5482 = "tosa.transpose"(%arg594, %5481) : (tensor<640x640x3x3xf32>, tensor<4xi32>) -> tensor<640x3x3x640xf32>
    %5483 = "tosa.conv2d"(%5480, %5482, %arg595) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x640xf32>, tensor<640x3x3x640xf32>, tensor<640xf32>) -> tensor<1x64x64x640xf32>
    %5484 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5485 = "tosa.transpose"(%5483, %5484) : (tensor<1x64x64x640xf32>, tensor<4xi32>) -> tensor<1x640x64x64xf32>
    %5486 = "tensor.empty"() : () -> tensor<1x960x64x64xf32>
    %5487 = "tensor.insert_slice"(%5485, %5486) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 640, 64, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x640x64x64xf32>, tensor<1x960x64x64xf32>) -> tensor<1x960x64x64xf32>
    %5488 = "tensor.insert_slice"(%775, %5487) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 640, 0, 0>, static_sizes = array<i64: 1, 320, 64, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x320x64x64xf32>, tensor<1x960x64x64xf32>) -> tensor<1x960x64x64xf32>
    %5489 = "tosa.reshape"(%5488) <{new_shape = array<i64: 1, 32, 30, 4096>}> : (tensor<1x960x64x64xf32>) -> tensor<1x32x30x4096xf32>
    %5490 = "tosa.reduce_sum"(%5489) <{axis = 2 : i32}> : (tensor<1x32x30x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5491 = "tosa.reduce_sum"(%5490) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5492 = "tosa.const"() <{value = dense<1.228800e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5493 = "tosa.reciprocal"(%5492) : (tensor<1xf32>) -> tensor<1xf32>
    %5494 = "tosa.mul"(%5493, %5491) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5495 = "tosa.sub"(%5489, %5494) : (tensor<1x32x30x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x30x4096xf32>
    %5496 = "tosa.mul"(%5495, %5495) <{shift = 0 : i8}> : (tensor<1x32x30x4096xf32>, tensor<1x32x30x4096xf32>) -> tensor<1x32x30x4096xf32>
    %5497 = "tosa.reduce_sum"(%5496) <{axis = 2 : i32}> : (tensor<1x32x30x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5498 = "tosa.reduce_sum"(%5497) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5499 = "tosa.const"() <{value = dense<1.228800e+05> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5500 = "tosa.reciprocal"(%5499) : (tensor<1xf32>) -> tensor<1xf32>
    %5501 = "tosa.mul"(%5500, %5498) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5502 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %5503 = "tosa.add"(%5501, %5502) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5504 = "tosa.rsqrt"(%5503) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5505 = "tosa.sub"(%5489, %5494) : (tensor<1x32x30x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x30x4096xf32>
    %5506 = "tosa.mul"(%5505, %5504) <{shift = 0 : i8}> : (tensor<1x32x30x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x30x4096xf32>
    %5507 = "tosa.reshape"(%5506) <{new_shape = array<i64: 1, 960, 64, 64>}> : (tensor<1x32x30x4096xf32>) -> tensor<1x960x64x64xf32>
    %5508 = "tosa.reshape"(%arg596) <{new_shape = array<i64: 1, 960>}> : (tensor<960xf32>) -> tensor<1x960xf32>
    %5509 = "tosa.reshape"(%5508) <{new_shape = array<i64: 1, 960, 1>}> : (tensor<1x960xf32>) -> tensor<1x960x1xf32>
    %5510 = "tosa.reshape"(%5509) <{new_shape = array<i64: 1, 960, 1, 1>}> : (tensor<1x960x1xf32>) -> tensor<1x960x1x1xf32>
    %5511 = "tosa.reshape"(%arg597) <{new_shape = array<i64: 1, 960>}> : (tensor<960xf32>) -> tensor<1x960xf32>
    %5512 = "tosa.reshape"(%5511) <{new_shape = array<i64: 1, 960, 1>}> : (tensor<1x960xf32>) -> tensor<1x960x1xf32>
    %5513 = "tosa.reshape"(%5512) <{new_shape = array<i64: 1, 960, 1, 1>}> : (tensor<1x960x1xf32>) -> tensor<1x960x1x1xf32>
    %5514 = "tosa.mul"(%5507, %5513) <{shift = 0 : i8}> : (tensor<1x960x64x64xf32>, tensor<1x960x1x1xf32>) -> tensor<1x960x64x64xf32>
    %5515 = "tosa.add"(%5514, %5510) : (tensor<1x960x64x64xf32>, tensor<1x960x1x1xf32>) -> tensor<1x960x64x64xf32>
    %5516 = "tosa.sigmoid"(%5515) : (tensor<1x960x64x64xf32>) -> tensor<1x960x64x64xf32>
    %5517 = "tosa.mul"(%5515, %5516) <{shift = 0 : i8}> : (tensor<1x960x64x64xf32>, tensor<1x960x64x64xf32>) -> tensor<1x960x64x64xf32>
    %5518 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5519 = "tosa.transpose"(%5517, %5518) : (tensor<1x960x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x960xf32>
    %5520 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5521 = "tosa.transpose"(%arg598, %5520) : (tensor<320x960x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x960xf32>
    %5522 = "tosa.conv2d"(%5519, %5521, %arg599) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x960xf32>, tensor<320x3x3x960xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %5523 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5524 = "tosa.transpose"(%5522, %5523) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %5525 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %5526 = "tosa.mul"(%50, %5525) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %5527 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5528 = "tosa.transpose"(%arg600, %5527) : (tensor<320x1280xf32>, tensor<2xi32>) -> tensor<1280x320xf32>
    %5529 = "tosa.reshape"(%5526) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %5530 = "tosa.reshape"(%5528) <{new_shape = array<i64: 1, 1280, 320>}> : (tensor<1280x320xf32>) -> tensor<1x1280x320xf32>
    %5531 = "tosa.matmul"(%5529, %5530) : (tensor<1x1x1280xf32>, tensor<1x1280x320xf32>) -> tensor<1x1x320xf32>
    %5532 = "tosa.reshape"(%5531) <{new_shape = array<i64: 1, 320>}> : (tensor<1x1x320xf32>) -> tensor<1x320xf32>
    %5533 = "tosa.reshape"(%arg601) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5534 = "tosa.add"(%5533, %5532) : (tensor<1x320xf32>, tensor<1x320xf32>) -> tensor<1x320xf32>
    %5535 = "tensor.extract_slice"(%5534) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %5536 = "tensor.extract_slice"(%5535) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %5537 = "tosa.reshape"(%5536) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %5538 = "tosa.reshape"(%5537) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %5539 = "tosa.add"(%5524, %5538) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %5540 = "tosa.reshape"(%5539) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %5541 = "tosa.reduce_sum"(%5540) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5542 = "tosa.reduce_sum"(%5541) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5543 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5544 = "tosa.reciprocal"(%5543) : (tensor<1xf32>) -> tensor<1xf32>
    %5545 = "tosa.mul"(%5544, %5542) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5546 = "tosa.sub"(%5540, %5545) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5547 = "tosa.mul"(%5546, %5546) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %5548 = "tosa.reduce_sum"(%5547) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5549 = "tosa.reduce_sum"(%5548) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5550 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5551 = "tosa.reciprocal"(%5550) : (tensor<1xf32>) -> tensor<1xf32>
    %5552 = "tosa.mul"(%5551, %5549) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5553 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %5554 = "tosa.add"(%5552, %5553) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5555 = "tosa.rsqrt"(%5554) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5556 = "tosa.sub"(%5540, %5545) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5557 = "tosa.mul"(%5556, %5555) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5558 = "tosa.reshape"(%5557) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %5559 = "tosa.reshape"(%arg602) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5560 = "tosa.reshape"(%5559) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %5561 = "tosa.reshape"(%5560) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %5562 = "tosa.reshape"(%arg603) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5563 = "tosa.reshape"(%5562) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %5564 = "tosa.reshape"(%5563) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %5565 = "tosa.mul"(%5558, %5564) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %5566 = "tosa.add"(%5565, %5561) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %5567 = "tosa.sigmoid"(%5566) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5568 = "tosa.mul"(%5566, %5567) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5569 = "tosa.identity"(%5568) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5570 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5571 = "tosa.transpose"(%5569, %5570) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %5572 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5573 = "tosa.transpose"(%arg604, %5572) : (tensor<320x320x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x320xf32>
    %5574 = "tosa.conv2d"(%5571, %5573, %arg605) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x320xf32>, tensor<320x3x3x320xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %5575 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5576 = "tosa.transpose"(%5574, %5575) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %5577 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5578 = "tosa.transpose"(%5488, %5577) : (tensor<1x960x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x960xf32>
    %5579 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5580 = "tosa.transpose"(%arg606, %5579) : (tensor<320x960x1x1xf32>, tensor<4xi32>) -> tensor<320x1x1x960xf32>
    %5581 = "tosa.conv2d"(%5578, %5580, %arg607) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x960xf32>, tensor<320x1x1x960xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %5582 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5583 = "tosa.transpose"(%5581, %5582) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %5584 = "tosa.add"(%5583, %5576) : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5585 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x320x64x64xf32>}> : () -> tensor<1x320x64x64xf32>
    %5586 = "tosa.reciprocal"(%5585) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5587 = "tosa.mul"(%5584, %5586) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5588 = "tosa.reshape"(%5587) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %5589 = "tosa.reduce_sum"(%5588) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5590 = "tosa.reduce_sum"(%5589) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5591 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5592 = "tosa.reciprocal"(%5591) : (tensor<1xf32>) -> tensor<1xf32>
    %5593 = "tosa.mul"(%5592, %5590) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5594 = "tosa.sub"(%5588, %5593) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5595 = "tosa.mul"(%5594, %5594) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %5596 = "tosa.reduce_sum"(%5595) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5597 = "tosa.reduce_sum"(%5596) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5598 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5599 = "tosa.reciprocal"(%5598) : (tensor<1xf32>) -> tensor<1xf32>
    %5600 = "tosa.mul"(%5599, %5597) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5601 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %5602 = "tosa.add"(%5600, %5601) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5603 = "tosa.rsqrt"(%5602) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5604 = "tosa.sub"(%5588, %5593) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5605 = "tosa.mul"(%5604, %5603) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5606 = "tosa.reshape"(%5605) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %5607 = "tosa.reshape"(%arg608) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5608 = "tosa.reshape"(%5607) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %5609 = "tosa.reshape"(%5608) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %5610 = "tosa.reshape"(%arg609) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5611 = "tosa.reshape"(%5610) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %5612 = "tosa.reshape"(%5611) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %5613 = "tosa.mul"(%5606, %5612) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %5614 = "tosa.add"(%5613, %5609) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %5615 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5616 = "tosa.transpose"(%5614, %5615) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %5617 = "tosa.reshape"(%5616) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x64x64x320xf32>) -> tensor<1x4096x320xf32>
    %5618 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5619 = "tosa.transpose"(%arg610, %5618) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %5620 = "tosa.reshape"(%5617) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5621 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %5622 = "linalg.matmul"(%5620, %5619, %5621) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %5623 = "tosa.reshape"(%5622) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5624 = "tosa.reshape"(%arg611) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %5625 = "tosa.add"(%5623, %5624) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %5626 = "tosa.reduce_sum"(%5625) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %5627 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5628 = "tosa.reciprocal"(%5627) : (tensor<1xf32>) -> tensor<1xf32>
    %5629 = "tosa.mul"(%5628, %5626) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5630 = "tosa.sub"(%5625, %5629) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %5631 = "tosa.mul"(%5630, %5630) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5632 = "tosa.reduce_sum"(%5631) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %5633 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5634 = "tosa.reciprocal"(%5633) : (tensor<1xf32>) -> tensor<1xf32>
    %5635 = "tosa.mul"(%5634, %5632) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5636 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %5637 = "tosa.add"(%5635, %5636) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5638 = "tosa.rsqrt"(%5637) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5639 = "tosa.sub"(%5625, %5629) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %5640 = "tosa.mul"(%5639, %5638) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %5641 = "tosa.reshape"(%arg612) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %5642 = "tosa.mul"(%5640, %5641) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %5643 = "tosa.reshape"(%arg613) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %5644 = "tosa.add"(%5642, %5643) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %5645 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5646 = "tosa.transpose"(%arg614, %5645) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %5647 = "tosa.reshape"(%5644) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5648 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %5649 = "linalg.matmul"(%5647, %5646, %5648) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %5650 = "tosa.reshape"(%5649) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5651 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5652 = "tosa.transpose"(%arg615, %5651) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %5653 = "tosa.reshape"(%5644) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5654 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %5655 = "linalg.matmul"(%5653, %5652, %5654) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %5656 = "tosa.reshape"(%5655) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5657 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5658 = "tosa.transpose"(%arg616, %5657) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %5659 = "tosa.reshape"(%5644) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5660 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %5661 = "linalg.matmul"(%5659, %5658, %5660) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %5662 = "tosa.reshape"(%5661) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5663 = "tosa.reshape"(%5650) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %5664 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5665 = "tosa.transpose"(%5663, %5664) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %5666 = "tosa.reshape"(%5656) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %5667 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5668 = "tosa.transpose"(%5666, %5667) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %5669 = "tosa.reshape"(%5662) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %5670 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5671 = "tosa.transpose"(%5669, %5670) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %5672 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x4096xf32>}> : () -> tensor<4096x4096xf32>
    %5673 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5674 = "tosa.transpose"(%5668, %5673) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x5x64x4096xf32>
    %5675 = "tosa.reshape"(%5665) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %5676 = "tosa.reshape"(%5674) <{new_shape = array<i64: 5, 64, 4096>}> : (tensor<1x5x64x4096xf32>) -> tensor<5x64x4096xf32>
    %5677 = "tosa.matmul"(%5675, %5676) : (tensor<5x4096x64xf32>, tensor<5x64x4096xf32>) -> tensor<5x4096x4096xf32>
    %5678 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<5x4096x4096xf32>}> : () -> tensor<5x4096x4096xf32>
    %5679 = "tosa.mul"(%5677, %5678) <{shift = 0 : i8}> : (tensor<5x4096x4096xf32>, tensor<5x4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %5680 = "tosa.add"(%5679, %5672) : (tensor<5x4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %5681 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<5x4096x4096xf32>}> : () -> tensor<5x4096x4096xf32>
    %5682 = "linalg.softmax"(%5680, %5681) <{dimension = 3 : i64}> : (tensor<5x4096x4096xf32>, tensor<5x4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %5683 = "tosa.reshape"(%5671) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %5684 = "tosa.matmul"(%5682, %5683) : (tensor<5x4096x4096xf32>, tensor<5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %5685 = "tosa.reshape"(%5684) <{new_shape = array<i64: 1, 5, 4096, 64>}> : (tensor<5x4096x64xf32>) -> tensor<1x5x4096x64xf32>
    %5686 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5687 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5688 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5689 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5690 = "tosa.transpose"(%5685, %5689) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x4096x5x64xf32>
    %5691 = "tosa.reshape"(%5690) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x4096x5x64xf32>) -> tensor<1x4096x320xf32>
    %5692 = "tosa.reshape"(%5691) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5693 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5694 = "tosa.transpose"(%arg617, %5693) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %5695 = "tosa.reshape"(%5692) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5696 = "tosa.reshape"(%5694) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %5697 = "tosa.matmul"(%5695, %5696) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %5698 = "tosa.reshape"(%5697) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5699 = "tosa.reshape"(%arg618) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5700 = "tosa.add"(%5699, %5698) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %5701 = "tosa.reshape"(%5700) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5702 = "tosa.identity"(%5701) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5703 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x320xf32>}> : () -> tensor<1x4096x320xf32>
    %5704 = "tosa.reciprocal"(%5703) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5705 = "tosa.mul"(%5702, %5704) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5706 = "tosa.add"(%5705, %5625) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5707 = "tosa.reduce_sum"(%5706) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %5708 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5709 = "tosa.reciprocal"(%5708) : (tensor<1xf32>) -> tensor<1xf32>
    %5710 = "tosa.mul"(%5709, %5707) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5711 = "tosa.sub"(%5706, %5710) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %5712 = "tosa.mul"(%5711, %5711) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5713 = "tosa.reduce_sum"(%5712) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %5714 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5715 = "tosa.reciprocal"(%5714) : (tensor<1xf32>) -> tensor<1xf32>
    %5716 = "tosa.mul"(%5715, %5713) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5717 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %5718 = "tosa.add"(%5716, %5717) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5719 = "tosa.rsqrt"(%5718) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5720 = "tosa.sub"(%5706, %5710) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %5721 = "tosa.mul"(%5720, %5719) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %5722 = "tosa.reshape"(%arg619) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %5723 = "tosa.mul"(%5721, %5722) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %5724 = "tosa.reshape"(%arg620) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %5725 = "tosa.add"(%5723, %5724) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %5726 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5727 = "tosa.transpose"(%arg621, %5726) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %5728 = "tosa.reshape"(%5725) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5729 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %5730 = "linalg.matmul"(%5728, %5727, %5729) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %5731 = "tosa.reshape"(%5730) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5732 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5733 = "tosa.transpose"(%arg622, %5732) : (tensor<320x1024xf32>, tensor<2xi32>) -> tensor<1024x320xf32>
    %5734 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %5735 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x320xf32>}> : () -> tensor<12x320xf32>
    %5736 = "linalg.matmul"(%5734, %5733, %5735) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x320xf32>, tensor<12x320xf32>) -> tensor<12x320xf32>
    %5737 = "tosa.reshape"(%5736) <{new_shape = array<i64: 1, 12, 320>}> : (tensor<12x320xf32>) -> tensor<1x12x320xf32>
    %5738 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5739 = "tosa.transpose"(%arg624, %5738) : (tensor<320x1024xf32>, tensor<2xi32>) -> tensor<1024x320xf32>
    %5740 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %5741 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x320xf32>}> : () -> tensor<12x320xf32>
    %5742 = "linalg.matmul"(%5740, %5739, %5741) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x320xf32>, tensor<12x320xf32>) -> tensor<12x320xf32>
    %5743 = "tosa.reshape"(%5742) <{new_shape = array<i64: 1, 12, 320>}> : (tensor<12x320xf32>) -> tensor<1x12x320xf32>
    %5744 = "tosa.reshape"(%5731) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %5745 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5746 = "tosa.transpose"(%5744, %5745) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %5747 = "tosa.reshape"(%5737) <{new_shape = array<i64: 1, 12, 5, 64>}> : (tensor<1x12x320xf32>) -> tensor<1x12x5x64xf32>
    %5748 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5749 = "tosa.transpose"(%5747, %5748) : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %5750 = "tosa.reshape"(%5743) <{new_shape = array<i64: 1, 12, 5, 64>}> : (tensor<1x12x320xf32>) -> tensor<1x12x5x64xf32>
    %5751 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5752 = "tosa.transpose"(%5750, %5751) : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %5753 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x12xf32>}> : () -> tensor<4096x12xf32>
    %5754 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5755 = "tosa.transpose"(%5749, %5754) : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x5x64x12xf32>
    %5756 = "tosa.reshape"(%5746) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %5757 = "tosa.reshape"(%5755) <{new_shape = array<i64: 5, 64, 12>}> : (tensor<1x5x64x12xf32>) -> tensor<5x64x12xf32>
    %5758 = "tosa.matmul"(%5756, %5757) : (tensor<5x4096x64xf32>, tensor<5x64x12xf32>) -> tensor<5x4096x12xf32>
    %5759 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<5x4096x12xf32>}> : () -> tensor<5x4096x12xf32>
    %5760 = "tosa.mul"(%5758, %5759) <{shift = 0 : i8}> : (tensor<5x4096x12xf32>, tensor<5x4096x12xf32>) -> tensor<5x4096x12xf32>
    %5761 = "tosa.add"(%5760, %5753) : (tensor<5x4096x12xf32>, tensor<4096x12xf32>) -> tensor<5x4096x12xf32>
    %5762 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<5x4096x12xf32>}> : () -> tensor<5x4096x12xf32>
    %5763 = "linalg.softmax"(%5761, %5762) <{dimension = 3 : i64}> : (tensor<5x4096x12xf32>, tensor<5x4096x12xf32>) -> tensor<5x4096x12xf32>
    %5764 = "tosa.reshape"(%5752) <{new_shape = array<i64: 5, 12, 64>}> : (tensor<1x5x12x64xf32>) -> tensor<5x12x64xf32>
    %5765 = "tosa.matmul"(%5763, %5764) : (tensor<5x4096x12xf32>, tensor<5x12x64xf32>) -> tensor<5x4096x64xf32>
    %5766 = "tosa.reshape"(%5765) <{new_shape = array<i64: 1, 5, 4096, 64>}> : (tensor<5x4096x64xf32>) -> tensor<1x5x4096x64xf32>
    %5767 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5768 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5769 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %5770 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5771 = "tosa.transpose"(%5766, %5770) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x4096x5x64xf32>
    %5772 = "tosa.reshape"(%5771) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x4096x5x64xf32>) -> tensor<1x4096x320xf32>
    %5773 = "tosa.reshape"(%5772) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5774 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5775 = "tosa.transpose"(%arg626, %5774) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %5776 = "tosa.reshape"(%5773) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5777 = "tosa.reshape"(%5775) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %5778 = "tosa.matmul"(%5776, %5777) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %5779 = "tosa.reshape"(%5778) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5780 = "tosa.reshape"(%arg627) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5781 = "tosa.add"(%5780, %5779) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %5782 = "tosa.reshape"(%5781) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5783 = "tosa.identity"(%5782) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5784 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x320xf32>}> : () -> tensor<1x4096x320xf32>
    %5785 = "tosa.reciprocal"(%5784) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5786 = "tosa.mul"(%5783, %5785) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5787 = "tosa.add"(%5786, %5706) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5788 = "tosa.reduce_sum"(%5787) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %5789 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5790 = "tosa.reciprocal"(%5789) : (tensor<1xf32>) -> tensor<1xf32>
    %5791 = "tosa.mul"(%5790, %5788) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5792 = "tosa.sub"(%5787, %5791) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %5793 = "tosa.mul"(%5792, %5792) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5794 = "tosa.reduce_sum"(%5793) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %5795 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5796 = "tosa.reciprocal"(%5795) : (tensor<1xf32>) -> tensor<1xf32>
    %5797 = "tosa.mul"(%5796, %5794) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5798 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %5799 = "tosa.add"(%5797, %5798) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5800 = "tosa.rsqrt"(%5799) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5801 = "tosa.sub"(%5787, %5791) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %5802 = "tosa.mul"(%5801, %5800) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %5803 = "tosa.reshape"(%arg628) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %5804 = "tosa.mul"(%5802, %5803) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %5805 = "tosa.reshape"(%arg629) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %5806 = "tosa.add"(%5804, %5805) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %5807 = "tosa.reshape"(%5806) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5808 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5809 = "tosa.transpose"(%arg630, %5808) : (tensor<2560x320xf32>, tensor<2xi32>) -> tensor<320x2560xf32>
    %5810 = "tosa.reshape"(%5807) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5811 = "tosa.reshape"(%5809) <{new_shape = array<i64: 1, 320, 2560>}> : (tensor<320x2560xf32>) -> tensor<1x320x2560xf32>
    %5812 = "tosa.matmul"(%5810, %5811) : (tensor<1x4096x320xf32>, tensor<1x320x2560xf32>) -> tensor<1x4096x2560xf32>
    %5813 = "tosa.reshape"(%5812) <{new_shape = array<i64: 4096, 2560>}> : (tensor<1x4096x2560xf32>) -> tensor<4096x2560xf32>
    %5814 = "tosa.reshape"(%arg631) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %5815 = "tosa.add"(%5814, %5813) : (tensor<1x2560xf32>, tensor<4096x2560xf32>) -> tensor<4096x2560xf32>
    %5816 = "tosa.reshape"(%5815) <{new_shape = array<i64: 1, 4096, 2560>}> : (tensor<4096x2560xf32>) -> tensor<1x4096x2560xf32>
    %5817 = "tosa.slice"(%5816) <{size = array<i64: 0, 0, 1280>, start = array<i64: 0, 0, 0>}> : (tensor<1x4096x2560xf32>) -> tensor<1x4096x1280xf32>
    %5818 = "tosa.slice"(%5816) <{size = array<i64: 0, 0, 2560>, start = array<i64: 0, 0, 1280>}> : (tensor<1x4096x2560xf32>) -> tensor<1x4096x1280xf32>
    %5819 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %5820 = "tosa.mul"(%5818, %5819) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %5821 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %5822 = "tosa.mul"(%5818, %5821) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %5823 = "math.erf"(%5822) <{fastmath = #arith.fastmath<none>}> : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %5824 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %5825 = "tosa.add"(%5823, %5824) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %5826 = "tosa.mul"(%5820, %5825) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %5827 = "tosa.mul"(%5817, %5826) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %5828 = "tosa.identity"(%5827) : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %5829 = "tosa.reshape"(%5828) <{new_shape = array<i64: 4096, 1280>}> : (tensor<1x4096x1280xf32>) -> tensor<4096x1280xf32>
    %5830 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5831 = "tosa.transpose"(%arg632, %5830) : (tensor<320x1280xf32>, tensor<2xi32>) -> tensor<1280x320xf32>
    %5832 = "tosa.reshape"(%5829) <{new_shape = array<i64: 1, 4096, 1280>}> : (tensor<4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %5833 = "tosa.reshape"(%5831) <{new_shape = array<i64: 1, 1280, 320>}> : (tensor<1280x320xf32>) -> tensor<1x1280x320xf32>
    %5834 = "tosa.matmul"(%5832, %5833) : (tensor<1x4096x1280xf32>, tensor<1x1280x320xf32>) -> tensor<1x4096x320xf32>
    %5835 = "tosa.reshape"(%5834) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5836 = "tosa.reshape"(%arg633) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5837 = "tosa.add"(%5836, %5835) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %5838 = "tosa.reshape"(%5837) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5839 = "tosa.add"(%5838, %5787) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %5840 = "tosa.reshape"(%5839) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5841 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5842 = "tosa.transpose"(%arg634, %5841) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %5843 = "tosa.reshape"(%5840) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5844 = "tosa.reshape"(%5842) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %5845 = "tosa.matmul"(%5843, %5844) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %5846 = "tosa.reshape"(%5845) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5847 = "tosa.reshape"(%arg635) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5848 = "tosa.add"(%5847, %5846) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %5849 = "tosa.reshape"(%5848) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5850 = "tosa.reshape"(%5849) <{new_shape = array<i64: 1, 64, 64, 320>}> : (tensor<1x4096x320xf32>) -> tensor<1x64x64x320xf32>
    %5851 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5852 = "tosa.transpose"(%5850, %5851) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %5853 = "tosa.identity"(%5852) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5854 = "tosa.add"(%5853, %5587) : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5855 = "tensor.empty"() : () -> tensor<1x640x64x64xf32>
    %5856 = "tensor.insert_slice"(%5854, %5855) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 320, 64, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x320x64x64xf32>, tensor<1x640x64x64xf32>) -> tensor<1x640x64x64xf32>
    %5857 = "tensor.insert_slice"(%416, %5856) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 320, 0, 0>, static_sizes = array<i64: 1, 320, 64, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x320x64x64xf32>, tensor<1x640x64x64xf32>) -> tensor<1x640x64x64xf32>
    %5858 = "tosa.reshape"(%5857) <{new_shape = array<i64: 1, 32, 20, 4096>}> : (tensor<1x640x64x64xf32>) -> tensor<1x32x20x4096xf32>
    %5859 = "tosa.reduce_sum"(%5858) <{axis = 2 : i32}> : (tensor<1x32x20x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5860 = "tosa.reduce_sum"(%5859) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5861 = "tosa.const"() <{value = dense<8.192000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5862 = "tosa.reciprocal"(%5861) : (tensor<1xf32>) -> tensor<1xf32>
    %5863 = "tosa.mul"(%5862, %5860) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5864 = "tosa.sub"(%5858, %5863) : (tensor<1x32x20x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x4096xf32>
    %5865 = "tosa.mul"(%5864, %5864) <{shift = 0 : i8}> : (tensor<1x32x20x4096xf32>, tensor<1x32x20x4096xf32>) -> tensor<1x32x20x4096xf32>
    %5866 = "tosa.reduce_sum"(%5865) <{axis = 2 : i32}> : (tensor<1x32x20x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5867 = "tosa.reduce_sum"(%5866) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5868 = "tosa.const"() <{value = dense<8.192000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5869 = "tosa.reciprocal"(%5868) : (tensor<1xf32>) -> tensor<1xf32>
    %5870 = "tosa.mul"(%5869, %5867) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5871 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %5872 = "tosa.add"(%5870, %5871) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5873 = "tosa.rsqrt"(%5872) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5874 = "tosa.sub"(%5858, %5863) : (tensor<1x32x20x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x4096xf32>
    %5875 = "tosa.mul"(%5874, %5873) <{shift = 0 : i8}> : (tensor<1x32x20x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x4096xf32>
    %5876 = "tosa.reshape"(%5875) <{new_shape = array<i64: 1, 640, 64, 64>}> : (tensor<1x32x20x4096xf32>) -> tensor<1x640x64x64xf32>
    %5877 = "tosa.reshape"(%arg636) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5878 = "tosa.reshape"(%5877) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %5879 = "tosa.reshape"(%5878) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %5880 = "tosa.reshape"(%arg637) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %5881 = "tosa.reshape"(%5880) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %5882 = "tosa.reshape"(%5881) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %5883 = "tosa.mul"(%5876, %5882) <{shift = 0 : i8}> : (tensor<1x640x64x64xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x64x64xf32>
    %5884 = "tosa.add"(%5883, %5879) : (tensor<1x640x64x64xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x64x64xf32>
    %5885 = "tosa.sigmoid"(%5884) : (tensor<1x640x64x64xf32>) -> tensor<1x640x64x64xf32>
    %5886 = "tosa.mul"(%5884, %5885) <{shift = 0 : i8}> : (tensor<1x640x64x64xf32>, tensor<1x640x64x64xf32>) -> tensor<1x640x64x64xf32>
    %5887 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5888 = "tosa.transpose"(%5886, %5887) : (tensor<1x640x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x640xf32>
    %5889 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5890 = "tosa.transpose"(%arg638, %5889) : (tensor<320x640x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x640xf32>
    %5891 = "tosa.conv2d"(%5888, %5890, %arg639) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x640xf32>, tensor<320x3x3x640xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %5892 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5893 = "tosa.transpose"(%5891, %5892) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %5894 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %5895 = "tosa.mul"(%50, %5894) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %5896 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5897 = "tosa.transpose"(%arg640, %5896) : (tensor<320x1280xf32>, tensor<2xi32>) -> tensor<1280x320xf32>
    %5898 = "tosa.reshape"(%5895) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %5899 = "tosa.reshape"(%5897) <{new_shape = array<i64: 1, 1280, 320>}> : (tensor<1280x320xf32>) -> tensor<1x1280x320xf32>
    %5900 = "tosa.matmul"(%5898, %5899) : (tensor<1x1x1280xf32>, tensor<1x1280x320xf32>) -> tensor<1x1x320xf32>
    %5901 = "tosa.reshape"(%5900) <{new_shape = array<i64: 1, 320>}> : (tensor<1x1x320xf32>) -> tensor<1x320xf32>
    %5902 = "tosa.reshape"(%arg641) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5903 = "tosa.add"(%5902, %5901) : (tensor<1x320xf32>, tensor<1x320xf32>) -> tensor<1x320xf32>
    %5904 = "tensor.extract_slice"(%5903) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %5905 = "tensor.extract_slice"(%5904) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %5906 = "tosa.reshape"(%5905) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %5907 = "tosa.reshape"(%5906) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %5908 = "tosa.add"(%5893, %5907) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %5909 = "tosa.reshape"(%5908) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %5910 = "tosa.reduce_sum"(%5909) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5911 = "tosa.reduce_sum"(%5910) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5912 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5913 = "tosa.reciprocal"(%5912) : (tensor<1xf32>) -> tensor<1xf32>
    %5914 = "tosa.mul"(%5913, %5911) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5915 = "tosa.sub"(%5909, %5914) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5916 = "tosa.mul"(%5915, %5915) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %5917 = "tosa.reduce_sum"(%5916) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5918 = "tosa.reduce_sum"(%5917) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5919 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5920 = "tosa.reciprocal"(%5919) : (tensor<1xf32>) -> tensor<1xf32>
    %5921 = "tosa.mul"(%5920, %5918) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5922 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %5923 = "tosa.add"(%5921, %5922) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5924 = "tosa.rsqrt"(%5923) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5925 = "tosa.sub"(%5909, %5914) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5926 = "tosa.mul"(%5925, %5924) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5927 = "tosa.reshape"(%5926) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %5928 = "tosa.reshape"(%arg642) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5929 = "tosa.reshape"(%5928) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %5930 = "tosa.reshape"(%5929) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %5931 = "tosa.reshape"(%arg643) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5932 = "tosa.reshape"(%5931) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %5933 = "tosa.reshape"(%5932) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %5934 = "tosa.mul"(%5927, %5933) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %5935 = "tosa.add"(%5934, %5930) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %5936 = "tosa.sigmoid"(%5935) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5937 = "tosa.mul"(%5935, %5936) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5938 = "tosa.identity"(%5937) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5939 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5940 = "tosa.transpose"(%5938, %5939) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %5941 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5942 = "tosa.transpose"(%arg644, %5941) : (tensor<320x320x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x320xf32>
    %5943 = "tosa.conv2d"(%5940, %5942, %arg645) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x320xf32>, tensor<320x3x3x320xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %5944 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5945 = "tosa.transpose"(%5943, %5944) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %5946 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5947 = "tosa.transpose"(%5857, %5946) : (tensor<1x640x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x640xf32>
    %5948 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5949 = "tosa.transpose"(%arg646, %5948) : (tensor<320x640x1x1xf32>, tensor<4xi32>) -> tensor<320x1x1x640xf32>
    %5950 = "tosa.conv2d"(%5947, %5949, %arg647) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x640xf32>, tensor<320x1x1x640xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %5951 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5952 = "tosa.transpose"(%5950, %5951) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %5953 = "tosa.add"(%5952, %5945) : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5954 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x320x64x64xf32>}> : () -> tensor<1x320x64x64xf32>
    %5955 = "tosa.reciprocal"(%5954) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5956 = "tosa.mul"(%5953, %5955) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %5957 = "tosa.reshape"(%5956) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %5958 = "tosa.reduce_sum"(%5957) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5959 = "tosa.reduce_sum"(%5958) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5960 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5961 = "tosa.reciprocal"(%5960) : (tensor<1xf32>) -> tensor<1xf32>
    %5962 = "tosa.mul"(%5961, %5959) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5963 = "tosa.sub"(%5957, %5962) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5964 = "tosa.mul"(%5963, %5963) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %5965 = "tosa.reduce_sum"(%5964) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %5966 = "tosa.reduce_sum"(%5965) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %5967 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5968 = "tosa.reciprocal"(%5967) : (tensor<1xf32>) -> tensor<1xf32>
    %5969 = "tosa.mul"(%5968, %5966) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5970 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %5971 = "tosa.add"(%5969, %5970) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5972 = "tosa.rsqrt"(%5971) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %5973 = "tosa.sub"(%5957, %5962) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5974 = "tosa.mul"(%5973, %5972) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %5975 = "tosa.reshape"(%5974) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %5976 = "tosa.reshape"(%arg648) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5977 = "tosa.reshape"(%5976) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %5978 = "tosa.reshape"(%5977) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %5979 = "tosa.reshape"(%arg649) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %5980 = "tosa.reshape"(%5979) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %5981 = "tosa.reshape"(%5980) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %5982 = "tosa.mul"(%5975, %5981) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %5983 = "tosa.add"(%5982, %5978) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %5984 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5985 = "tosa.transpose"(%5983, %5984) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %5986 = "tosa.reshape"(%5985) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x64x64x320xf32>) -> tensor<1x4096x320xf32>
    %5987 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5988 = "tosa.transpose"(%arg650, %5987) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %5989 = "tosa.reshape"(%5986) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %5990 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %5991 = "linalg.matmul"(%5989, %5988, %5990) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %5992 = "tosa.reshape"(%5991) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %5993 = "tosa.reshape"(%arg651) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %5994 = "tosa.add"(%5992, %5993) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %5995 = "tosa.reduce_sum"(%5994) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %5996 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5997 = "tosa.reciprocal"(%5996) : (tensor<1xf32>) -> tensor<1xf32>
    %5998 = "tosa.mul"(%5997, %5995) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %5999 = "tosa.sub"(%5994, %5998) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6000 = "tosa.mul"(%5999, %5999) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6001 = "tosa.reduce_sum"(%6000) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %6002 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6003 = "tosa.reciprocal"(%6002) : (tensor<1xf32>) -> tensor<1xf32>
    %6004 = "tosa.mul"(%6003, %6001) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6005 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %6006 = "tosa.add"(%6004, %6005) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6007 = "tosa.rsqrt"(%6006) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6008 = "tosa.sub"(%5994, %5998) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6009 = "tosa.mul"(%6008, %6007) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6010 = "tosa.reshape"(%arg652) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6011 = "tosa.mul"(%6009, %6010) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6012 = "tosa.reshape"(%arg653) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6013 = "tosa.add"(%6011, %6012) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6014 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6015 = "tosa.transpose"(%arg654, %6014) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6016 = "tosa.reshape"(%6013) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6017 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %6018 = "linalg.matmul"(%6016, %6015, %6017) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6019 = "tosa.reshape"(%6018) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6020 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6021 = "tosa.transpose"(%arg655, %6020) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6022 = "tosa.reshape"(%6013) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6023 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %6024 = "linalg.matmul"(%6022, %6021, %6023) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6025 = "tosa.reshape"(%6024) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6026 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6027 = "tosa.transpose"(%arg656, %6026) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6028 = "tosa.reshape"(%6013) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6029 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %6030 = "linalg.matmul"(%6028, %6027, %6029) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6031 = "tosa.reshape"(%6030) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6032 = "tosa.reshape"(%6019) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %6033 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6034 = "tosa.transpose"(%6032, %6033) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %6035 = "tosa.reshape"(%6025) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %6036 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6037 = "tosa.transpose"(%6035, %6036) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %6038 = "tosa.reshape"(%6031) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %6039 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6040 = "tosa.transpose"(%6038, %6039) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %6041 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x4096xf32>}> : () -> tensor<4096x4096xf32>
    %6042 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6043 = "tosa.transpose"(%6037, %6042) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x5x64x4096xf32>
    %6044 = "tosa.reshape"(%6034) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %6045 = "tosa.reshape"(%6043) <{new_shape = array<i64: 5, 64, 4096>}> : (tensor<1x5x64x4096xf32>) -> tensor<5x64x4096xf32>
    %6046 = "tosa.matmul"(%6044, %6045) : (tensor<5x4096x64xf32>, tensor<5x64x4096xf32>) -> tensor<5x4096x4096xf32>
    %6047 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<5x4096x4096xf32>}> : () -> tensor<5x4096x4096xf32>
    %6048 = "tosa.mul"(%6046, %6047) <{shift = 0 : i8}> : (tensor<5x4096x4096xf32>, tensor<5x4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %6049 = "tosa.add"(%6048, %6041) : (tensor<5x4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %6050 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<5x4096x4096xf32>}> : () -> tensor<5x4096x4096xf32>
    %6051 = "linalg.softmax"(%6049, %6050) <{dimension = 3 : i64}> : (tensor<5x4096x4096xf32>, tensor<5x4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %6052 = "tosa.reshape"(%6040) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %6053 = "tosa.matmul"(%6051, %6052) : (tensor<5x4096x4096xf32>, tensor<5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %6054 = "tosa.reshape"(%6053) <{new_shape = array<i64: 1, 5, 4096, 64>}> : (tensor<5x4096x64xf32>) -> tensor<1x5x4096x64xf32>
    %6055 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6056 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6057 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6058 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6059 = "tosa.transpose"(%6054, %6058) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x4096x5x64xf32>
    %6060 = "tosa.reshape"(%6059) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x4096x5x64xf32>) -> tensor<1x4096x320xf32>
    %6061 = "tosa.reshape"(%6060) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6062 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6063 = "tosa.transpose"(%arg657, %6062) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6064 = "tosa.reshape"(%6061) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6065 = "tosa.reshape"(%6063) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %6066 = "tosa.matmul"(%6064, %6065) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %6067 = "tosa.reshape"(%6066) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6068 = "tosa.reshape"(%arg658) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6069 = "tosa.add"(%6068, %6067) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6070 = "tosa.reshape"(%6069) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6071 = "tosa.identity"(%6070) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6072 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x320xf32>}> : () -> tensor<1x4096x320xf32>
    %6073 = "tosa.reciprocal"(%6072) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6074 = "tosa.mul"(%6071, %6073) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6075 = "tosa.add"(%6074, %5994) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6076 = "tosa.reduce_sum"(%6075) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %6077 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6078 = "tosa.reciprocal"(%6077) : (tensor<1xf32>) -> tensor<1xf32>
    %6079 = "tosa.mul"(%6078, %6076) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6080 = "tosa.sub"(%6075, %6079) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6081 = "tosa.mul"(%6080, %6080) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6082 = "tosa.reduce_sum"(%6081) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %6083 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6084 = "tosa.reciprocal"(%6083) : (tensor<1xf32>) -> tensor<1xf32>
    %6085 = "tosa.mul"(%6084, %6082) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6086 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %6087 = "tosa.add"(%6085, %6086) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6088 = "tosa.rsqrt"(%6087) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6089 = "tosa.sub"(%6075, %6079) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6090 = "tosa.mul"(%6089, %6088) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6091 = "tosa.reshape"(%arg659) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6092 = "tosa.mul"(%6090, %6091) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6093 = "tosa.reshape"(%arg660) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6094 = "tosa.add"(%6092, %6093) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6095 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6096 = "tosa.transpose"(%arg661, %6095) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6097 = "tosa.reshape"(%6094) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6098 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %6099 = "linalg.matmul"(%6097, %6096, %6098) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6100 = "tosa.reshape"(%6099) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6101 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6102 = "tosa.transpose"(%arg662, %6101) : (tensor<320x1024xf32>, tensor<2xi32>) -> tensor<1024x320xf32>
    %6103 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %6104 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x320xf32>}> : () -> tensor<12x320xf32>
    %6105 = "linalg.matmul"(%6103, %6102, %6104) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x320xf32>, tensor<12x320xf32>) -> tensor<12x320xf32>
    %6106 = "tosa.reshape"(%6105) <{new_shape = array<i64: 1, 12, 320>}> : (tensor<12x320xf32>) -> tensor<1x12x320xf32>
    %6107 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6108 = "tosa.transpose"(%arg664, %6107) : (tensor<320x1024xf32>, tensor<2xi32>) -> tensor<1024x320xf32>
    %6109 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %6110 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x320xf32>}> : () -> tensor<12x320xf32>
    %6111 = "linalg.matmul"(%6109, %6108, %6110) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x320xf32>, tensor<12x320xf32>) -> tensor<12x320xf32>
    %6112 = "tosa.reshape"(%6111) <{new_shape = array<i64: 1, 12, 320>}> : (tensor<12x320xf32>) -> tensor<1x12x320xf32>
    %6113 = "tosa.reshape"(%6100) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %6114 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6115 = "tosa.transpose"(%6113, %6114) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %6116 = "tosa.reshape"(%6106) <{new_shape = array<i64: 1, 12, 5, 64>}> : (tensor<1x12x320xf32>) -> tensor<1x12x5x64xf32>
    %6117 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6118 = "tosa.transpose"(%6116, %6117) : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %6119 = "tosa.reshape"(%6112) <{new_shape = array<i64: 1, 12, 5, 64>}> : (tensor<1x12x320xf32>) -> tensor<1x12x5x64xf32>
    %6120 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6121 = "tosa.transpose"(%6119, %6120) : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %6122 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x12xf32>}> : () -> tensor<4096x12xf32>
    %6123 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6124 = "tosa.transpose"(%6118, %6123) : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x5x64x12xf32>
    %6125 = "tosa.reshape"(%6115) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %6126 = "tosa.reshape"(%6124) <{new_shape = array<i64: 5, 64, 12>}> : (tensor<1x5x64x12xf32>) -> tensor<5x64x12xf32>
    %6127 = "tosa.matmul"(%6125, %6126) : (tensor<5x4096x64xf32>, tensor<5x64x12xf32>) -> tensor<5x4096x12xf32>
    %6128 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<5x4096x12xf32>}> : () -> tensor<5x4096x12xf32>
    %6129 = "tosa.mul"(%6127, %6128) <{shift = 0 : i8}> : (tensor<5x4096x12xf32>, tensor<5x4096x12xf32>) -> tensor<5x4096x12xf32>
    %6130 = "tosa.add"(%6129, %6122) : (tensor<5x4096x12xf32>, tensor<4096x12xf32>) -> tensor<5x4096x12xf32>
    %6131 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<5x4096x12xf32>}> : () -> tensor<5x4096x12xf32>
    %6132 = "linalg.softmax"(%6130, %6131) <{dimension = 3 : i64}> : (tensor<5x4096x12xf32>, tensor<5x4096x12xf32>) -> tensor<5x4096x12xf32>
    %6133 = "tosa.reshape"(%6121) <{new_shape = array<i64: 5, 12, 64>}> : (tensor<1x5x12x64xf32>) -> tensor<5x12x64xf32>
    %6134 = "tosa.matmul"(%6132, %6133) : (tensor<5x4096x12xf32>, tensor<5x12x64xf32>) -> tensor<5x4096x64xf32>
    %6135 = "tosa.reshape"(%6134) <{new_shape = array<i64: 1, 5, 4096, 64>}> : (tensor<5x4096x64xf32>) -> tensor<1x5x4096x64xf32>
    %6136 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6137 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6138 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6139 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6140 = "tosa.transpose"(%6135, %6139) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x4096x5x64xf32>
    %6141 = "tosa.reshape"(%6140) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x4096x5x64xf32>) -> tensor<1x4096x320xf32>
    %6142 = "tosa.reshape"(%6141) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6143 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6144 = "tosa.transpose"(%arg666, %6143) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6145 = "tosa.reshape"(%6142) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6146 = "tosa.reshape"(%6144) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %6147 = "tosa.matmul"(%6145, %6146) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %6148 = "tosa.reshape"(%6147) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6149 = "tosa.reshape"(%arg667) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6150 = "tosa.add"(%6149, %6148) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6151 = "tosa.reshape"(%6150) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6152 = "tosa.identity"(%6151) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6153 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x320xf32>}> : () -> tensor<1x4096x320xf32>
    %6154 = "tosa.reciprocal"(%6153) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6155 = "tosa.mul"(%6152, %6154) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6156 = "tosa.add"(%6155, %6075) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6157 = "tosa.reduce_sum"(%6156) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %6158 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6159 = "tosa.reciprocal"(%6158) : (tensor<1xf32>) -> tensor<1xf32>
    %6160 = "tosa.mul"(%6159, %6157) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6161 = "tosa.sub"(%6156, %6160) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6162 = "tosa.mul"(%6161, %6161) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6163 = "tosa.reduce_sum"(%6162) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %6164 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6165 = "tosa.reciprocal"(%6164) : (tensor<1xf32>) -> tensor<1xf32>
    %6166 = "tosa.mul"(%6165, %6163) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6167 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %6168 = "tosa.add"(%6166, %6167) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6169 = "tosa.rsqrt"(%6168) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6170 = "tosa.sub"(%6156, %6160) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6171 = "tosa.mul"(%6170, %6169) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6172 = "tosa.reshape"(%arg668) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6173 = "tosa.mul"(%6171, %6172) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6174 = "tosa.reshape"(%arg669) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6175 = "tosa.add"(%6173, %6174) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6176 = "tosa.reshape"(%6175) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6177 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6178 = "tosa.transpose"(%arg670, %6177) : (tensor<2560x320xf32>, tensor<2xi32>) -> tensor<320x2560xf32>
    %6179 = "tosa.reshape"(%6176) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6180 = "tosa.reshape"(%6178) <{new_shape = array<i64: 1, 320, 2560>}> : (tensor<320x2560xf32>) -> tensor<1x320x2560xf32>
    %6181 = "tosa.matmul"(%6179, %6180) : (tensor<1x4096x320xf32>, tensor<1x320x2560xf32>) -> tensor<1x4096x2560xf32>
    %6182 = "tosa.reshape"(%6181) <{new_shape = array<i64: 4096, 2560>}> : (tensor<1x4096x2560xf32>) -> tensor<4096x2560xf32>
    %6183 = "tosa.reshape"(%arg671) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %6184 = "tosa.add"(%6183, %6182) : (tensor<1x2560xf32>, tensor<4096x2560xf32>) -> tensor<4096x2560xf32>
    %6185 = "tosa.reshape"(%6184) <{new_shape = array<i64: 1, 4096, 2560>}> : (tensor<4096x2560xf32>) -> tensor<1x4096x2560xf32>
    %6186 = "tosa.slice"(%6185) <{size = array<i64: 0, 0, 1280>, start = array<i64: 0, 0, 0>}> : (tensor<1x4096x2560xf32>) -> tensor<1x4096x1280xf32>
    %6187 = "tosa.slice"(%6185) <{size = array<i64: 0, 0, 2560>, start = array<i64: 0, 0, 1280>}> : (tensor<1x4096x2560xf32>) -> tensor<1x4096x1280xf32>
    %6188 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %6189 = "tosa.mul"(%6187, %6188) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6190 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %6191 = "tosa.mul"(%6187, %6190) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6192 = "math.erf"(%6191) <{fastmath = #arith.fastmath<none>}> : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6193 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %6194 = "tosa.add"(%6192, %6193) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6195 = "tosa.mul"(%6189, %6194) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6196 = "tosa.mul"(%6186, %6195) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6197 = "tosa.identity"(%6196) : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6198 = "tosa.reshape"(%6197) <{new_shape = array<i64: 4096, 1280>}> : (tensor<1x4096x1280xf32>) -> tensor<4096x1280xf32>
    %6199 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6200 = "tosa.transpose"(%arg672, %6199) : (tensor<320x1280xf32>, tensor<2xi32>) -> tensor<1280x320xf32>
    %6201 = "tosa.reshape"(%6198) <{new_shape = array<i64: 1, 4096, 1280>}> : (tensor<4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6202 = "tosa.reshape"(%6200) <{new_shape = array<i64: 1, 1280, 320>}> : (tensor<1280x320xf32>) -> tensor<1x1280x320xf32>
    %6203 = "tosa.matmul"(%6201, %6202) : (tensor<1x4096x1280xf32>, tensor<1x1280x320xf32>) -> tensor<1x4096x320xf32>
    %6204 = "tosa.reshape"(%6203) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6205 = "tosa.reshape"(%arg673) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6206 = "tosa.add"(%6205, %6204) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6207 = "tosa.reshape"(%6206) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6208 = "tosa.add"(%6207, %6156) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6209 = "tosa.reshape"(%6208) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6210 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6211 = "tosa.transpose"(%arg674, %6210) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6212 = "tosa.reshape"(%6209) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6213 = "tosa.reshape"(%6211) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %6214 = "tosa.matmul"(%6212, %6213) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %6215 = "tosa.reshape"(%6214) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6216 = "tosa.reshape"(%arg675) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6217 = "tosa.add"(%6216, %6215) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6218 = "tosa.reshape"(%6217) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6219 = "tosa.reshape"(%6218) <{new_shape = array<i64: 1, 64, 64, 320>}> : (tensor<1x4096x320xf32>) -> tensor<1x64x64x320xf32>
    %6220 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6221 = "tosa.transpose"(%6219, %6220) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %6222 = "tosa.identity"(%6221) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6223 = "tosa.add"(%6222, %5956) : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6224 = "tensor.empty"() : () -> tensor<1x640x64x64xf32>
    %6225 = "tensor.insert_slice"(%6223, %6224) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 320, 64, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x320x64x64xf32>, tensor<1x640x64x64xf32>) -> tensor<1x640x64x64xf32>
    %6226 = "tensor.insert_slice"(%57, %6225) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 320, 0, 0>, static_sizes = array<i64: 1, 320, 64, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x320x64x64xf32>, tensor<1x640x64x64xf32>) -> tensor<1x640x64x64xf32>
    %6227 = "tosa.reshape"(%6226) <{new_shape = array<i64: 1, 32, 20, 4096>}> : (tensor<1x640x64x64xf32>) -> tensor<1x32x20x4096xf32>
    %6228 = "tosa.reduce_sum"(%6227) <{axis = 2 : i32}> : (tensor<1x32x20x4096xf32>) -> tensor<1x32x1x4096xf32>
    %6229 = "tosa.reduce_sum"(%6228) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %6230 = "tosa.const"() <{value = dense<8.192000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6231 = "tosa.reciprocal"(%6230) : (tensor<1xf32>) -> tensor<1xf32>
    %6232 = "tosa.mul"(%6231, %6229) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6233 = "tosa.sub"(%6227, %6232) : (tensor<1x32x20x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x4096xf32>
    %6234 = "tosa.mul"(%6233, %6233) <{shift = 0 : i8}> : (tensor<1x32x20x4096xf32>, tensor<1x32x20x4096xf32>) -> tensor<1x32x20x4096xf32>
    %6235 = "tosa.reduce_sum"(%6234) <{axis = 2 : i32}> : (tensor<1x32x20x4096xf32>) -> tensor<1x32x1x4096xf32>
    %6236 = "tosa.reduce_sum"(%6235) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %6237 = "tosa.const"() <{value = dense<8.192000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6238 = "tosa.reciprocal"(%6237) : (tensor<1xf32>) -> tensor<1xf32>
    %6239 = "tosa.mul"(%6238, %6236) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6240 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %6241 = "tosa.add"(%6239, %6240) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6242 = "tosa.rsqrt"(%6241) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6243 = "tosa.sub"(%6227, %6232) : (tensor<1x32x20x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x4096xf32>
    %6244 = "tosa.mul"(%6243, %6242) <{shift = 0 : i8}> : (tensor<1x32x20x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x20x4096xf32>
    %6245 = "tosa.reshape"(%6244) <{new_shape = array<i64: 1, 640, 64, 64>}> : (tensor<1x32x20x4096xf32>) -> tensor<1x640x64x64xf32>
    %6246 = "tosa.reshape"(%arg676) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %6247 = "tosa.reshape"(%6246) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %6248 = "tosa.reshape"(%6247) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %6249 = "tosa.reshape"(%arg677) <{new_shape = array<i64: 1, 640>}> : (tensor<640xf32>) -> tensor<1x640xf32>
    %6250 = "tosa.reshape"(%6249) <{new_shape = array<i64: 1, 640, 1>}> : (tensor<1x640xf32>) -> tensor<1x640x1xf32>
    %6251 = "tosa.reshape"(%6250) <{new_shape = array<i64: 1, 640, 1, 1>}> : (tensor<1x640x1xf32>) -> tensor<1x640x1x1xf32>
    %6252 = "tosa.mul"(%6245, %6251) <{shift = 0 : i8}> : (tensor<1x640x64x64xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x64x64xf32>
    %6253 = "tosa.add"(%6252, %6248) : (tensor<1x640x64x64xf32>, tensor<1x640x1x1xf32>) -> tensor<1x640x64x64xf32>
    %6254 = "tosa.sigmoid"(%6253) : (tensor<1x640x64x64xf32>) -> tensor<1x640x64x64xf32>
    %6255 = "tosa.mul"(%6253, %6254) <{shift = 0 : i8}> : (tensor<1x640x64x64xf32>, tensor<1x640x64x64xf32>) -> tensor<1x640x64x64xf32>
    %6256 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6257 = "tosa.transpose"(%6255, %6256) : (tensor<1x640x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x640xf32>
    %6258 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6259 = "tosa.transpose"(%arg678, %6258) : (tensor<320x640x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x640xf32>
    %6260 = "tosa.conv2d"(%6257, %6259, %arg679) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x640xf32>, tensor<320x3x3x640xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %6261 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6262 = "tosa.transpose"(%6260, %6261) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %6263 = "tosa.sigmoid"(%50) : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %6264 = "tosa.mul"(%50, %6263) <{shift = 0 : i8}> : (tensor<1x1280xf32>, tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %6265 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6266 = "tosa.transpose"(%arg680, %6265) : (tensor<320x1280xf32>, tensor<2xi32>) -> tensor<1280x320xf32>
    %6267 = "tosa.reshape"(%6264) <{new_shape = array<i64: 1, 1, 1280>}> : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %6268 = "tosa.reshape"(%6266) <{new_shape = array<i64: 1, 1280, 320>}> : (tensor<1280x320xf32>) -> tensor<1x1280x320xf32>
    %6269 = "tosa.matmul"(%6267, %6268) : (tensor<1x1x1280xf32>, tensor<1x1280x320xf32>) -> tensor<1x1x320xf32>
    %6270 = "tosa.reshape"(%6269) <{new_shape = array<i64: 1, 320>}> : (tensor<1x1x320xf32>) -> tensor<1x320xf32>
    %6271 = "tosa.reshape"(%arg681) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6272 = "tosa.add"(%6271, %6270) : (tensor<1x320xf32>, tensor<1x320xf32>) -> tensor<1x320xf32>
    %6273 = "tensor.extract_slice"(%6272) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %6274 = "tensor.extract_slice"(%6273) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 320>, static_strides = array<i64: 1, 1>}> : (tensor<1x320xf32>) -> tensor<1x320xf32>
    %6275 = "tosa.reshape"(%6274) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %6276 = "tosa.reshape"(%6275) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %6277 = "tosa.add"(%6262, %6276) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %6278 = "tosa.reshape"(%6277) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %6279 = "tosa.reduce_sum"(%6278) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %6280 = "tosa.reduce_sum"(%6279) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %6281 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6282 = "tosa.reciprocal"(%6281) : (tensor<1xf32>) -> tensor<1xf32>
    %6283 = "tosa.mul"(%6282, %6280) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6284 = "tosa.sub"(%6278, %6283) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %6285 = "tosa.mul"(%6284, %6284) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %6286 = "tosa.reduce_sum"(%6285) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %6287 = "tosa.reduce_sum"(%6286) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %6288 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6289 = "tosa.reciprocal"(%6288) : (tensor<1xf32>) -> tensor<1xf32>
    %6290 = "tosa.mul"(%6289, %6287) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6291 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %6292 = "tosa.add"(%6290, %6291) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6293 = "tosa.rsqrt"(%6292) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6294 = "tosa.sub"(%6278, %6283) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %6295 = "tosa.mul"(%6294, %6293) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %6296 = "tosa.reshape"(%6295) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %6297 = "tosa.reshape"(%arg682) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6298 = "tosa.reshape"(%6297) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %6299 = "tosa.reshape"(%6298) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %6300 = "tosa.reshape"(%arg683) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6301 = "tosa.reshape"(%6300) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %6302 = "tosa.reshape"(%6301) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %6303 = "tosa.mul"(%6296, %6302) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %6304 = "tosa.add"(%6303, %6299) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %6305 = "tosa.sigmoid"(%6304) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6306 = "tosa.mul"(%6304, %6305) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6307 = "tosa.identity"(%6306) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6308 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6309 = "tosa.transpose"(%6307, %6308) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %6310 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6311 = "tosa.transpose"(%arg684, %6310) : (tensor<320x320x3x3xf32>, tensor<4xi32>) -> tensor<320x3x3x320xf32>
    %6312 = "tosa.conv2d"(%6309, %6311, %arg685) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x320xf32>, tensor<320x3x3x320xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %6313 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6314 = "tosa.transpose"(%6312, %6313) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %6315 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6316 = "tosa.transpose"(%6226, %6315) : (tensor<1x640x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x640xf32>
    %6317 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6318 = "tosa.transpose"(%arg686, %6317) : (tensor<320x640x1x1xf32>, tensor<4xi32>) -> tensor<320x1x1x640xf32>
    %6319 = "tosa.conv2d"(%6316, %6318, %arg687) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x640xf32>, tensor<320x1x1x640xf32>, tensor<320xf32>) -> tensor<1x64x64x320xf32>
    %6320 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6321 = "tosa.transpose"(%6319, %6320) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %6322 = "tosa.add"(%6321, %6314) : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6323 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x320x64x64xf32>}> : () -> tensor<1x320x64x64xf32>
    %6324 = "tosa.reciprocal"(%6323) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6325 = "tosa.mul"(%6322, %6324) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6326 = "tosa.reshape"(%6325) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %6327 = "tosa.reduce_sum"(%6326) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %6328 = "tosa.reduce_sum"(%6327) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %6329 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6330 = "tosa.reciprocal"(%6329) : (tensor<1xf32>) -> tensor<1xf32>
    %6331 = "tosa.mul"(%6330, %6328) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6332 = "tosa.sub"(%6326, %6331) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %6333 = "tosa.mul"(%6332, %6332) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %6334 = "tosa.reduce_sum"(%6333) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %6335 = "tosa.reduce_sum"(%6334) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %6336 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6337 = "tosa.reciprocal"(%6336) : (tensor<1xf32>) -> tensor<1xf32>
    %6338 = "tosa.mul"(%6337, %6335) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6339 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %6340 = "tosa.add"(%6338, %6339) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6341 = "tosa.rsqrt"(%6340) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6342 = "tosa.sub"(%6326, %6331) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %6343 = "tosa.mul"(%6342, %6341) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %6344 = "tosa.reshape"(%6343) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %6345 = "tosa.reshape"(%arg688) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6346 = "tosa.reshape"(%6345) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %6347 = "tosa.reshape"(%6346) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %6348 = "tosa.reshape"(%arg689) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6349 = "tosa.reshape"(%6348) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %6350 = "tosa.reshape"(%6349) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %6351 = "tosa.mul"(%6344, %6350) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %6352 = "tosa.add"(%6351, %6347) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %6353 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6354 = "tosa.transpose"(%6352, %6353) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %6355 = "tosa.reshape"(%6354) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x64x64x320xf32>) -> tensor<1x4096x320xf32>
    %6356 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6357 = "tosa.transpose"(%arg690, %6356) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6358 = "tosa.reshape"(%6355) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6359 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %6360 = "linalg.matmul"(%6358, %6357, %6359) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6361 = "tosa.reshape"(%6360) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6362 = "tosa.reshape"(%arg691) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6363 = "tosa.add"(%6361, %6362) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6364 = "tosa.reduce_sum"(%6363) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %6365 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6366 = "tosa.reciprocal"(%6365) : (tensor<1xf32>) -> tensor<1xf32>
    %6367 = "tosa.mul"(%6366, %6364) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6368 = "tosa.sub"(%6363, %6367) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6369 = "tosa.mul"(%6368, %6368) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6370 = "tosa.reduce_sum"(%6369) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %6371 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6372 = "tosa.reciprocal"(%6371) : (tensor<1xf32>) -> tensor<1xf32>
    %6373 = "tosa.mul"(%6372, %6370) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6374 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %6375 = "tosa.add"(%6373, %6374) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6376 = "tosa.rsqrt"(%6375) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6377 = "tosa.sub"(%6363, %6367) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6378 = "tosa.mul"(%6377, %6376) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6379 = "tosa.reshape"(%arg692) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6380 = "tosa.mul"(%6378, %6379) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6381 = "tosa.reshape"(%arg693) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6382 = "tosa.add"(%6380, %6381) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6383 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6384 = "tosa.transpose"(%arg694, %6383) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6385 = "tosa.reshape"(%6382) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6386 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %6387 = "linalg.matmul"(%6385, %6384, %6386) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6388 = "tosa.reshape"(%6387) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6389 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6390 = "tosa.transpose"(%arg695, %6389) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6391 = "tosa.reshape"(%6382) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6392 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %6393 = "linalg.matmul"(%6391, %6390, %6392) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6394 = "tosa.reshape"(%6393) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6395 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6396 = "tosa.transpose"(%arg696, %6395) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6397 = "tosa.reshape"(%6382) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6398 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %6399 = "linalg.matmul"(%6397, %6396, %6398) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6400 = "tosa.reshape"(%6399) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6401 = "tosa.reshape"(%6388) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %6402 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6403 = "tosa.transpose"(%6401, %6402) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %6404 = "tosa.reshape"(%6394) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %6405 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6406 = "tosa.transpose"(%6404, %6405) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %6407 = "tosa.reshape"(%6400) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %6408 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6409 = "tosa.transpose"(%6407, %6408) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %6410 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x4096xf32>}> : () -> tensor<4096x4096xf32>
    %6411 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6412 = "tosa.transpose"(%6406, %6411) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x5x64x4096xf32>
    %6413 = "tosa.reshape"(%6403) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %6414 = "tosa.reshape"(%6412) <{new_shape = array<i64: 5, 64, 4096>}> : (tensor<1x5x64x4096xf32>) -> tensor<5x64x4096xf32>
    %6415 = "tosa.matmul"(%6413, %6414) : (tensor<5x4096x64xf32>, tensor<5x64x4096xf32>) -> tensor<5x4096x4096xf32>
    %6416 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<5x4096x4096xf32>}> : () -> tensor<5x4096x4096xf32>
    %6417 = "tosa.mul"(%6415, %6416) <{shift = 0 : i8}> : (tensor<5x4096x4096xf32>, tensor<5x4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %6418 = "tosa.add"(%6417, %6410) : (tensor<5x4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %6419 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<5x4096x4096xf32>}> : () -> tensor<5x4096x4096xf32>
    %6420 = "linalg.softmax"(%6418, %6419) <{dimension = 3 : i64}> : (tensor<5x4096x4096xf32>, tensor<5x4096x4096xf32>) -> tensor<5x4096x4096xf32>
    %6421 = "tosa.reshape"(%6409) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %6422 = "tosa.matmul"(%6420, %6421) : (tensor<5x4096x4096xf32>, tensor<5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %6423 = "tosa.reshape"(%6422) <{new_shape = array<i64: 1, 5, 4096, 64>}> : (tensor<5x4096x64xf32>) -> tensor<1x5x4096x64xf32>
    %6424 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6425 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6426 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6427 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6428 = "tosa.transpose"(%6423, %6427) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x4096x5x64xf32>
    %6429 = "tosa.reshape"(%6428) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x4096x5x64xf32>) -> tensor<1x4096x320xf32>
    %6430 = "tosa.reshape"(%6429) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6431 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6432 = "tosa.transpose"(%arg697, %6431) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6433 = "tosa.reshape"(%6430) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6434 = "tosa.reshape"(%6432) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %6435 = "tosa.matmul"(%6433, %6434) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %6436 = "tosa.reshape"(%6435) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6437 = "tosa.reshape"(%arg698) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6438 = "tosa.add"(%6437, %6436) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6439 = "tosa.reshape"(%6438) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6440 = "tosa.identity"(%6439) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6441 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x320xf32>}> : () -> tensor<1x4096x320xf32>
    %6442 = "tosa.reciprocal"(%6441) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6443 = "tosa.mul"(%6440, %6442) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6444 = "tosa.add"(%6443, %6363) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6445 = "tosa.reduce_sum"(%6444) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %6446 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6447 = "tosa.reciprocal"(%6446) : (tensor<1xf32>) -> tensor<1xf32>
    %6448 = "tosa.mul"(%6447, %6445) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6449 = "tosa.sub"(%6444, %6448) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6450 = "tosa.mul"(%6449, %6449) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6451 = "tosa.reduce_sum"(%6450) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %6452 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6453 = "tosa.reciprocal"(%6452) : (tensor<1xf32>) -> tensor<1xf32>
    %6454 = "tosa.mul"(%6453, %6451) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6455 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %6456 = "tosa.add"(%6454, %6455) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6457 = "tosa.rsqrt"(%6456) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6458 = "tosa.sub"(%6444, %6448) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6459 = "tosa.mul"(%6458, %6457) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6460 = "tosa.reshape"(%arg699) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6461 = "tosa.mul"(%6459, %6460) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6462 = "tosa.reshape"(%arg700) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6463 = "tosa.add"(%6461, %6462) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6464 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6465 = "tosa.transpose"(%arg701, %6464) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6466 = "tosa.reshape"(%6463) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6467 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x320xf32>}> : () -> tensor<4096x320xf32>
    %6468 = "linalg.matmul"(%6466, %6465, %6467) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<4096x320xf32>, tensor<320x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6469 = "tosa.reshape"(%6468) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6470 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6471 = "tosa.transpose"(%arg702, %6470) : (tensor<320x1024xf32>, tensor<2xi32>) -> tensor<1024x320xf32>
    %6472 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %6473 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x320xf32>}> : () -> tensor<12x320xf32>
    %6474 = "linalg.matmul"(%6472, %6471, %6473) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x320xf32>, tensor<12x320xf32>) -> tensor<12x320xf32>
    %6475 = "tosa.reshape"(%6474) <{new_shape = array<i64: 1, 12, 320>}> : (tensor<12x320xf32>) -> tensor<1x12x320xf32>
    %6476 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6477 = "tosa.transpose"(%arg704, %6476) : (tensor<320x1024xf32>, tensor<2xi32>) -> tensor<1024x320xf32>
    %6478 = "tosa.reshape"(%arg705) <{new_shape = array<i64: 12, 1024>}> : (tensor<1x12x1024xf32>) -> tensor<12x1024xf32>
    %6479 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<12x320xf32>}> : () -> tensor<12x320xf32>
    %6480 = "linalg.matmul"(%6478, %6477, %6479) <{cast = #linalg.type_fn<cast_signed>, operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg720: f32, %arg721: f32, %arg722: f32):
      %6629 = "arith.mulf"(%arg720, %arg721) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6630 = "arith.addf"(%arg722, %6629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%6630) : (f32) -> ()
    }) : (tensor<12x1024xf32>, tensor<1024x320xf32>, tensor<12x320xf32>) -> tensor<12x320xf32>
    %6481 = "tosa.reshape"(%6480) <{new_shape = array<i64: 1, 12, 320>}> : (tensor<12x320xf32>) -> tensor<1x12x320xf32>
    %6482 = "tosa.reshape"(%6469) <{new_shape = array<i64: 1, 4096, 5, 64>}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x5x64xf32>
    %6483 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6484 = "tosa.transpose"(%6482, %6483) : (tensor<1x4096x5x64xf32>, tensor<4xi32>) -> tensor<1x5x4096x64xf32>
    %6485 = "tosa.reshape"(%6475) <{new_shape = array<i64: 1, 12, 5, 64>}> : (tensor<1x12x320xf32>) -> tensor<1x12x5x64xf32>
    %6486 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6487 = "tosa.transpose"(%6485, %6486) : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %6488 = "tosa.reshape"(%6481) <{new_shape = array<i64: 1, 12, 5, 64>}> : (tensor<1x12x320xf32>) -> tensor<1x12x5x64xf32>
    %6489 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6490 = "tosa.transpose"(%6488, %6489) : (tensor<1x12x5x64xf32>, tensor<4xi32>) -> tensor<1x5x12x64xf32>
    %6491 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4096x12xf32>}> : () -> tensor<4096x12xf32>
    %6492 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6493 = "tosa.transpose"(%6487, %6492) : (tensor<1x5x12x64xf32>, tensor<4xi32>) -> tensor<1x5x64x12xf32>
    %6494 = "tosa.reshape"(%6484) <{new_shape = array<i64: 5, 4096, 64>}> : (tensor<1x5x4096x64xf32>) -> tensor<5x4096x64xf32>
    %6495 = "tosa.reshape"(%6493) <{new_shape = array<i64: 5, 64, 12>}> : (tensor<1x5x64x12xf32>) -> tensor<5x64x12xf32>
    %6496 = "tosa.matmul"(%6494, %6495) : (tensor<5x4096x64xf32>, tensor<5x64x12xf32>) -> tensor<5x4096x12xf32>
    %6497 = "arith.constant"() <{value = dense<1.250000e-01> : tensor<5x4096x12xf32>}> : () -> tensor<5x4096x12xf32>
    %6498 = "tosa.mul"(%6496, %6497) <{shift = 0 : i8}> : (tensor<5x4096x12xf32>, tensor<5x4096x12xf32>) -> tensor<5x4096x12xf32>
    %6499 = "tosa.add"(%6498, %6491) : (tensor<5x4096x12xf32>, tensor<4096x12xf32>) -> tensor<5x4096x12xf32>
    %6500 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<5x4096x12xf32>}> : () -> tensor<5x4096x12xf32>
    %6501 = "linalg.softmax"(%6499, %6500) <{dimension = 3 : i64}> : (tensor<5x4096x12xf32>, tensor<5x4096x12xf32>) -> tensor<5x4096x12xf32>
    %6502 = "tosa.reshape"(%6490) <{new_shape = array<i64: 5, 12, 64>}> : (tensor<1x5x12x64xf32>) -> tensor<5x12x64xf32>
    %6503 = "tosa.matmul"(%6501, %6502) : (tensor<5x4096x12xf32>, tensor<5x12x64xf32>) -> tensor<5x4096x64xf32>
    %6504 = "tosa.reshape"(%6503) <{new_shape = array<i64: 1, 5, 4096, 64>}> : (tensor<5x4096x64xf32>) -> tensor<1x5x4096x64xf32>
    %6505 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6506 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6507 = "tosa.const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
    %6508 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6509 = "tosa.transpose"(%6504, %6508) : (tensor<1x5x4096x64xf32>, tensor<4xi32>) -> tensor<1x4096x5x64xf32>
    %6510 = "tosa.reshape"(%6509) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<1x4096x5x64xf32>) -> tensor<1x4096x320xf32>
    %6511 = "tosa.reshape"(%6510) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6512 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6513 = "tosa.transpose"(%arg706, %6512) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6514 = "tosa.reshape"(%6511) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6515 = "tosa.reshape"(%6513) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %6516 = "tosa.matmul"(%6514, %6515) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %6517 = "tosa.reshape"(%6516) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6518 = "tosa.reshape"(%arg707) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6519 = "tosa.add"(%6518, %6517) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6520 = "tosa.reshape"(%6519) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6521 = "tosa.identity"(%6520) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6522 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x320xf32>}> : () -> tensor<1x4096x320xf32>
    %6523 = "tosa.reciprocal"(%6522) : (tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6524 = "tosa.mul"(%6521, %6523) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6525 = "tosa.add"(%6524, %6444) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6526 = "tosa.reduce_sum"(%6525) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %6527 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6528 = "tosa.reciprocal"(%6527) : (tensor<1xf32>) -> tensor<1xf32>
    %6529 = "tosa.mul"(%6528, %6526) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6530 = "tosa.sub"(%6525, %6529) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6531 = "tosa.mul"(%6530, %6530) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6532 = "tosa.reduce_sum"(%6531) <{axis = 2 : i32}> : (tensor<1x4096x320xf32>) -> tensor<1x4096x1xf32>
    %6533 = "tosa.const"() <{value = dense<3.200000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6534 = "tosa.reciprocal"(%6533) : (tensor<1xf32>) -> tensor<1xf32>
    %6535 = "tosa.mul"(%6534, %6532) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6536 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x4096x1xf32>}> : () -> tensor<1x4096x1xf32>
    %6537 = "tosa.add"(%6535, %6536) : (tensor<1x4096x1xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6538 = "tosa.rsqrt"(%6537) : (tensor<1x4096x1xf32>) -> tensor<1x4096x1xf32>
    %6539 = "tosa.sub"(%6525, %6529) : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6540 = "tosa.mul"(%6539, %6538) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x4096x1xf32>) -> tensor<1x4096x320xf32>
    %6541 = "tosa.reshape"(%arg708) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6542 = "tosa.mul"(%6540, %6541) <{shift = 0 : i8}> : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6543 = "tosa.reshape"(%arg709) <{new_shape = array<i64: 1, 1, 320>}> : (tensor<320xf32>) -> tensor<1x1x320xf32>
    %6544 = "tosa.add"(%6542, %6543) : (tensor<1x4096x320xf32>, tensor<1x1x320xf32>) -> tensor<1x4096x320xf32>
    %6545 = "tosa.reshape"(%6544) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6546 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6547 = "tosa.transpose"(%arg710, %6546) : (tensor<2560x320xf32>, tensor<2xi32>) -> tensor<320x2560xf32>
    %6548 = "tosa.reshape"(%6545) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6549 = "tosa.reshape"(%6547) <{new_shape = array<i64: 1, 320, 2560>}> : (tensor<320x2560xf32>) -> tensor<1x320x2560xf32>
    %6550 = "tosa.matmul"(%6548, %6549) : (tensor<1x4096x320xf32>, tensor<1x320x2560xf32>) -> tensor<1x4096x2560xf32>
    %6551 = "tosa.reshape"(%6550) <{new_shape = array<i64: 4096, 2560>}> : (tensor<1x4096x2560xf32>) -> tensor<4096x2560xf32>
    %6552 = "tosa.reshape"(%arg711) <{new_shape = array<i64: 1, 2560>}> : (tensor<2560xf32>) -> tensor<1x2560xf32>
    %6553 = "tosa.add"(%6552, %6551) : (tensor<1x2560xf32>, tensor<4096x2560xf32>) -> tensor<4096x2560xf32>
    %6554 = "tosa.reshape"(%6553) <{new_shape = array<i64: 1, 4096, 2560>}> : (tensor<4096x2560xf32>) -> tensor<1x4096x2560xf32>
    %6555 = "tosa.slice"(%6554) <{size = array<i64: 0, 0, 1280>, start = array<i64: 0, 0, 0>}> : (tensor<1x4096x2560xf32>) -> tensor<1x4096x1280xf32>
    %6556 = "tosa.slice"(%6554) <{size = array<i64: 0, 0, 2560>, start = array<i64: 0, 0, 1280>}> : (tensor<1x4096x2560xf32>) -> tensor<1x4096x1280xf32>
    %6557 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %6558 = "tosa.mul"(%6556, %6557) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6559 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %6560 = "tosa.mul"(%6556, %6559) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6561 = "math.erf"(%6560) <{fastmath = #arith.fastmath<none>}> : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6562 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x4096x1280xf32>}> : () -> tensor<1x4096x1280xf32>
    %6563 = "tosa.add"(%6561, %6562) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6564 = "tosa.mul"(%6558, %6563) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6565 = "tosa.mul"(%6555, %6564) <{shift = 0 : i8}> : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6566 = "tosa.identity"(%6565) : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6567 = "tosa.reshape"(%6566) <{new_shape = array<i64: 4096, 1280>}> : (tensor<1x4096x1280xf32>) -> tensor<4096x1280xf32>
    %6568 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6569 = "tosa.transpose"(%arg712, %6568) : (tensor<320x1280xf32>, tensor<2xi32>) -> tensor<1280x320xf32>
    %6570 = "tosa.reshape"(%6567) <{new_shape = array<i64: 1, 4096, 1280>}> : (tensor<4096x1280xf32>) -> tensor<1x4096x1280xf32>
    %6571 = "tosa.reshape"(%6569) <{new_shape = array<i64: 1, 1280, 320>}> : (tensor<1280x320xf32>) -> tensor<1x1280x320xf32>
    %6572 = "tosa.matmul"(%6570, %6571) : (tensor<1x4096x1280xf32>, tensor<1x1280x320xf32>) -> tensor<1x4096x320xf32>
    %6573 = "tosa.reshape"(%6572) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6574 = "tosa.reshape"(%arg713) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6575 = "tosa.add"(%6574, %6573) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6576 = "tosa.reshape"(%6575) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6577 = "tosa.add"(%6576, %6525) : (tensor<1x4096x320xf32>, tensor<1x4096x320xf32>) -> tensor<1x4096x320xf32>
    %6578 = "tosa.reshape"(%6577) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6579 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6580 = "tosa.transpose"(%arg714, %6579) : (tensor<320x320xf32>, tensor<2xi32>) -> tensor<320x320xf32>
    %6581 = "tosa.reshape"(%6578) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6582 = "tosa.reshape"(%6580) <{new_shape = array<i64: 1, 320, 320>}> : (tensor<320x320xf32>) -> tensor<1x320x320xf32>
    %6583 = "tosa.matmul"(%6581, %6582) : (tensor<1x4096x320xf32>, tensor<1x320x320xf32>) -> tensor<1x4096x320xf32>
    %6584 = "tosa.reshape"(%6583) <{new_shape = array<i64: 4096, 320>}> : (tensor<1x4096x320xf32>) -> tensor<4096x320xf32>
    %6585 = "tosa.reshape"(%arg715) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6586 = "tosa.add"(%6585, %6584) : (tensor<1x320xf32>, tensor<4096x320xf32>) -> tensor<4096x320xf32>
    %6587 = "tosa.reshape"(%6586) <{new_shape = array<i64: 1, 4096, 320>}> : (tensor<4096x320xf32>) -> tensor<1x4096x320xf32>
    %6588 = "tosa.reshape"(%6587) <{new_shape = array<i64: 1, 64, 64, 320>}> : (tensor<1x4096x320xf32>) -> tensor<1x64x64x320xf32>
    %6589 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6590 = "tosa.transpose"(%6588, %6589) : (tensor<1x64x64x320xf32>, tensor<4xi32>) -> tensor<1x320x64x64xf32>
    %6591 = "tosa.identity"(%6590) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6592 = "tosa.add"(%6591, %6325) : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6593 = "tosa.reshape"(%6592) <{new_shape = array<i64: 1, 32, 10, 4096>}> : (tensor<1x320x64x64xf32>) -> tensor<1x32x10x4096xf32>
    %6594 = "tosa.reduce_sum"(%6593) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %6595 = "tosa.reduce_sum"(%6594) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %6596 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6597 = "tosa.reciprocal"(%6596) : (tensor<1xf32>) -> tensor<1xf32>
    %6598 = "tosa.mul"(%6597, %6595) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6599 = "tosa.sub"(%6593, %6598) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %6600 = "tosa.mul"(%6599, %6599) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x10x4096xf32>) -> tensor<1x32x10x4096xf32>
    %6601 = "tosa.reduce_sum"(%6600) <{axis = 2 : i32}> : (tensor<1x32x10x4096xf32>) -> tensor<1x32x1x4096xf32>
    %6602 = "tosa.reduce_sum"(%6601) <{axis = 3 : i32}> : (tensor<1x32x1x4096xf32>) -> tensor<1x32x1x1xf32>
    %6603 = "tosa.const"() <{value = dense<4.096000e+04> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6604 = "tosa.reciprocal"(%6603) : (tensor<1xf32>) -> tensor<1xf32>
    %6605 = "tosa.mul"(%6604, %6602) <{shift = 0 : i8}> : (tensor<1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6606 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x32x1x1xf32>}> : () -> tensor<1x32x1x1xf32>
    %6607 = "tosa.add"(%6605, %6606) : (tensor<1x32x1x1xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6608 = "tosa.rsqrt"(%6607) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %6609 = "tosa.sub"(%6593, %6598) : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %6610 = "tosa.mul"(%6609, %6608) <{shift = 0 : i8}> : (tensor<1x32x10x4096xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x10x4096xf32>
    %6611 = "tosa.reshape"(%6610) <{new_shape = array<i64: 1, 320, 64, 64>}> : (tensor<1x32x10x4096xf32>) -> tensor<1x320x64x64xf32>
    %6612 = "tosa.reshape"(%arg716) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6613 = "tosa.reshape"(%6612) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %6614 = "tosa.reshape"(%6613) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %6615 = "tosa.reshape"(%arg717) <{new_shape = array<i64: 1, 320>}> : (tensor<320xf32>) -> tensor<1x320xf32>
    %6616 = "tosa.reshape"(%6615) <{new_shape = array<i64: 1, 320, 1>}> : (tensor<1x320xf32>) -> tensor<1x320x1xf32>
    %6617 = "tosa.reshape"(%6616) <{new_shape = array<i64: 1, 320, 1, 1>}> : (tensor<1x320x1xf32>) -> tensor<1x320x1x1xf32>
    %6618 = "tosa.mul"(%6611, %6617) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %6619 = "tosa.add"(%6618, %6614) : (tensor<1x320x64x64xf32>, tensor<1x320x1x1xf32>) -> tensor<1x320x64x64xf32>
    %6620 = "tosa.sigmoid"(%6619) : (tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6621 = "tosa.mul"(%6619, %6620) <{shift = 0 : i8}> : (tensor<1x320x64x64xf32>, tensor<1x320x64x64xf32>) -> tensor<1x320x64x64xf32>
    %6622 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6623 = "tosa.transpose"(%6621, %6622) : (tensor<1x320x64x64xf32>, tensor<4xi32>) -> tensor<1x64x64x320xf32>
    %6624 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6625 = "tosa.transpose"(%arg718, %6624) : (tensor<4x320x3x3xf32>, tensor<4xi32>) -> tensor<4x3x3x320xf32>
    %6626 = "tosa.conv2d"(%6623, %6625, %arg719) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x64x64x320xf32>, tensor<4x3x3x320xf32>, tensor<4xf32>) -> tensor<1x64x64x4xf32>
    %6627 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6628 = "tosa.transpose"(%6626, %6627) : (tensor<1x64x64x4xf32>, tensor<4xi32>) -> tensor<1x4x64x64xf32>
    "func.return"(%6628) : (tensor<1x4x64x64xf32>) -> ()
  }) : () -> ()
}) : () -> ()

