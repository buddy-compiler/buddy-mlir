#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map8 = affine_map<(d0, d1) -> (d1, d0)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map10 = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
#map11 = affine_map<(d0, d1) -> (0, d0, d1)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @forward(%arg0: tensor<6755192832xbf16>, %arg1: tensor<1x40xi64>) -> (tensor<1x40x32000xf32>, tensor<1x40x4096xbf16>) {
    %extracted_slice = tensor.extract_slice %arg0[0] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_0 = tensor.extract_slice %arg0[4096] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_1 = tensor.extract_slice %arg0[8192] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_2 = tensor.extract_slice %arg0[12288] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_3 = tensor.extract_slice %arg0[16384] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_4 = tensor.extract_slice %arg0[20480] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_5 = tensor.extract_slice %arg0[24576] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_6 = tensor.extract_slice %arg0[28672] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_7 = tensor.extract_slice %arg0[32768] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_8 = tensor.extract_slice %arg0[36864] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_9 = tensor.extract_slice %arg0[40960] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_10 = tensor.extract_slice %arg0[45056] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_11 = tensor.extract_slice %arg0[49152] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_12 = tensor.extract_slice %arg0[53248] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_13 = tensor.extract_slice %arg0[57344] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_14 = tensor.extract_slice %arg0[61440] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_15 = tensor.extract_slice %arg0[65536] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_16 = tensor.extract_slice %arg0[69632] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_17 = tensor.extract_slice %arg0[73728] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_18 = tensor.extract_slice %arg0[77824] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_19 = tensor.extract_slice %arg0[81920] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_20 = tensor.extract_slice %arg0[86016] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_21 = tensor.extract_slice %arg0[90112] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_22 = tensor.extract_slice %arg0[94208] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_23 = tensor.extract_slice %arg0[98304] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_24 = tensor.extract_slice %arg0[102400] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_25 = tensor.extract_slice %arg0[106496] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_26 = tensor.extract_slice %arg0[110592] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_27 = tensor.extract_slice %arg0[114688] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_28 = tensor.extract_slice %arg0[118784] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_29 = tensor.extract_slice %arg0[122880] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_30 = tensor.extract_slice %arg0[126976] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_31 = tensor.extract_slice %arg0[131072] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_32 = tensor.extract_slice %arg0[135168] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_33 = tensor.extract_slice %arg0[139264] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_34 = tensor.extract_slice %arg0[143360] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_35 = tensor.extract_slice %arg0[147456] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_36 = tensor.extract_slice %arg0[151552] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_37 = tensor.extract_slice %arg0[155648] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_38 = tensor.extract_slice %arg0[159744] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_39 = tensor.extract_slice %arg0[163840] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_40 = tensor.extract_slice %arg0[167936] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_41 = tensor.extract_slice %arg0[172032] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_42 = tensor.extract_slice %arg0[176128] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_43 = tensor.extract_slice %arg0[180224] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_44 = tensor.extract_slice %arg0[184320] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_45 = tensor.extract_slice %arg0[188416] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_46 = tensor.extract_slice %arg0[192512] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_47 = tensor.extract_slice %arg0[196608] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_48 = tensor.extract_slice %arg0[200704] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_49 = tensor.extract_slice %arg0[204800] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_50 = tensor.extract_slice %arg0[208896] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_51 = tensor.extract_slice %arg0[212992] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_52 = tensor.extract_slice %arg0[217088] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_53 = tensor.extract_slice %arg0[221184] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_54 = tensor.extract_slice %arg0[225280] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_55 = tensor.extract_slice %arg0[229376] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_56 = tensor.extract_slice %arg0[233472] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_57 = tensor.extract_slice %arg0[237568] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_58 = tensor.extract_slice %arg0[241664] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_59 = tensor.extract_slice %arg0[245760] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_60 = tensor.extract_slice %arg0[249856] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_61 = tensor.extract_slice %arg0[253952] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_62 = tensor.extract_slice %arg0[258048] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_63 = tensor.extract_slice %arg0[262144] [4096] [1] : tensor<6755192832xbf16> to tensor<4096xbf16>
    %extracted_slice_64 = tensor.extract_slice %arg0[266240] [131072000] [1] : tensor<6755192832xbf16> to tensor<131072000xbf16>
    %expanded = tensor.expand_shape %extracted_slice_64 [[0, 1]] : tensor<131072000xbf16> into tensor<32000x4096xbf16>
    %extracted_slice_65 = tensor.extract_slice %arg0[131338240] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_66 = tensor.expand_shape %extracted_slice_65 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_67 = tensor.extract_slice %arg0[148115456] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_68 = tensor.expand_shape %extracted_slice_67 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_69 = tensor.extract_slice %arg0[164892672] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_70 = tensor.expand_shape %extracted_slice_69 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_71 = tensor.extract_slice %arg0[181669888] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_72 = tensor.expand_shape %extracted_slice_71 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_73 = tensor.extract_slice %arg0[198447104] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_74 = tensor.expand_shape %extracted_slice_73 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_75 = tensor.extract_slice %arg0[243535872] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_76 = tensor.expand_shape %extracted_slice_75 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_77 = tensor.extract_slice %arg0[288624640] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_78 = tensor.expand_shape %extracted_slice_77 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_79 = tensor.extract_slice %arg0[333713408] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_80 = tensor.expand_shape %extracted_slice_79 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_81 = tensor.extract_slice %arg0[350490624] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_82 = tensor.expand_shape %extracted_slice_81 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_83 = tensor.extract_slice %arg0[367267840] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_84 = tensor.expand_shape %extracted_slice_83 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_85 = tensor.extract_slice %arg0[384045056] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_86 = tensor.expand_shape %extracted_slice_85 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_87 = tensor.extract_slice %arg0[400822272] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_88 = tensor.expand_shape %extracted_slice_87 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_89 = tensor.extract_slice %arg0[445911040] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_90 = tensor.expand_shape %extracted_slice_89 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_91 = tensor.extract_slice %arg0[490999808] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_92 = tensor.expand_shape %extracted_slice_91 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_93 = tensor.extract_slice %arg0[536088576] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_94 = tensor.expand_shape %extracted_slice_93 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_95 = tensor.extract_slice %arg0[552865792] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_96 = tensor.expand_shape %extracted_slice_95 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_97 = tensor.extract_slice %arg0[569643008] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_98 = tensor.expand_shape %extracted_slice_97 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_99 = tensor.extract_slice %arg0[586420224] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_100 = tensor.expand_shape %extracted_slice_99 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_101 = tensor.extract_slice %arg0[603197440] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_102 = tensor.expand_shape %extracted_slice_101 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_103 = tensor.extract_slice %arg0[648286208] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_104 = tensor.expand_shape %extracted_slice_103 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_105 = tensor.extract_slice %arg0[693374976] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_106 = tensor.expand_shape %extracted_slice_105 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_107 = tensor.extract_slice %arg0[738463744] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_108 = tensor.expand_shape %extracted_slice_107 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_109 = tensor.extract_slice %arg0[755240960] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_110 = tensor.expand_shape %extracted_slice_109 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_111 = tensor.extract_slice %arg0[772018176] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_112 = tensor.expand_shape %extracted_slice_111 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_113 = tensor.extract_slice %arg0[788795392] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_114 = tensor.expand_shape %extracted_slice_113 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_115 = tensor.extract_slice %arg0[805572608] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_116 = tensor.expand_shape %extracted_slice_115 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_117 = tensor.extract_slice %arg0[850661376] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_118 = tensor.expand_shape %extracted_slice_117 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_119 = tensor.extract_slice %arg0[895750144] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_120 = tensor.expand_shape %extracted_slice_119 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_121 = tensor.extract_slice %arg0[940838912] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_122 = tensor.expand_shape %extracted_slice_121 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_123 = tensor.extract_slice %arg0[957616128] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_124 = tensor.expand_shape %extracted_slice_123 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_125 = tensor.extract_slice %arg0[974393344] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_126 = tensor.expand_shape %extracted_slice_125 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_127 = tensor.extract_slice %arg0[991170560] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_128 = tensor.expand_shape %extracted_slice_127 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_129 = tensor.extract_slice %arg0[1007947776] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_130 = tensor.expand_shape %extracted_slice_129 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_131 = tensor.extract_slice %arg0[1053036544] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_132 = tensor.expand_shape %extracted_slice_131 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_133 = tensor.extract_slice %arg0[1098125312] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_134 = tensor.expand_shape %extracted_slice_133 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_135 = tensor.extract_slice %arg0[1143214080] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_136 = tensor.expand_shape %extracted_slice_135 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_137 = tensor.extract_slice %arg0[1159991296] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_138 = tensor.expand_shape %extracted_slice_137 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_139 = tensor.extract_slice %arg0[1176768512] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_140 = tensor.expand_shape %extracted_slice_139 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_141 = tensor.extract_slice %arg0[1193545728] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_142 = tensor.expand_shape %extracted_slice_141 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_143 = tensor.extract_slice %arg0[1210322944] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_144 = tensor.expand_shape %extracted_slice_143 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_145 = tensor.extract_slice %arg0[1255411712] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_146 = tensor.expand_shape %extracted_slice_145 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_147 = tensor.extract_slice %arg0[1300500480] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_148 = tensor.expand_shape %extracted_slice_147 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_149 = tensor.extract_slice %arg0[1345589248] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_150 = tensor.expand_shape %extracted_slice_149 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_151 = tensor.extract_slice %arg0[1362366464] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_152 = tensor.expand_shape %extracted_slice_151 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_153 = tensor.extract_slice %arg0[1379143680] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_154 = tensor.expand_shape %extracted_slice_153 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_155 = tensor.extract_slice %arg0[1395920896] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_156 = tensor.expand_shape %extracted_slice_155 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_157 = tensor.extract_slice %arg0[1412698112] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_158 = tensor.expand_shape %extracted_slice_157 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_159 = tensor.extract_slice %arg0[1457786880] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_160 = tensor.expand_shape %extracted_slice_159 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_161 = tensor.extract_slice %arg0[1502875648] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_162 = tensor.expand_shape %extracted_slice_161 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_163 = tensor.extract_slice %arg0[1547964416] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_164 = tensor.expand_shape %extracted_slice_163 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_165 = tensor.extract_slice %arg0[1564741632] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_166 = tensor.expand_shape %extracted_slice_165 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_167 = tensor.extract_slice %arg0[1581518848] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_168 = tensor.expand_shape %extracted_slice_167 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_169 = tensor.extract_slice %arg0[1598296064] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_170 = tensor.expand_shape %extracted_slice_169 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_171 = tensor.extract_slice %arg0[1615073280] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_172 = tensor.expand_shape %extracted_slice_171 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_173 = tensor.extract_slice %arg0[1660162048] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_174 = tensor.expand_shape %extracted_slice_173 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_175 = tensor.extract_slice %arg0[1705250816] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_176 = tensor.expand_shape %extracted_slice_175 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_177 = tensor.extract_slice %arg0[1750339584] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_178 = tensor.expand_shape %extracted_slice_177 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_179 = tensor.extract_slice %arg0[1767116800] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_180 = tensor.expand_shape %extracted_slice_179 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_181 = tensor.extract_slice %arg0[1783894016] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_182 = tensor.expand_shape %extracted_slice_181 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_183 = tensor.extract_slice %arg0[1800671232] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_184 = tensor.expand_shape %extracted_slice_183 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_185 = tensor.extract_slice %arg0[1817448448] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_186 = tensor.expand_shape %extracted_slice_185 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_187 = tensor.extract_slice %arg0[1862537216] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_188 = tensor.expand_shape %extracted_slice_187 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_189 = tensor.extract_slice %arg0[1907625984] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_190 = tensor.expand_shape %extracted_slice_189 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_191 = tensor.extract_slice %arg0[1952714752] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_192 = tensor.expand_shape %extracted_slice_191 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_193 = tensor.extract_slice %arg0[1969491968] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_194 = tensor.expand_shape %extracted_slice_193 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_195 = tensor.extract_slice %arg0[1986269184] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_196 = tensor.expand_shape %extracted_slice_195 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_197 = tensor.extract_slice %arg0[2003046400] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_198 = tensor.expand_shape %extracted_slice_197 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_199 = tensor.extract_slice %arg0[2019823616] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_200 = tensor.expand_shape %extracted_slice_199 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_201 = tensor.extract_slice %arg0[2064912384] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_202 = tensor.expand_shape %extracted_slice_201 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_203 = tensor.extract_slice %arg0[2110001152] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_204 = tensor.expand_shape %extracted_slice_203 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_205 = tensor.extract_slice %arg0[2155089920] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_206 = tensor.expand_shape %extracted_slice_205 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_207 = tensor.extract_slice %arg0[2171867136] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_208 = tensor.expand_shape %extracted_slice_207 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_209 = tensor.extract_slice %arg0[2188644352] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_210 = tensor.expand_shape %extracted_slice_209 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_211 = tensor.extract_slice %arg0[2205421568] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_212 = tensor.expand_shape %extracted_slice_211 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_213 = tensor.extract_slice %arg0[2222198784] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_214 = tensor.expand_shape %extracted_slice_213 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_215 = tensor.extract_slice %arg0[2267287552] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_216 = tensor.expand_shape %extracted_slice_215 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_217 = tensor.extract_slice %arg0[2312376320] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_218 = tensor.expand_shape %extracted_slice_217 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_219 = tensor.extract_slice %arg0[2357465088] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_220 = tensor.expand_shape %extracted_slice_219 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_221 = tensor.extract_slice %arg0[2374242304] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_222 = tensor.expand_shape %extracted_slice_221 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_223 = tensor.extract_slice %arg0[2391019520] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_224 = tensor.expand_shape %extracted_slice_223 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_225 = tensor.extract_slice %arg0[2407796736] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_226 = tensor.expand_shape %extracted_slice_225 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_227 = tensor.extract_slice %arg0[2424573952] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_228 = tensor.expand_shape %extracted_slice_227 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_229 = tensor.extract_slice %arg0[2469662720] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_230 = tensor.expand_shape %extracted_slice_229 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_231 = tensor.extract_slice %arg0[2514751488] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_232 = tensor.expand_shape %extracted_slice_231 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_233 = tensor.extract_slice %arg0[2559840256] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_234 = tensor.expand_shape %extracted_slice_233 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_235 = tensor.extract_slice %arg0[2576617472] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_236 = tensor.expand_shape %extracted_slice_235 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_237 = tensor.extract_slice %arg0[2593394688] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_238 = tensor.expand_shape %extracted_slice_237 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_239 = tensor.extract_slice %arg0[2610171904] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_240 = tensor.expand_shape %extracted_slice_239 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_241 = tensor.extract_slice %arg0[2626949120] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_242 = tensor.expand_shape %extracted_slice_241 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_243 = tensor.extract_slice %arg0[2672037888] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_244 = tensor.expand_shape %extracted_slice_243 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_245 = tensor.extract_slice %arg0[2717126656] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_246 = tensor.expand_shape %extracted_slice_245 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_247 = tensor.extract_slice %arg0[2762215424] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_248 = tensor.expand_shape %extracted_slice_247 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_249 = tensor.extract_slice %arg0[2778992640] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_250 = tensor.expand_shape %extracted_slice_249 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_251 = tensor.extract_slice %arg0[2795769856] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_252 = tensor.expand_shape %extracted_slice_251 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_253 = tensor.extract_slice %arg0[2812547072] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_254 = tensor.expand_shape %extracted_slice_253 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_255 = tensor.extract_slice %arg0[2829324288] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_256 = tensor.expand_shape %extracted_slice_255 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_257 = tensor.extract_slice %arg0[2874413056] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_258 = tensor.expand_shape %extracted_slice_257 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_259 = tensor.extract_slice %arg0[2919501824] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_260 = tensor.expand_shape %extracted_slice_259 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_261 = tensor.extract_slice %arg0[2964590592] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_262 = tensor.expand_shape %extracted_slice_261 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_263 = tensor.extract_slice %arg0[2981367808] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_264 = tensor.expand_shape %extracted_slice_263 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_265 = tensor.extract_slice %arg0[2998145024] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_266 = tensor.expand_shape %extracted_slice_265 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_267 = tensor.extract_slice %arg0[3014922240] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_268 = tensor.expand_shape %extracted_slice_267 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_269 = tensor.extract_slice %arg0[3031699456] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_270 = tensor.expand_shape %extracted_slice_269 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_271 = tensor.extract_slice %arg0[3076788224] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_272 = tensor.expand_shape %extracted_slice_271 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_273 = tensor.extract_slice %arg0[3121876992] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_274 = tensor.expand_shape %extracted_slice_273 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_275 = tensor.extract_slice %arg0[3166965760] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_276 = tensor.expand_shape %extracted_slice_275 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_277 = tensor.extract_slice %arg0[3183742976] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_278 = tensor.expand_shape %extracted_slice_277 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_279 = tensor.extract_slice %arg0[3200520192] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_280 = tensor.expand_shape %extracted_slice_279 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_281 = tensor.extract_slice %arg0[3217297408] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_282 = tensor.expand_shape %extracted_slice_281 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_283 = tensor.extract_slice %arg0[3234074624] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_284 = tensor.expand_shape %extracted_slice_283 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_285 = tensor.extract_slice %arg0[3279163392] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_286 = tensor.expand_shape %extracted_slice_285 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_287 = tensor.extract_slice %arg0[3324252160] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_288 = tensor.expand_shape %extracted_slice_287 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_289 = tensor.extract_slice %arg0[3369340928] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_290 = tensor.expand_shape %extracted_slice_289 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_291 = tensor.extract_slice %arg0[3386118144] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_292 = tensor.expand_shape %extracted_slice_291 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_293 = tensor.extract_slice %arg0[3402895360] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_294 = tensor.expand_shape %extracted_slice_293 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_295 = tensor.extract_slice %arg0[3419672576] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_296 = tensor.expand_shape %extracted_slice_295 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_297 = tensor.extract_slice %arg0[3436449792] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_298 = tensor.expand_shape %extracted_slice_297 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_299 = tensor.extract_slice %arg0[3481538560] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_300 = tensor.expand_shape %extracted_slice_299 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_301 = tensor.extract_slice %arg0[3526627328] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_302 = tensor.expand_shape %extracted_slice_301 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_303 = tensor.extract_slice %arg0[3571716096] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_304 = tensor.expand_shape %extracted_slice_303 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_305 = tensor.extract_slice %arg0[3588493312] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_306 = tensor.expand_shape %extracted_slice_305 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_307 = tensor.extract_slice %arg0[3605270528] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_308 = tensor.expand_shape %extracted_slice_307 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_309 = tensor.extract_slice %arg0[3622047744] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_310 = tensor.expand_shape %extracted_slice_309 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_311 = tensor.extract_slice %arg0[3638824960] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_312 = tensor.expand_shape %extracted_slice_311 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_313 = tensor.extract_slice %arg0[3683913728] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_314 = tensor.expand_shape %extracted_slice_313 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_315 = tensor.extract_slice %arg0[3729002496] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_316 = tensor.expand_shape %extracted_slice_315 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_317 = tensor.extract_slice %arg0[3774091264] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_318 = tensor.expand_shape %extracted_slice_317 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_319 = tensor.extract_slice %arg0[3790868480] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_320 = tensor.expand_shape %extracted_slice_319 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_321 = tensor.extract_slice %arg0[3807645696] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_322 = tensor.expand_shape %extracted_slice_321 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_323 = tensor.extract_slice %arg0[3824422912] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_324 = tensor.expand_shape %extracted_slice_323 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_325 = tensor.extract_slice %arg0[3841200128] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_326 = tensor.expand_shape %extracted_slice_325 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_327 = tensor.extract_slice %arg0[3886288896] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_328 = tensor.expand_shape %extracted_slice_327 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_329 = tensor.extract_slice %arg0[3931377664] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_330 = tensor.expand_shape %extracted_slice_329 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_331 = tensor.extract_slice %arg0[3976466432] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_332 = tensor.expand_shape %extracted_slice_331 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_333 = tensor.extract_slice %arg0[3993243648] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_334 = tensor.expand_shape %extracted_slice_333 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_335 = tensor.extract_slice %arg0[4010020864] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_336 = tensor.expand_shape %extracted_slice_335 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_337 = tensor.extract_slice %arg0[4026798080] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_338 = tensor.expand_shape %extracted_slice_337 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_339 = tensor.extract_slice %arg0[4043575296] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_340 = tensor.expand_shape %extracted_slice_339 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_341 = tensor.extract_slice %arg0[4088664064] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_342 = tensor.expand_shape %extracted_slice_341 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_343 = tensor.extract_slice %arg0[4133752832] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_344 = tensor.expand_shape %extracted_slice_343 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_345 = tensor.extract_slice %arg0[4178841600] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_346 = tensor.expand_shape %extracted_slice_345 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_347 = tensor.extract_slice %arg0[4195618816] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_348 = tensor.expand_shape %extracted_slice_347 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_349 = tensor.extract_slice %arg0[4212396032] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_350 = tensor.expand_shape %extracted_slice_349 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_351 = tensor.extract_slice %arg0[4229173248] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_352 = tensor.expand_shape %extracted_slice_351 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_353 = tensor.extract_slice %arg0[4245950464] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_354 = tensor.expand_shape %extracted_slice_353 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_355 = tensor.extract_slice %arg0[4291039232] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_356 = tensor.expand_shape %extracted_slice_355 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_357 = tensor.extract_slice %arg0[4336128000] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_358 = tensor.expand_shape %extracted_slice_357 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_359 = tensor.extract_slice %arg0[4381216768] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_360 = tensor.expand_shape %extracted_slice_359 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_361 = tensor.extract_slice %arg0[4397993984] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_362 = tensor.expand_shape %extracted_slice_361 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_363 = tensor.extract_slice %arg0[4414771200] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_364 = tensor.expand_shape %extracted_slice_363 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_365 = tensor.extract_slice %arg0[4431548416] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_366 = tensor.expand_shape %extracted_slice_365 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_367 = tensor.extract_slice %arg0[4448325632] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_368 = tensor.expand_shape %extracted_slice_367 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_369 = tensor.extract_slice %arg0[4493414400] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_370 = tensor.expand_shape %extracted_slice_369 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_371 = tensor.extract_slice %arg0[4538503168] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_372 = tensor.expand_shape %extracted_slice_371 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_373 = tensor.extract_slice %arg0[4583591936] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_374 = tensor.expand_shape %extracted_slice_373 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_375 = tensor.extract_slice %arg0[4600369152] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_376 = tensor.expand_shape %extracted_slice_375 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_377 = tensor.extract_slice %arg0[4617146368] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_378 = tensor.expand_shape %extracted_slice_377 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_379 = tensor.extract_slice %arg0[4633923584] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_380 = tensor.expand_shape %extracted_slice_379 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_381 = tensor.extract_slice %arg0[4650700800] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_382 = tensor.expand_shape %extracted_slice_381 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_383 = tensor.extract_slice %arg0[4695789568] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_384 = tensor.expand_shape %extracted_slice_383 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_385 = tensor.extract_slice %arg0[4740878336] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_386 = tensor.expand_shape %extracted_slice_385 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_387 = tensor.extract_slice %arg0[4785967104] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_388 = tensor.expand_shape %extracted_slice_387 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_389 = tensor.extract_slice %arg0[4802744320] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_390 = tensor.expand_shape %extracted_slice_389 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_391 = tensor.extract_slice %arg0[4819521536] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_392 = tensor.expand_shape %extracted_slice_391 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_393 = tensor.extract_slice %arg0[4836298752] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_394 = tensor.expand_shape %extracted_slice_393 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_395 = tensor.extract_slice %arg0[4853075968] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_396 = tensor.expand_shape %extracted_slice_395 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_397 = tensor.extract_slice %arg0[4898164736] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_398 = tensor.expand_shape %extracted_slice_397 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_399 = tensor.extract_slice %arg0[4943253504] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_400 = tensor.expand_shape %extracted_slice_399 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_401 = tensor.extract_slice %arg0[4988342272] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_402 = tensor.expand_shape %extracted_slice_401 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_403 = tensor.extract_slice %arg0[5005119488] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_404 = tensor.expand_shape %extracted_slice_403 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_405 = tensor.extract_slice %arg0[5021896704] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_406 = tensor.expand_shape %extracted_slice_405 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_407 = tensor.extract_slice %arg0[5038673920] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_408 = tensor.expand_shape %extracted_slice_407 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_409 = tensor.extract_slice %arg0[5055451136] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_410 = tensor.expand_shape %extracted_slice_409 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_411 = tensor.extract_slice %arg0[5100539904] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_412 = tensor.expand_shape %extracted_slice_411 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_413 = tensor.extract_slice %arg0[5145628672] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_414 = tensor.expand_shape %extracted_slice_413 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_415 = tensor.extract_slice %arg0[5190717440] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_416 = tensor.expand_shape %extracted_slice_415 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_417 = tensor.extract_slice %arg0[5207494656] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_418 = tensor.expand_shape %extracted_slice_417 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_419 = tensor.extract_slice %arg0[5224271872] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_420 = tensor.expand_shape %extracted_slice_419 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_421 = tensor.extract_slice %arg0[5241049088] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_422 = tensor.expand_shape %extracted_slice_421 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_423 = tensor.extract_slice %arg0[5257826304] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_424 = tensor.expand_shape %extracted_slice_423 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_425 = tensor.extract_slice %arg0[5302915072] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_426 = tensor.expand_shape %extracted_slice_425 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_427 = tensor.extract_slice %arg0[5348003840] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_428 = tensor.expand_shape %extracted_slice_427 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_429 = tensor.extract_slice %arg0[5393092608] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_430 = tensor.expand_shape %extracted_slice_429 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_431 = tensor.extract_slice %arg0[5409869824] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_432 = tensor.expand_shape %extracted_slice_431 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_433 = tensor.extract_slice %arg0[5426647040] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_434 = tensor.expand_shape %extracted_slice_433 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_435 = tensor.extract_slice %arg0[5443424256] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_436 = tensor.expand_shape %extracted_slice_435 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_437 = tensor.extract_slice %arg0[5460201472] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_438 = tensor.expand_shape %extracted_slice_437 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_439 = tensor.extract_slice %arg0[5505290240] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_440 = tensor.expand_shape %extracted_slice_439 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_441 = tensor.extract_slice %arg0[5550379008] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_442 = tensor.expand_shape %extracted_slice_441 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_443 = tensor.extract_slice %arg0[5595467776] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_444 = tensor.expand_shape %extracted_slice_443 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_445 = tensor.extract_slice %arg0[5612244992] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_446 = tensor.expand_shape %extracted_slice_445 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_447 = tensor.extract_slice %arg0[5629022208] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_448 = tensor.expand_shape %extracted_slice_447 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_449 = tensor.extract_slice %arg0[5645799424] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_450 = tensor.expand_shape %extracted_slice_449 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_451 = tensor.extract_slice %arg0[5662576640] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_452 = tensor.expand_shape %extracted_slice_451 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_453 = tensor.extract_slice %arg0[5707665408] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_454 = tensor.expand_shape %extracted_slice_453 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_455 = tensor.extract_slice %arg0[5752754176] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_456 = tensor.expand_shape %extracted_slice_455 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_457 = tensor.extract_slice %arg0[5797842944] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_458 = tensor.expand_shape %extracted_slice_457 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_459 = tensor.extract_slice %arg0[5814620160] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_460 = tensor.expand_shape %extracted_slice_459 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_461 = tensor.extract_slice %arg0[5831397376] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_462 = tensor.expand_shape %extracted_slice_461 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_463 = tensor.extract_slice %arg0[5848174592] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_464 = tensor.expand_shape %extracted_slice_463 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_465 = tensor.extract_slice %arg0[5864951808] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_466 = tensor.expand_shape %extracted_slice_465 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_467 = tensor.extract_slice %arg0[5910040576] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_468 = tensor.expand_shape %extracted_slice_467 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_469 = tensor.extract_slice %arg0[5955129344] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_470 = tensor.expand_shape %extracted_slice_469 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_471 = tensor.extract_slice %arg0[6000218112] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_472 = tensor.expand_shape %extracted_slice_471 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_473 = tensor.extract_slice %arg0[6016995328] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_474 = tensor.expand_shape %extracted_slice_473 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_475 = tensor.extract_slice %arg0[6033772544] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_476 = tensor.expand_shape %extracted_slice_475 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_477 = tensor.extract_slice %arg0[6050549760] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_478 = tensor.expand_shape %extracted_slice_477 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_479 = tensor.extract_slice %arg0[6067326976] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_480 = tensor.expand_shape %extracted_slice_479 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_481 = tensor.extract_slice %arg0[6112415744] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_482 = tensor.expand_shape %extracted_slice_481 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_483 = tensor.extract_slice %arg0[6157504512] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_484 = tensor.expand_shape %extracted_slice_483 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_485 = tensor.extract_slice %arg0[6202593280] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_486 = tensor.expand_shape %extracted_slice_485 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_487 = tensor.extract_slice %arg0[6219370496] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_488 = tensor.expand_shape %extracted_slice_487 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_489 = tensor.extract_slice %arg0[6236147712] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_490 = tensor.expand_shape %extracted_slice_489 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_491 = tensor.extract_slice %arg0[6252924928] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_492 = tensor.expand_shape %extracted_slice_491 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_493 = tensor.extract_slice %arg0[6269702144] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_494 = tensor.expand_shape %extracted_slice_493 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_495 = tensor.extract_slice %arg0[6314790912] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_496 = tensor.expand_shape %extracted_slice_495 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_497 = tensor.extract_slice %arg0[6359879680] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_498 = tensor.expand_shape %extracted_slice_497 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_499 = tensor.extract_slice %arg0[6404968448] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_500 = tensor.expand_shape %extracted_slice_499 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_501 = tensor.extract_slice %arg0[6421745664] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_502 = tensor.expand_shape %extracted_slice_501 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_503 = tensor.extract_slice %arg0[6438522880] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_504 = tensor.expand_shape %extracted_slice_503 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_505 = tensor.extract_slice %arg0[6455300096] [16777216] [1] : tensor<6755192832xbf16> to tensor<16777216xbf16>
    %expanded_506 = tensor.expand_shape %extracted_slice_505 [[0, 1]] : tensor<16777216xbf16> into tensor<4096x4096xbf16>
    %extracted_slice_507 = tensor.extract_slice %arg0[6472077312] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_508 = tensor.expand_shape %extracted_slice_507 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_509 = tensor.extract_slice %arg0[6517166080] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_510 = tensor.expand_shape %extracted_slice_509 [[0, 1]] : tensor<45088768xbf16> into tensor<11008x4096xbf16>
    %extracted_slice_511 = tensor.extract_slice %arg0[6562254848] [45088768] [1] : tensor<6755192832xbf16> to tensor<45088768xbf16>
    %expanded_512 = tensor.expand_shape %extracted_slice_511 [[0, 1]] : tensor<45088768xbf16> into tensor<4096x11008xbf16>
    %extracted_slice_513 = tensor.extract_slice %arg0[6607343616] [131072000] [1] : tensor<6755192832xbf16> to tensor<131072000xbf16>
    %expanded_514 = tensor.expand_shape %extracted_slice_513 [[0, 1]] : tensor<131072000xbf16> into tensor<32000x4096xbf16>
    %extracted_slice_515 = tensor.extract_slice %arg0[6738415616] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_516 = tensor.expand_shape %extracted_slice_515 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_517 = tensor.extract_slice %arg0[6738677760] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_518 = tensor.expand_shape %extracted_slice_517 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_519 = tensor.extract_slice %arg0[6738939904] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_520 = tensor.expand_shape %extracted_slice_519 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_521 = tensor.extract_slice %arg0[6739202048] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_522 = tensor.expand_shape %extracted_slice_521 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_523 = tensor.extract_slice %arg0[6739464192] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_524 = tensor.expand_shape %extracted_slice_523 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_525 = tensor.extract_slice %arg0[6739726336] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_526 = tensor.expand_shape %extracted_slice_525 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_527 = tensor.extract_slice %arg0[6739988480] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_528 = tensor.expand_shape %extracted_slice_527 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_529 = tensor.extract_slice %arg0[6740250624] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_530 = tensor.expand_shape %extracted_slice_529 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_531 = tensor.extract_slice %arg0[6740512768] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_532 = tensor.expand_shape %extracted_slice_531 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_533 = tensor.extract_slice %arg0[6740774912] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_534 = tensor.expand_shape %extracted_slice_533 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_535 = tensor.extract_slice %arg0[6741037056] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_536 = tensor.expand_shape %extracted_slice_535 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_537 = tensor.extract_slice %arg0[6741299200] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_538 = tensor.expand_shape %extracted_slice_537 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_539 = tensor.extract_slice %arg0[6741561344] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_540 = tensor.expand_shape %extracted_slice_539 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_541 = tensor.extract_slice %arg0[6741823488] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_542 = tensor.expand_shape %extracted_slice_541 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_543 = tensor.extract_slice %arg0[6742085632] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_544 = tensor.expand_shape %extracted_slice_543 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_545 = tensor.extract_slice %arg0[6742347776] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_546 = tensor.expand_shape %extracted_slice_545 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_547 = tensor.extract_slice %arg0[6742609920] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_548 = tensor.expand_shape %extracted_slice_547 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_549 = tensor.extract_slice %arg0[6742872064] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_550 = tensor.expand_shape %extracted_slice_549 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_551 = tensor.extract_slice %arg0[6743134208] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_552 = tensor.expand_shape %extracted_slice_551 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_553 = tensor.extract_slice %arg0[6743396352] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_554 = tensor.expand_shape %extracted_slice_553 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_555 = tensor.extract_slice %arg0[6743658496] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_556 = tensor.expand_shape %extracted_slice_555 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_557 = tensor.extract_slice %arg0[6743920640] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_558 = tensor.expand_shape %extracted_slice_557 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_559 = tensor.extract_slice %arg0[6744182784] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_560 = tensor.expand_shape %extracted_slice_559 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_561 = tensor.extract_slice %arg0[6744444928] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_562 = tensor.expand_shape %extracted_slice_561 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_563 = tensor.extract_slice %arg0[6744707072] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_564 = tensor.expand_shape %extracted_slice_563 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_565 = tensor.extract_slice %arg0[6744969216] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_566 = tensor.expand_shape %extracted_slice_565 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_567 = tensor.extract_slice %arg0[6745231360] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_568 = tensor.expand_shape %extracted_slice_567 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_569 = tensor.extract_slice %arg0[6745493504] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_570 = tensor.expand_shape %extracted_slice_569 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_571 = tensor.extract_slice %arg0[6745755648] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_572 = tensor.expand_shape %extracted_slice_571 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_573 = tensor.extract_slice %arg0[6746017792] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_574 = tensor.expand_shape %extracted_slice_573 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_575 = tensor.extract_slice %arg0[6746279936] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_576 = tensor.expand_shape %extracted_slice_575 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_577 = tensor.extract_slice %arg0[6746542080] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_578 = tensor.expand_shape %extracted_slice_577 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_579 = tensor.extract_slice %arg0[6746804224] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_580 = tensor.expand_shape %extracted_slice_579 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_581 = tensor.extract_slice %arg0[6747066368] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_582 = tensor.expand_shape %extracted_slice_581 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_583 = tensor.extract_slice %arg0[6747328512] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_584 = tensor.expand_shape %extracted_slice_583 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_585 = tensor.extract_slice %arg0[6747590656] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_586 = tensor.expand_shape %extracted_slice_585 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_587 = tensor.extract_slice %arg0[6747852800] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_588 = tensor.expand_shape %extracted_slice_587 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_589 = tensor.extract_slice %arg0[6748114944] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_590 = tensor.expand_shape %extracted_slice_589 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_591 = tensor.extract_slice %arg0[6748377088] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_592 = tensor.expand_shape %extracted_slice_591 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_593 = tensor.extract_slice %arg0[6748639232] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_594 = tensor.expand_shape %extracted_slice_593 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_595 = tensor.extract_slice %arg0[6748901376] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_596 = tensor.expand_shape %extracted_slice_595 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_597 = tensor.extract_slice %arg0[6749163520] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_598 = tensor.expand_shape %extracted_slice_597 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_599 = tensor.extract_slice %arg0[6749425664] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_600 = tensor.expand_shape %extracted_slice_599 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_601 = tensor.extract_slice %arg0[6749687808] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_602 = tensor.expand_shape %extracted_slice_601 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_603 = tensor.extract_slice %arg0[6749949952] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_604 = tensor.expand_shape %extracted_slice_603 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_605 = tensor.extract_slice %arg0[6750212096] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_606 = tensor.expand_shape %extracted_slice_605 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_607 = tensor.extract_slice %arg0[6750474240] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_608 = tensor.expand_shape %extracted_slice_607 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_609 = tensor.extract_slice %arg0[6750736384] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_610 = tensor.expand_shape %extracted_slice_609 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_611 = tensor.extract_slice %arg0[6750998528] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_612 = tensor.expand_shape %extracted_slice_611 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_613 = tensor.extract_slice %arg0[6751260672] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_614 = tensor.expand_shape %extracted_slice_613 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_615 = tensor.extract_slice %arg0[6751522816] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_616 = tensor.expand_shape %extracted_slice_615 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_617 = tensor.extract_slice %arg0[6751784960] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_618 = tensor.expand_shape %extracted_slice_617 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_619 = tensor.extract_slice %arg0[6752047104] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_620 = tensor.expand_shape %extracted_slice_619 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_621 = tensor.extract_slice %arg0[6752309248] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_622 = tensor.expand_shape %extracted_slice_621 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_623 = tensor.extract_slice %arg0[6752571392] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_624 = tensor.expand_shape %extracted_slice_623 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_625 = tensor.extract_slice %arg0[6752833536] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_626 = tensor.expand_shape %extracted_slice_625 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_627 = tensor.extract_slice %arg0[6753095680] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_628 = tensor.expand_shape %extracted_slice_627 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_629 = tensor.extract_slice %arg0[6753357824] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_630 = tensor.expand_shape %extracted_slice_629 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_631 = tensor.extract_slice %arg0[6753619968] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_632 = tensor.expand_shape %extracted_slice_631 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_633 = tensor.extract_slice %arg0[6753882112] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_634 = tensor.expand_shape %extracted_slice_633 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_635 = tensor.extract_slice %arg0[6754144256] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_636 = tensor.expand_shape %extracted_slice_635 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_637 = tensor.extract_slice %arg0[6754406400] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_638 = tensor.expand_shape %extracted_slice_637 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_639 = tensor.extract_slice %arg0[6754668544] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_640 = tensor.expand_shape %extracted_slice_639 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %extracted_slice_641 = tensor.extract_slice %arg0[6754930688] [262144] [1] : tensor<6755192832xbf16> to tensor<262144xbf16>
    %expanded_642 = tensor.expand_shape %extracted_slice_641 [[0, 1, 2, 3]] : tensor<262144xbf16> into tensor<1x1x2048x128xbf16>
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>
    %0 = tosa.reshape %cst {new_shape = array<i64: 1, 40>} : (tensor<40xi64>) -> tensor<1x40xi64>
    %1 = tosa.reshape %0 {new_shape = array<i64: 1, 40>} : (tensor<1x40xi64>) -> tensor<1x40xi64>
    %2 = tosa.cast %arg1 : (tensor<1x40xi64>) -> tensor<1x40xi32>
    %3 = tosa.reshape %expanded {new_shape = array<i64: 1, 32000, 4096>} : (tensor<32000x4096xbf16>) -> tensor<1x32000x4096xbf16>
    %4 = tosa.gather %3, %2 : (tensor<1x32000x4096xbf16>, tensor<1x40xi32>) -> tensor<1x40x4096xbf16>
    %5 = tosa.reshape %4 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %cst_643 = arith.constant dense<true> : tensor<1x40xi1>
    %cst_644 = arith.constant dense<-3.38953139E+38> : tensor<40x40xf32>
    %cst_645 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>
    %6 = "tosa.const"() <{value = dense<1> : tensor<40xi64>}> : () -> tensor<40xi64>
    %7 = tosa.add %cst_645, %6 : (tensor<40xi64>, tensor<40xi64>) -> tensor<40xi64>
    %8 = tosa.reshape %7 {new_shape = array<i64: 40, 1>} : (tensor<40xi64>) -> tensor<40x1xi64>
    %9 = tensor.empty() : tensor<40x40xi1>
    %10 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cst_645, %8 : tensor<40xi64>, tensor<40x1xi64>) outs(%9 : tensor<40x40xi1>) {
    ^bb0(%in: i64, %in_1454: i64, %out: i1):
      %4564 = arith.cmpi slt, %in, %in_1454 : i64
      linalg.yield %4564 : i1
    } -> tensor<40x40xi1>
    %cst_646 = arith.constant 0.000000e+00 : f32
    %11 = tensor.empty() : tensor<40x40xf32>
    %12 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_644, %10 : tensor<40x40xf32>, tensor<40x40xi1>) outs(%11 : tensor<40x40xf32>) {
    ^bb0(%in: f32, %in_1454: i1, %out: f32):
      %4564 = arith.select %in_1454, %cst_646, %in : f32
      linalg.yield %4564 : f32
    } -> tensor<40x40xf32>
    %13 = tensor.empty() : tensor<40x40xbf16>
    %14 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<40x40xf32>) outs(%13 : tensor<40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<40x40xbf16>
    %15 = tosa.reshape %14 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xbf16>) -> tensor<1x40x40xbf16>
    %16 = tosa.reshape %15 {new_shape = array<i64: 1, 1, 40, 40>} : (tensor<1x40x40xbf16>) -> tensor<1x1x40x40xbf16>
    %extracted_slice_647 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xbf16> to tensor<1x1x40x40xbf16>
    %extracted_slice_648 = tensor.extract_slice %extracted_slice_647[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xbf16> to tensor<1x1x40x40xbf16>
    %17 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x40xbf16>}> : () -> tensor<1x1x40x40xbf16>
    %18 = tosa.add %extracted_slice_648, %17 : (tensor<1x1x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x1x40x40xbf16>
    %extracted_slice_649 = tensor.extract_slice %cst_643[0, 0] [1, 40] [1, 1] : tensor<1x40xi1> to tensor<1x40xi1>
    %19 = tosa.reshape %extracted_slice_649 {new_shape = array<i64: 1, 1, 40>} : (tensor<1x40xi1>) -> tensor<1x1x40xi1>
    %20 = tosa.reshape %19 {new_shape = array<i64: 1, 1, 1, 40>} : (tensor<1x1x40xi1>) -> tensor<1x1x1x40xi1>
    %extracted_slice_650 = tensor.extract_slice %20[0, 0, 0, 0] [1, 1, 1, 40] [1, 1, 1, 1] : tensor<1x1x1x40xi1> to tensor<1x1x1x40xi1>
    %21 = "tosa.const"() <{value = dense<false> : tensor<1x1x40x40xi1>}> : () -> tensor<1x1x40x40xi1>
    %22 = tosa.add %extracted_slice_650, %21 : (tensor<1x1x1x40xi1>, tensor<1x1x40x40xi1>) -> tensor<1x1x40x40xi1>
    %23 = tensor.empty() : tensor<1x1x40x40xbf16>
    %24 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22 : tensor<1x1x40x40xi1>) outs(%23 : tensor<1x1x40x40xbf16>) {
    ^bb0(%in: i1, %out: bf16):
      %4564 = arith.extui %in : i1 to i16
      %4565 = arith.sitofp %4564 : i16 to bf16
      linalg.yield %4565 : bf16
    } -> tensor<1x1x40x40xbf16>
    %cst_651 = arith.constant 1.000000e+00 : bf16
    %25 = tensor.empty() : tensor<1x1x40x40xbf16>
    %26 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24 : tensor<1x1x40x40xbf16>) outs(%25 : tensor<1x1x40x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.subf %cst_651, %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x1x40x40xbf16>
    %27 = tensor.empty() : tensor<1x1x40x40xi1>
    %28 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26 : tensor<1x1x40x40xbf16>) outs(%27 : tensor<1x1x40x40xi1>) {
    ^bb0(%in: bf16, %out: i1):
      %4564 = arith.fptosi %in : bf16 to i16
      %4565 = arith.trunci %4564 : i16 to i1
      linalg.yield %4565 : i1
    } -> tensor<1x1x40x40xi1>
    %cst_652 = arith.constant -3.389530e+38 : bf16
    %29 = tensor.empty() : tensor<1x1x40x40xbf16>
    %30 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %28 : tensor<1x1x40x40xbf16>, tensor<1x1x40x40xi1>) outs(%29 : tensor<1x1x40x40xbf16>) {
    ^bb0(%in: bf16, %in_1454: i1, %out: bf16):
      %4564 = arith.select %in_1454, %cst_652, %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x1x40x40xbf16>
    %31 = tosa.add %30, %18 : (tensor<1x1x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x1x40x40xbf16>
    %32 = tensor.empty() : tensor<1x40x4096xf32>
    %33 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<1x40x4096xbf16>) outs(%32 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %34 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32 = arith.constant 2 : i32
    %35 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33 : tensor<1x40x4096xf32>) outs(%34 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_653 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %36 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%35 : tensor<1x40x4096xf32>) outs(%cst_653 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %37 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %38 = tosa.add %36, %37 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %39 = tosa.rsqrt %38 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %40 = tosa.mul %33, %39 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %41 = tensor.empty() : tensor<1x40x4096xbf16>
    %42 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40 : tensor<1x40x4096xf32>) outs(%41 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %43 = tosa.reshape %extracted_slice {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %44 = tosa.mul %43, %42 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %45 = tensor.empty() : tensor<4096x4096xbf16>
    %46 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_66 : tensor<4096x4096xbf16>) outs(%45 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %47 = tosa.reshape %44 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_654 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %48 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%47, %46 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_654 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %49 = tosa.reshape %48 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %50 = tensor.empty() : tensor<4096x4096xbf16>
    %51 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_68 : tensor<4096x4096xbf16>) outs(%50 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %52 = tosa.reshape %44 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_655 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %53 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%52, %51 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_655 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %54 = tosa.reshape %53 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %55 = tensor.empty() : tensor<4096x4096xbf16>
    %56 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_70 : tensor<4096x4096xbf16>) outs(%55 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %57 = tosa.reshape %44 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_656 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %58 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%57, %56 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_656 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %59 = tosa.reshape %58 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %60 = tosa.reshape %49 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %61 = tensor.empty() : tensor<1x32x40x128xbf16>
    %62 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60 : tensor<1x40x32x128xbf16>) outs(%61 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %63 = tosa.reshape %54 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %64 = tensor.empty() : tensor<1x32x40x128xbf16>
    %65 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63 : tensor<1x40x32x128xbf16>) outs(%64 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %66 = tosa.reshape %59 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %67 = tensor.empty() : tensor<1x32x40x128xbf16>
    %68 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66 : tensor<1x40x32x128xbf16>) outs(%67 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_657 = tensor.extract_slice %expanded_516[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_658 = tensor.extract_slice %extracted_slice_657[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_659 = tensor.extract_slice %extracted_slice_658[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_660 = tensor.extract_slice %expanded_518[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_661 = tensor.extract_slice %extracted_slice_660[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_662 = tensor.extract_slice %extracted_slice_661[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %69 = tensor.empty() : tensor<1x40x128xbf16>
    %70 = tensor.collapse_shape %extracted_slice_659 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %71 = tensor.empty() : tensor<40x128xbf16>
    %72 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%70 : tensor<1x40x128xbf16>) outs(%71 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %73 = tensor.empty() : tensor<1x40x128xbf16>
    %74  = tensor.collapse_shape %extracted_slice_662 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %75 = tensor.empty() : tensor<40x128xbf16>
    %76 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%74 : tensor<1x40x128xbf16>) outs(%75 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %77 = tensor.empty() : tensor<1x40x128xbf16>
    %78 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%77 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %72[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %79 = tosa.reshape %78 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %80 = tensor.empty() : tensor<1x40x128xbf16>
    %81 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%80 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %76[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %82 = tosa.reshape %81 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %83 = tosa.mul %62, %79 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_663 = tensor.extract_slice %62[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_664 = tensor.extract_slice %62[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %84 = tensor.empty() : tensor<1x32x40x64xbf16>
    %85 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_664 : tensor<1x32x40x64xbf16>) outs(%84 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %86 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice = tensor.insert_slice %85 into %86[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_665 = tensor.insert_slice %extracted_slice_663 into %inserted_slice[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %87 = tosa.mul %inserted_slice_665, %82 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %88 = tosa.add %83, %87 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %89 = tosa.mul %65, %79 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_666 = tensor.extract_slice %65[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_667 = tensor.extract_slice %65[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %90 = tensor.empty() : tensor<1x32x40x64xbf16>
    %91 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_667 : tensor<1x32x40x64xbf16>) outs(%90 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %92 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_668 = tensor.insert_slice %91 into %92[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_669 = tensor.insert_slice %extracted_slice_666 into %inserted_slice_668[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %93 = tosa.mul %inserted_slice_669, %82 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %94 = tosa.add %89, %93 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %95 = tensor.empty() : tensor<1x32x128x40xbf16>
    %96 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94 : tensor<1x32x40x128xbf16>) outs(%95 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %97 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %98 = tosa.add %88, %97 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %99 = tosa.reshape %98 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %100 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %101 = tosa.add %96, %100 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %102 = tosa.reshape %101 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %103 = tosa.matmul %99, %102 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %104 = tosa.reshape %103 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %105 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %106 = tosa.reciprocal %105 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %107 = tosa.mul %104, %106 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %108 = tosa.add %107, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %109 = tensor.empty() : tensor<1x32x40x40xf32>
    %110 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108 : tensor<1x32x40x40xbf16>) outs(%109 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %111 = tensor.empty() : tensor<1x32x40x1xf32>
    %112 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%111 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %113 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%110 : tensor<1x32x40x40xf32>) outs(%111 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %114 = tensor.empty() : tensor<1x32x40x40xf32>
    %115 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %113 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%114 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %116 = tensor.empty() : tensor<1x32x40x1xf32>
    %117 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%116 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %118 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%115 : tensor<1x32x40x40xf32>) outs(%117 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %119 = tensor.empty() : tensor<1x32x40x40xf32>
    %120 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%115, %118 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%119 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %121 = tensor.empty() : tensor<1x32x40x40xbf16>
    %122 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120 : tensor<1x32x40x40xf32>) outs(%121 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %123 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %124 = tosa.add %122, %123 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %125 = tosa.reshape %124 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %126 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %127 = tosa.add %68, %126 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %128 = tosa.reshape %127 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %129 = tosa.matmul %125, %128 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %130 = tosa.reshape %129 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %131 = tensor.empty() : tensor<1x40x32x128xbf16>
    %132 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%130 : tensor<1x32x40x128xbf16>) outs(%131 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %133 = tosa.identity %132 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %134 = tosa.reshape %133 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %135 = tensor.empty() : tensor<4096x4096xbf16>
    %136 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_72 : tensor<4096x4096xbf16>) outs(%135 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %137 = tosa.reshape %134 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_670 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %138 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%137, %136 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_670 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %139 = tosa.reshape %138 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %140 = tosa.add %5, %139 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %141 = tensor.empty() : tensor<1x40x4096xf32>
    %142 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%140 : tensor<1x40x4096xbf16>) outs(%141 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %143 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_671 = arith.constant 2 : i32
    %144 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%142 : tensor<1x40x4096xf32>) outs(%143 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_671 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_672 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %145 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%144 : tensor<1x40x4096xf32>) outs(%cst_672 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %146 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %147 = tosa.add %145, %146 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %148 = tosa.rsqrt %147 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %149 = tosa.mul %142, %148 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %150 = tensor.empty() : tensor<1x40x4096xbf16>
    %151 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%149 : tensor<1x40x4096xf32>) outs(%150 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %152 = tosa.reshape %extracted_slice_0 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %153 = tosa.mul %152, %151 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %154 = tensor.empty() : tensor<4096x11008xbf16>
    %155 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_74 : tensor<11008x4096xbf16>) outs(%154 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %156 = tosa.reshape %153 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_673 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %157 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%156, %155 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_673 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %158 = tosa.reshape %157 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %159 = tensor.empty() : tensor<1x40x11008xbf16>
    %160 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%158 : tensor<1x40x11008xbf16>) outs(%159 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %161 = tensor.empty() : tensor<4096x11008xbf16>
    %162 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_76 : tensor<11008x4096xbf16>) outs(%161 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %163 = tosa.reshape %153 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_674 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %164 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%163, %162 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_674 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %165 = tosa.reshape %164 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %166 = tosa.mul %160, %165 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %167 = tensor.empty() : tensor<11008x4096xbf16>
    %168 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_78 : tensor<4096x11008xbf16>) outs(%167 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %169 = tosa.reshape %166 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_675 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %170 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%169, %168 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_675 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %171 = tosa.reshape %170 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %172 = tosa.add %140, %171 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %173 = tensor.empty() : tensor<1x40x4096xf32>
    %174 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%172 : tensor<1x40x4096xbf16>) outs(%173 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %175 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_676 = arith.constant 2 : i32
    %176 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%174 : tensor<1x40x4096xf32>) outs(%175 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_676 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_677 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %177 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%176 : tensor<1x40x4096xf32>) outs(%cst_677 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %178 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %179 = tosa.add %177, %178 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %180 = tosa.rsqrt %179 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %181 = tosa.mul %174, %180 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %182 = tensor.empty() : tensor<1x40x4096xbf16>
    %183 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%181 : tensor<1x40x4096xf32>) outs(%182 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %184 = tosa.reshape %extracted_slice_1 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %185 = tosa.mul %184, %183 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %186 = tensor.empty() : tensor<4096x4096xbf16>
    %187 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_80 : tensor<4096x4096xbf16>) outs(%186 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %188 = tosa.reshape %185 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_678 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %189 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%188, %187 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_678 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %190 = tosa.reshape %189 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %191 = tensor.empty() : tensor<4096x4096xbf16>
    %192 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_82 : tensor<4096x4096xbf16>) outs(%191 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %193 = tosa.reshape %185 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_679 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %194 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%193, %192 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_679 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %195 = tosa.reshape %194 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %196 = tensor.empty() : tensor<4096x4096xbf16>
    %197 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_84 : tensor<4096x4096xbf16>) outs(%196 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %198 = tosa.reshape %185 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_680 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %199 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%198, %197 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_680 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %200 = tosa.reshape %199 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %201 = tosa.reshape %190 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %202 = tensor.empty() : tensor<1x32x40x128xbf16>
    %203 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%201 : tensor<1x40x32x128xbf16>) outs(%202 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %204 = tosa.reshape %195 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %205 = tensor.empty() : tensor<1x32x40x128xbf16>
    %206 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%204 : tensor<1x40x32x128xbf16>) outs(%205 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %207 = tosa.reshape %200 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %208 = tensor.empty() : tensor<1x32x40x128xbf16>
    %209 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%207 : tensor<1x40x32x128xbf16>) outs(%208 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_681 = tensor.extract_slice %expanded_520[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_682 = tensor.extract_slice %extracted_slice_681[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_683 = tensor.extract_slice %extracted_slice_682[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_684 = tensor.extract_slice %expanded_522[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_685 = tensor.extract_slice %extracted_slice_684[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_686 = tensor.extract_slice %extracted_slice_685[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %210 = tensor.empty() : tensor<1x40x128xbf16>
    %211  = tensor.collapse_shape %extracted_slice_683 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %212 = tensor.empty() : tensor<40x128xbf16>
    %213 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%211 : tensor<1x40x128xbf16>) outs(%212 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %214 = tensor.empty() : tensor<1x40x128xbf16>
    %215  = tensor.collapse_shape %extracted_slice_686 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %216 = tensor.empty() : tensor<40x128xbf16>
    %217 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%215 : tensor<1x40x128xbf16>) outs(%216 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %218 = tensor.empty() : tensor<1x40x128xbf16>
    %219 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%218 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %213[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %220 = tosa.reshape %219 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %221 = tensor.empty() : tensor<1x40x128xbf16>
    %222 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%221 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %217[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %223 = tosa.reshape %222 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %224 = tosa.mul %203, %220 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_687 = tensor.extract_slice %203[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_688 = tensor.extract_slice %203[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %225 = tensor.empty() : tensor<1x32x40x64xbf16>
    %226 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_688 : tensor<1x32x40x64xbf16>) outs(%225 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %227 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_689 = tensor.insert_slice %226 into %227[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_690 = tensor.insert_slice %extracted_slice_687 into %inserted_slice_689[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %228 = tosa.mul %inserted_slice_690, %223 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %229 = tosa.add %224, %228 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %230 = tosa.mul %206, %220 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_691 = tensor.extract_slice %206[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_692 = tensor.extract_slice %206[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %231 = tensor.empty() : tensor<1x32x40x64xbf16>
    %232 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_692 : tensor<1x32x40x64xbf16>) outs(%231 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %233 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_693 = tensor.insert_slice %232 into %233[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_694 = tensor.insert_slice %extracted_slice_691 into %inserted_slice_693[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %234 = tosa.mul %inserted_slice_694, %223 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %235 = tosa.add %230, %234 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %236 = tensor.empty() : tensor<1x32x128x40xbf16>
    %237 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%235 : tensor<1x32x40x128xbf16>) outs(%236 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %238 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %239 = tosa.add %229, %238 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %240 = tosa.reshape %239 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %241 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %242 = tosa.add %237, %241 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %243 = tosa.reshape %242 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %244 = tosa.matmul %240, %243 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %245 = tosa.reshape %244 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %246 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %247 = tosa.reciprocal %246 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %248 = tosa.mul %245, %247 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %249 = tosa.add %248, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %250 = tensor.empty() : tensor<1x32x40x40xf32>
    %251 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%249 : tensor<1x32x40x40xbf16>) outs(%250 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %252 = tensor.empty() : tensor<1x32x40x1xf32>
    %253 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%252 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %254 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%251 : tensor<1x32x40x40xf32>) outs(%252 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %255 = tensor.empty() : tensor<1x32x40x40xf32>
    %256 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%251, %254 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%255 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %257 = tensor.empty() : tensor<1x32x40x1xf32>
    %258 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%257 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %259 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%256 : tensor<1x32x40x40xf32>) outs(%258 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %260 = tensor.empty() : tensor<1x32x40x40xf32>
    %261 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%256, %259 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%260 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %262 = tensor.empty() : tensor<1x32x40x40xbf16>
    %263 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%261 : tensor<1x32x40x40xf32>) outs(%262 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %264 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %265 = tosa.add %263, %264 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %266 = tosa.reshape %265 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %267 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %268 = tosa.add %209, %267 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %269 = tosa.reshape %268 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %270 = tosa.matmul %266, %269 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %271 = tosa.reshape %270 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %272 = tensor.empty() : tensor<1x40x32x128xbf16>
    %273 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%271 : tensor<1x32x40x128xbf16>) outs(%272 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %274 = tosa.identity %273 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %275 = tosa.reshape %274 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %276 = tensor.empty() : tensor<4096x4096xbf16>
    %277 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_86 : tensor<4096x4096xbf16>) outs(%276 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %278 = tosa.reshape %275 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_695 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %279 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%278, %277 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_695 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %280 = tosa.reshape %279 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %281 = tosa.add %172, %280 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %282 = tensor.empty() : tensor<1x40x4096xf32>
    %283 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%281 : tensor<1x40x4096xbf16>) outs(%282 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %284 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_696 = arith.constant 2 : i32
    %285 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%283 : tensor<1x40x4096xf32>) outs(%284 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_696 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_697 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %286 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%285 : tensor<1x40x4096xf32>) outs(%cst_697 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %287 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %288 = tosa.add %286, %287 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %289 = tosa.rsqrt %288 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %290 = tosa.mul %283, %289 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %291 = tensor.empty() : tensor<1x40x4096xbf16>
    %292 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%290 : tensor<1x40x4096xf32>) outs(%291 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %293 = tosa.reshape %extracted_slice_2 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %294 = tosa.mul %293, %292 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %295 = tensor.empty() : tensor<4096x11008xbf16>
    %296 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_88 : tensor<11008x4096xbf16>) outs(%295 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %297 = tosa.reshape %294 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_698 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %298 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%297, %296 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_698 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %299 = tosa.reshape %298 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %300 = tensor.empty() : tensor<1x40x11008xbf16>
    %301 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%299 : tensor<1x40x11008xbf16>) outs(%300 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %302 = tensor.empty() : tensor<4096x11008xbf16>
    %303 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_90 : tensor<11008x4096xbf16>) outs(%302 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %304 = tosa.reshape %294 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_699 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %305 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%304, %303 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_699 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %306 = tosa.reshape %305 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %307 = tosa.mul %301, %306 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %308 = tensor.empty() : tensor<11008x4096xbf16>
    %309 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_92 : tensor<4096x11008xbf16>) outs(%308 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %310 = tosa.reshape %307 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_700 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %311 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%310, %309 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_700 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %312 = tosa.reshape %311 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %313 = tosa.add %281, %312 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %314 = tensor.empty() : tensor<1x40x4096xf32>
    %315 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%313 : tensor<1x40x4096xbf16>) outs(%314 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %316 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_701 = arith.constant 2 : i32
    %317 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%315 : tensor<1x40x4096xf32>) outs(%316 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_701 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_702 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %318 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%317 : tensor<1x40x4096xf32>) outs(%cst_702 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %319 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %320 = tosa.add %318, %319 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %321 = tosa.rsqrt %320 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %322 = tosa.mul %315, %321 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %323 = tensor.empty() : tensor<1x40x4096xbf16>
    %324 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%322 : tensor<1x40x4096xf32>) outs(%323 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %325 = tosa.reshape %extracted_slice_3 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %326 = tosa.mul %325, %324 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %327 = tensor.empty() : tensor<4096x4096xbf16>
    %328 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_94 : tensor<4096x4096xbf16>) outs(%327 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %329 = tosa.reshape %326 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_703 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %330 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%329, %328 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_703 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %331 = tosa.reshape %330 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %332 = tensor.empty() : tensor<4096x4096xbf16>
    %333 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_96 : tensor<4096x4096xbf16>) outs(%332 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %334 = tosa.reshape %326 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_704 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %335 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%334, %333 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_704 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %336 = tosa.reshape %335 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %337 = tensor.empty() : tensor<4096x4096xbf16>
    %338 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_98 : tensor<4096x4096xbf16>) outs(%337 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %339 = tosa.reshape %326 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_705 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %340 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%339, %338 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_705 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %341 = tosa.reshape %340 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %342 = tosa.reshape %331 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %343 = tensor.empty() : tensor<1x32x40x128xbf16>
    %344 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%342 : tensor<1x40x32x128xbf16>) outs(%343 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %345 = tosa.reshape %336 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %346 = tensor.empty() : tensor<1x32x40x128xbf16>
    %347 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%345 : tensor<1x40x32x128xbf16>) outs(%346 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %348 = tosa.reshape %341 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %349 = tensor.empty() : tensor<1x32x40x128xbf16>
    %350 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%348 : tensor<1x40x32x128xbf16>) outs(%349 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_706 = tensor.extract_slice %expanded_524[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_707 = tensor.extract_slice %extracted_slice_706[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_708 = tensor.extract_slice %extracted_slice_707[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_709 = tensor.extract_slice %expanded_526[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_710 = tensor.extract_slice %extracted_slice_709[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_711 = tensor.extract_slice %extracted_slice_710[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %351 = tensor.empty() : tensor<1x40x128xbf16>
    %352  = tensor.collapse_shape %extracted_slice_708 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %353 = tensor.empty() : tensor<40x128xbf16>
    %354 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%352 : tensor<1x40x128xbf16>) outs(%353 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %355 = tensor.empty() : tensor<1x40x128xbf16>
    %356  = tensor.collapse_shape %extracted_slice_711 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %357 = tensor.empty() : tensor<40x128xbf16>
    %358 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%356 : tensor<1x40x128xbf16>) outs(%357 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %359 = tensor.empty() : tensor<1x40x128xbf16>
    %360 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%359 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %354[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %361 = tosa.reshape %360 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %362 = tensor.empty() : tensor<1x40x128xbf16>
    %363 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%362 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %358[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %364 = tosa.reshape %363 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %365 = tosa.mul %344, %361 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_712 = tensor.extract_slice %344[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_713 = tensor.extract_slice %344[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %366 = tensor.empty() : tensor<1x32x40x64xbf16>
    %367 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_713 : tensor<1x32x40x64xbf16>) outs(%366 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %368 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_714 = tensor.insert_slice %367 into %368[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_715 = tensor.insert_slice %extracted_slice_712 into %inserted_slice_714[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %369 = tosa.mul %inserted_slice_715, %364 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %370 = tosa.add %365, %369 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %371 = tosa.mul %347, %361 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_716 = tensor.extract_slice %347[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_717 = tensor.extract_slice %347[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %372 = tensor.empty() : tensor<1x32x40x64xbf16>
    %373 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_717 : tensor<1x32x40x64xbf16>) outs(%372 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %374 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_718 = tensor.insert_slice %373 into %374[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_719 = tensor.insert_slice %extracted_slice_716 into %inserted_slice_718[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %375 = tosa.mul %inserted_slice_719, %364 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %376 = tosa.add %371, %375 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %377 = tensor.empty() : tensor<1x32x128x40xbf16>
    %378 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%376 : tensor<1x32x40x128xbf16>) outs(%377 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %379 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %380 = tosa.add %370, %379 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %381 = tosa.reshape %380 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %382 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %383 = tosa.add %378, %382 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %384 = tosa.reshape %383 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %385 = tosa.matmul %381, %384 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %386 = tosa.reshape %385 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %387 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %388 = tosa.reciprocal %387 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %389 = tosa.mul %386, %388 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %390 = tosa.add %389, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %391 = tensor.empty() : tensor<1x32x40x40xf32>
    %392 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%390 : tensor<1x32x40x40xbf16>) outs(%391 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %393 = tensor.empty() : tensor<1x32x40x1xf32>
    %394 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%393 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %395 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%392 : tensor<1x32x40x40xf32>) outs(%393 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %396 = tensor.empty() : tensor<1x32x40x40xf32>
    %397 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%392, %395 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%396 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %398 = tensor.empty() : tensor<1x32x40x1xf32>
    %399 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%398 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %400 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%397 : tensor<1x32x40x40xf32>) outs(%399 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %401 = tensor.empty() : tensor<1x32x40x40xf32>
    %402 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%397, %400 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%401 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %403 = tensor.empty() : tensor<1x32x40x40xbf16>
    %404 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%402 : tensor<1x32x40x40xf32>) outs(%403 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %405 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %406 = tosa.add %404, %405 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %407 = tosa.reshape %406 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %408 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %409 = tosa.add %350, %408 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %410 = tosa.reshape %409 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %411 = tosa.matmul %407, %410 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %412 = tosa.reshape %411 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %413 = tensor.empty() : tensor<1x40x32x128xbf16>
    %414 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%412 : tensor<1x32x40x128xbf16>) outs(%413 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %415 = tosa.identity %414 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %416 = tosa.reshape %415 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %417 = tensor.empty() : tensor<4096x4096xbf16>
    %418 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_100 : tensor<4096x4096xbf16>) outs(%417 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %419 = tosa.reshape %416 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_720 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %420 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%419, %418 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_720 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %421 = tosa.reshape %420 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %422 = tosa.add %313, %421 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %423 = tensor.empty() : tensor<1x40x4096xf32>
    %424 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%422 : tensor<1x40x4096xbf16>) outs(%423 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %425 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_721 = arith.constant 2 : i32
    %426 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%424 : tensor<1x40x4096xf32>) outs(%425 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_721 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_722 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %427 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%426 : tensor<1x40x4096xf32>) outs(%cst_722 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %428 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %429 = tosa.add %427, %428 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %430 = tosa.rsqrt %429 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %431 = tosa.mul %424, %430 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %432 = tensor.empty() : tensor<1x40x4096xbf16>
    %433 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%431 : tensor<1x40x4096xf32>) outs(%432 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %434 = tosa.reshape %extracted_slice_4 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %435 = tosa.mul %434, %433 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %436 = tensor.empty() : tensor<4096x11008xbf16>
    %437 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_102 : tensor<11008x4096xbf16>) outs(%436 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %438 = tosa.reshape %435 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_723 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %439 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%438, %437 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_723 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %440 = tosa.reshape %439 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %441 = tensor.empty() : tensor<1x40x11008xbf16>
    %442 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%440 : tensor<1x40x11008xbf16>) outs(%441 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %443 = tensor.empty() : tensor<4096x11008xbf16>
    %444 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_104 : tensor<11008x4096xbf16>) outs(%443 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %445 = tosa.reshape %435 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_724 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %446 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%445, %444 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_724 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %447 = tosa.reshape %446 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %448 = tosa.mul %442, %447 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %449 = tensor.empty() : tensor<11008x4096xbf16>
    %450 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_106 : tensor<4096x11008xbf16>) outs(%449 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %451 = tosa.reshape %448 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_725 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %452 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%451, %450 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_725 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %453 = tosa.reshape %452 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %454 = tosa.add %422, %453 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %455 = tensor.empty() : tensor<1x40x4096xf32>
    %456 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%454 : tensor<1x40x4096xbf16>) outs(%455 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %457 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_726 = arith.constant 2 : i32
    %458 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%456 : tensor<1x40x4096xf32>) outs(%457 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_726 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_727 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %459 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%458 : tensor<1x40x4096xf32>) outs(%cst_727 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %460 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %461 = tosa.add %459, %460 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %462 = tosa.rsqrt %461 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %463 = tosa.mul %456, %462 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %464 = tensor.empty() : tensor<1x40x4096xbf16>
    %465 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%463 : tensor<1x40x4096xf32>) outs(%464 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %466 = tosa.reshape %extracted_slice_5 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %467 = tosa.mul %466, %465 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %468 = tensor.empty() : tensor<4096x4096xbf16>
    %469 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_108 : tensor<4096x4096xbf16>) outs(%468 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %470 = tosa.reshape %467 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_728 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %471 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%470, %469 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_728 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %472 = tosa.reshape %471 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %473 = tensor.empty() : tensor<4096x4096xbf16>
    %474 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_110 : tensor<4096x4096xbf16>) outs(%473 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %475 = tosa.reshape %467 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_729 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %476 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%475, %474 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_729 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %477 = tosa.reshape %476 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %478 = tensor.empty() : tensor<4096x4096xbf16>
    %479 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_112 : tensor<4096x4096xbf16>) outs(%478 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %480 = tosa.reshape %467 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_730 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %481 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%480, %479 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_730 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %482 = tosa.reshape %481 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %483 = tosa.reshape %472 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %484 = tensor.empty() : tensor<1x32x40x128xbf16>
    %485 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%483 : tensor<1x40x32x128xbf16>) outs(%484 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %486 = tosa.reshape %477 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %487 = tensor.empty() : tensor<1x32x40x128xbf16>
    %488 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%486 : tensor<1x40x32x128xbf16>) outs(%487 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %489 = tosa.reshape %482 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %490 = tensor.empty() : tensor<1x32x40x128xbf16>
    %491 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%489 : tensor<1x40x32x128xbf16>) outs(%490 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_731 = tensor.extract_slice %expanded_528[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_732 = tensor.extract_slice %extracted_slice_731[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_733 = tensor.extract_slice %extracted_slice_732[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_734 = tensor.extract_slice %expanded_530[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_735 = tensor.extract_slice %extracted_slice_734[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_736 = tensor.extract_slice %extracted_slice_735[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %492 = tensor.empty() : tensor<1x40x128xbf16>
    %493  = tensor.collapse_shape %extracted_slice_733 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %494 = tensor.empty() : tensor<40x128xbf16>
    %495 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%493 : tensor<1x40x128xbf16>) outs(%494 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %496 = tensor.empty() : tensor<1x40x128xbf16>
    %497  = tensor.collapse_shape %extracted_slice_736 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %498 = tensor.empty() : tensor<40x128xbf16>
    %499 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%497 : tensor<1x40x128xbf16>) outs(%498 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %500 = tensor.empty() : tensor<1x40x128xbf16>
    %501 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%500 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %495[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %502 = tosa.reshape %501 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %503 = tensor.empty() : tensor<1x40x128xbf16>
    %504 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%503 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %499[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %505 = tosa.reshape %504 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %506 = tosa.mul %485, %502 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_737 = tensor.extract_slice %485[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_738 = tensor.extract_slice %485[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %507 = tensor.empty() : tensor<1x32x40x64xbf16>
    %508 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_738 : tensor<1x32x40x64xbf16>) outs(%507 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %509 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_739 = tensor.insert_slice %508 into %509[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_740 = tensor.insert_slice %extracted_slice_737 into %inserted_slice_739[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %510 = tosa.mul %inserted_slice_740, %505 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %511 = tosa.add %506, %510 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %512 = tosa.mul %488, %502 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_741 = tensor.extract_slice %488[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_742 = tensor.extract_slice %488[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %513 = tensor.empty() : tensor<1x32x40x64xbf16>
    %514 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_742 : tensor<1x32x40x64xbf16>) outs(%513 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %515 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_743 = tensor.insert_slice %514 into %515[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_744 = tensor.insert_slice %extracted_slice_741 into %inserted_slice_743[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %516 = tosa.mul %inserted_slice_744, %505 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %517 = tosa.add %512, %516 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %518 = tensor.empty() : tensor<1x32x128x40xbf16>
    %519 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%517 : tensor<1x32x40x128xbf16>) outs(%518 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %520 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %521 = tosa.add %511, %520 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %522 = tosa.reshape %521 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %523 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %524 = tosa.add %519, %523 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %525 = tosa.reshape %524 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %526 = tosa.matmul %522, %525 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %527 = tosa.reshape %526 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %528 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %529 = tosa.reciprocal %528 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %530 = tosa.mul %527, %529 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %531 = tosa.add %530, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %532 = tensor.empty() : tensor<1x32x40x40xf32>
    %533 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%531 : tensor<1x32x40x40xbf16>) outs(%532 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %534 = tensor.empty() : tensor<1x32x40x1xf32>
    %535 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%534 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %536 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%533 : tensor<1x32x40x40xf32>) outs(%534 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %537 = tensor.empty() : tensor<1x32x40x40xf32>
    %538 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%533, %536 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%537 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %539 = tensor.empty() : tensor<1x32x40x1xf32>
    %540 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%539 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %541 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%538 : tensor<1x32x40x40xf32>) outs(%540 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %542 = tensor.empty() : tensor<1x32x40x40xf32>
    %543 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%538, %541 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%542 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %544 = tensor.empty() : tensor<1x32x40x40xbf16>
    %545 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%543 : tensor<1x32x40x40xf32>) outs(%544 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %546 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %547 = tosa.add %545, %546 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %548 = tosa.reshape %547 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %549 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %550 = tosa.add %491, %549 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %551 = tosa.reshape %550 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %552 = tosa.matmul %548, %551 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %553 = tosa.reshape %552 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %554 = tensor.empty() : tensor<1x40x32x128xbf16>
    %555 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%553 : tensor<1x32x40x128xbf16>) outs(%554 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %556 = tosa.identity %555 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %557 = tosa.reshape %556 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %558 = tensor.empty() : tensor<4096x4096xbf16>
    %559 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_114 : tensor<4096x4096xbf16>) outs(%558 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %560 = tosa.reshape %557 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_745 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %561 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%560, %559 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_745 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %562 = tosa.reshape %561 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %563 = tosa.add %454, %562 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %564 = tensor.empty() : tensor<1x40x4096xf32>
    %565 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%563 : tensor<1x40x4096xbf16>) outs(%564 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %566 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_746 = arith.constant 2 : i32
    %567 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%565 : tensor<1x40x4096xf32>) outs(%566 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_746 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_747 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %568 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%567 : tensor<1x40x4096xf32>) outs(%cst_747 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %569 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %570 = tosa.add %568, %569 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %571 = tosa.rsqrt %570 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %572 = tosa.mul %565, %571 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %573 = tensor.empty() : tensor<1x40x4096xbf16>
    %574 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%572 : tensor<1x40x4096xf32>) outs(%573 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %575 = tosa.reshape %extracted_slice_6 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %576 = tosa.mul %575, %574 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %577 = tensor.empty() : tensor<4096x11008xbf16>
    %578 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_116 : tensor<11008x4096xbf16>) outs(%577 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %579 = tosa.reshape %576 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_748 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %580 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%579, %578 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_748 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %581 = tosa.reshape %580 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %582 = tensor.empty() : tensor<1x40x11008xbf16>
    %583 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%581 : tensor<1x40x11008xbf16>) outs(%582 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %584 = tensor.empty() : tensor<4096x11008xbf16>
    %585 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_118 : tensor<11008x4096xbf16>) outs(%584 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %586 = tosa.reshape %576 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_749 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %587 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%586, %585 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_749 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %588 = tosa.reshape %587 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %589 = tosa.mul %583, %588 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %590 = tensor.empty() : tensor<11008x4096xbf16>
    %591 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_120 : tensor<4096x11008xbf16>) outs(%590 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %592 = tosa.reshape %589 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_750 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %593 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%592, %591 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_750 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %594 = tosa.reshape %593 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %595 = tosa.add %563, %594 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %596 = tensor.empty() : tensor<1x40x4096xf32>
    %597 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%595 : tensor<1x40x4096xbf16>) outs(%596 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %598 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_751 = arith.constant 2 : i32
    %599 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%597 : tensor<1x40x4096xf32>) outs(%598 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_751 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_752 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %600 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%599 : tensor<1x40x4096xf32>) outs(%cst_752 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %601 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %602 = tosa.add %600, %601 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %603 = tosa.rsqrt %602 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %604 = tosa.mul %597, %603 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %605 = tensor.empty() : tensor<1x40x4096xbf16>
    %606 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%604 : tensor<1x40x4096xf32>) outs(%605 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %607 = tosa.reshape %extracted_slice_7 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %608 = tosa.mul %607, %606 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %609 = tensor.empty() : tensor<4096x4096xbf16>
    %610 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_122 : tensor<4096x4096xbf16>) outs(%609 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %611 = tosa.reshape %608 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_753 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %612 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%611, %610 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_753 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %613 = tosa.reshape %612 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %614 = tensor.empty() : tensor<4096x4096xbf16>
    %615 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_124 : tensor<4096x4096xbf16>) outs(%614 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %616 = tosa.reshape %608 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_754 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %617 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%616, %615 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_754 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %618 = tosa.reshape %617 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %619 = tensor.empty() : tensor<4096x4096xbf16>
    %620 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_126 : tensor<4096x4096xbf16>) outs(%619 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %621 = tosa.reshape %608 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_755 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %622 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%621, %620 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_755 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %623 = tosa.reshape %622 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %624 = tosa.reshape %613 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %625 = tensor.empty() : tensor<1x32x40x128xbf16>
    %626 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%624 : tensor<1x40x32x128xbf16>) outs(%625 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %627 = tosa.reshape %618 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %628 = tensor.empty() : tensor<1x32x40x128xbf16>
    %629 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%627 : tensor<1x40x32x128xbf16>) outs(%628 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %630 = tosa.reshape %623 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %631 = tensor.empty() : tensor<1x32x40x128xbf16>
    %632 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%630 : tensor<1x40x32x128xbf16>) outs(%631 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_756 = tensor.extract_slice %expanded_532[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_757 = tensor.extract_slice %extracted_slice_756[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_758 = tensor.extract_slice %extracted_slice_757[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_759 = tensor.extract_slice %expanded_534[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_760 = tensor.extract_slice %extracted_slice_759[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_761 = tensor.extract_slice %extracted_slice_760[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %633 = tensor.empty() : tensor<1x40x128xbf16>
    %634  = tensor.collapse_shape %extracted_slice_758 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %635 = tensor.empty() : tensor<40x128xbf16>
    %636 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%634 : tensor<1x40x128xbf16>) outs(%635 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %637 = tensor.empty() : tensor<1x40x128xbf16>
    %638  = tensor.collapse_shape %extracted_slice_761 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %639 = tensor.empty() : tensor<40x128xbf16>
    %640 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%638 : tensor<1x40x128xbf16>) outs(%639 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %641 = tensor.empty() : tensor<1x40x128xbf16>
    %642 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%641 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %636[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %643 = tosa.reshape %642 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %644 = tensor.empty() : tensor<1x40x128xbf16>
    %645 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%644 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %640[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %646 = tosa.reshape %645 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %647 = tosa.mul %626, %643 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_762 = tensor.extract_slice %626[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_763 = tensor.extract_slice %626[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %648 = tensor.empty() : tensor<1x32x40x64xbf16>
    %649 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_763 : tensor<1x32x40x64xbf16>) outs(%648 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %650 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_764 = tensor.insert_slice %649 into %650[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_765 = tensor.insert_slice %extracted_slice_762 into %inserted_slice_764[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %651 = tosa.mul %inserted_slice_765, %646 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %652 = tosa.add %647, %651 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %653 = tosa.mul %629, %643 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_766 = tensor.extract_slice %629[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_767 = tensor.extract_slice %629[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %654 = tensor.empty() : tensor<1x32x40x64xbf16>
    %655 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_767 : tensor<1x32x40x64xbf16>) outs(%654 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %656 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_768 = tensor.insert_slice %655 into %656[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_769 = tensor.insert_slice %extracted_slice_766 into %inserted_slice_768[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %657 = tosa.mul %inserted_slice_769, %646 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %658 = tosa.add %653, %657 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %659 = tensor.empty() : tensor<1x32x128x40xbf16>
    %660 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%658 : tensor<1x32x40x128xbf16>) outs(%659 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %661 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %662 = tosa.add %652, %661 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %663 = tosa.reshape %662 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %664 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %665 = tosa.add %660, %664 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %666 = tosa.reshape %665 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %667 = tosa.matmul %663, %666 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %668 = tosa.reshape %667 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %669 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %670 = tosa.reciprocal %669 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %671 = tosa.mul %668, %670 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %672 = tosa.add %671, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %673 = tensor.empty() : tensor<1x32x40x40xf32>
    %674 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%672 : tensor<1x32x40x40xbf16>) outs(%673 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %675 = tensor.empty() : tensor<1x32x40x1xf32>
    %676 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%675 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %677 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%674 : tensor<1x32x40x40xf32>) outs(%675 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %678 = tensor.empty() : tensor<1x32x40x40xf32>
    %679 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%674, %677 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%678 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %680 = tensor.empty() : tensor<1x32x40x1xf32>
    %681 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%680 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %682 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%679 : tensor<1x32x40x40xf32>) outs(%681 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %683 = tensor.empty() : tensor<1x32x40x40xf32>
    %684 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%679, %682 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%683 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %685 = tensor.empty() : tensor<1x32x40x40xbf16>
    %686 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%684 : tensor<1x32x40x40xf32>) outs(%685 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %687 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %688 = tosa.add %686, %687 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %689 = tosa.reshape %688 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %690 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %691 = tosa.add %632, %690 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %692 = tosa.reshape %691 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %693 = tosa.matmul %689, %692 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %694 = tosa.reshape %693 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %695 = tensor.empty() : tensor<1x40x32x128xbf16>
    %696 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%694 : tensor<1x32x40x128xbf16>) outs(%695 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %697 = tosa.identity %696 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %698 = tosa.reshape %697 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %699 = tensor.empty() : tensor<4096x4096xbf16>
    %700 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_128 : tensor<4096x4096xbf16>) outs(%699 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %701 = tosa.reshape %698 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_770 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %702 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%701, %700 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_770 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %703 = tosa.reshape %702 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %704 = tosa.add %595, %703 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %705 = tensor.empty() : tensor<1x40x4096xf32>
    %706 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%704 : tensor<1x40x4096xbf16>) outs(%705 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %707 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_771 = arith.constant 2 : i32
    %708 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%706 : tensor<1x40x4096xf32>) outs(%707 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_771 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_772 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %709 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%708 : tensor<1x40x4096xf32>) outs(%cst_772 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %710 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %711 = tosa.add %709, %710 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %712 = tosa.rsqrt %711 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %713 = tosa.mul %706, %712 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %714 = tensor.empty() : tensor<1x40x4096xbf16>
    %715 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%713 : tensor<1x40x4096xf32>) outs(%714 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %716 = tosa.reshape %extracted_slice_8 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %717 = tosa.mul %716, %715 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %718 = tensor.empty() : tensor<4096x11008xbf16>
    %719 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_130 : tensor<11008x4096xbf16>) outs(%718 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %720 = tosa.reshape %717 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_773 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %721 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%720, %719 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_773 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %722 = tosa.reshape %721 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %723 = tensor.empty() : tensor<1x40x11008xbf16>
    %724 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%722 : tensor<1x40x11008xbf16>) outs(%723 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %725 = tensor.empty() : tensor<4096x11008xbf16>
    %726 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_132 : tensor<11008x4096xbf16>) outs(%725 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %727 = tosa.reshape %717 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_774 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %728 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%727, %726 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_774 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %729 = tosa.reshape %728 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %730 = tosa.mul %724, %729 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %731 = tensor.empty() : tensor<11008x4096xbf16>
    %732 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_134 : tensor<4096x11008xbf16>) outs(%731 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %733 = tosa.reshape %730 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_775 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %734 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%733, %732 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_775 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %735 = tosa.reshape %734 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %736 = tosa.add %704, %735 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %737 = tensor.empty() : tensor<1x40x4096xf32>
    %738 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%736 : tensor<1x40x4096xbf16>) outs(%737 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %739 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_776 = arith.constant 2 : i32
    %740 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%738 : tensor<1x40x4096xf32>) outs(%739 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_776 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_777 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %741 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%740 : tensor<1x40x4096xf32>) outs(%cst_777 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %742 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %743 = tosa.add %741, %742 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %744 = tosa.rsqrt %743 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %745 = tosa.mul %738, %744 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %746 = tensor.empty() : tensor<1x40x4096xbf16>
    %747 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%745 : tensor<1x40x4096xf32>) outs(%746 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %748 = tosa.reshape %extracted_slice_9 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %749 = tosa.mul %748, %747 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %750 = tensor.empty() : tensor<4096x4096xbf16>
    %751 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_136 : tensor<4096x4096xbf16>) outs(%750 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %752 = tosa.reshape %749 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_778 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %753 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%752, %751 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_778 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %754 = tosa.reshape %753 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %755 = tensor.empty() : tensor<4096x4096xbf16>
    %756 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_138 : tensor<4096x4096xbf16>) outs(%755 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %757 = tosa.reshape %749 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_779 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %758 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%757, %756 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_779 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %759 = tosa.reshape %758 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %760 = tensor.empty() : tensor<4096x4096xbf16>
    %761 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_140 : tensor<4096x4096xbf16>) outs(%760 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %762 = tosa.reshape %749 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_780 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %763 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%762, %761 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_780 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %764 = tosa.reshape %763 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %765 = tosa.reshape %754 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %766 = tensor.empty() : tensor<1x32x40x128xbf16>
    %767 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%765 : tensor<1x40x32x128xbf16>) outs(%766 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %768 = tosa.reshape %759 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %769 = tensor.empty() : tensor<1x32x40x128xbf16>
    %770 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%768 : tensor<1x40x32x128xbf16>) outs(%769 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %771 = tosa.reshape %764 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %772 = tensor.empty() : tensor<1x32x40x128xbf16>
    %773 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%771 : tensor<1x40x32x128xbf16>) outs(%772 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_781 = tensor.extract_slice %expanded_536[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_782 = tensor.extract_slice %extracted_slice_781[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_783 = tensor.extract_slice %extracted_slice_782[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_784 = tensor.extract_slice %expanded_538[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_785 = tensor.extract_slice %extracted_slice_784[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_786 = tensor.extract_slice %extracted_slice_785[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %774 = tensor.empty() : tensor<1x40x128xbf16>
    %775  = tensor.collapse_shape %extracted_slice_783 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %776 = tensor.empty() : tensor<40x128xbf16>
    %777 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%775 : tensor<1x40x128xbf16>) outs(%776 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %778 = tensor.empty() : tensor<1x40x128xbf16>
    %779  = tensor.collapse_shape %extracted_slice_786 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %780 = tensor.empty() : tensor<40x128xbf16>
    %781 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%779 : tensor<1x40x128xbf16>) outs(%780 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %782 = tensor.empty() : tensor<1x40x128xbf16>
    %783 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%782 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %777[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %784 = tosa.reshape %783 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %785 = tensor.empty() : tensor<1x40x128xbf16>
    %786 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%785 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %781[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %787 = tosa.reshape %786 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %788 = tosa.mul %767, %784 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_787 = tensor.extract_slice %767[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_788 = tensor.extract_slice %767[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %789 = tensor.empty() : tensor<1x32x40x64xbf16>
    %790 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_788 : tensor<1x32x40x64xbf16>) outs(%789 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %791 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_789 = tensor.insert_slice %790 into %791[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_790 = tensor.insert_slice %extracted_slice_787 into %inserted_slice_789[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %792 = tosa.mul %inserted_slice_790, %787 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %793 = tosa.add %788, %792 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %794 = tosa.mul %770, %784 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_791 = tensor.extract_slice %770[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_792 = tensor.extract_slice %770[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %795 = tensor.empty() : tensor<1x32x40x64xbf16>
    %796 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_792 : tensor<1x32x40x64xbf16>) outs(%795 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %797 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_793 = tensor.insert_slice %796 into %797[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_794 = tensor.insert_slice %extracted_slice_791 into %inserted_slice_793[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %798 = tosa.mul %inserted_slice_794, %787 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %799 = tosa.add %794, %798 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %800 = tensor.empty() : tensor<1x32x128x40xbf16>
    %801 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%799 : tensor<1x32x40x128xbf16>) outs(%800 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %802 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %803 = tosa.add %793, %802 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %804 = tosa.reshape %803 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %805 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %806 = tosa.add %801, %805 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %807 = tosa.reshape %806 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %808 = tosa.matmul %804, %807 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %809 = tosa.reshape %808 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %810 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %811 = tosa.reciprocal %810 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %812 = tosa.mul %809, %811 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %813 = tosa.add %812, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %814 = tensor.empty() : tensor<1x32x40x40xf32>
    %815 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%813 : tensor<1x32x40x40xbf16>) outs(%814 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %816 = tensor.empty() : tensor<1x32x40x1xf32>
    %817 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%816 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %818 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%815 : tensor<1x32x40x40xf32>) outs(%816 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %819 = tensor.empty() : tensor<1x32x40x40xf32>
    %820 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%815, %818 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%819 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %821 = tensor.empty() : tensor<1x32x40x1xf32>
    %822 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%821 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %823 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%820 : tensor<1x32x40x40xf32>) outs(%822 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %824 = tensor.empty() : tensor<1x32x40x40xf32>
    %825 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%820, %823 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%824 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %826 = tensor.empty() : tensor<1x32x40x40xbf16>
    %827 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%825 : tensor<1x32x40x40xf32>) outs(%826 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %828 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %829 = tosa.add %827, %828 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %830 = tosa.reshape %829 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %831 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %832 = tosa.add %773, %831 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %833 = tosa.reshape %832 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %834 = tosa.matmul %830, %833 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %835 = tosa.reshape %834 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %836 = tensor.empty() : tensor<1x40x32x128xbf16>
    %837 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%835 : tensor<1x32x40x128xbf16>) outs(%836 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %838 = tosa.identity %837 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %839 = tosa.reshape %838 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %840 = tensor.empty() : tensor<4096x4096xbf16>
    %841 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_142 : tensor<4096x4096xbf16>) outs(%840 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %842 = tosa.reshape %839 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_795 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %843 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%842, %841 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_795 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %844 = tosa.reshape %843 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %845 = tosa.add %736, %844 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %846 = tensor.empty() : tensor<1x40x4096xf32>
    %847 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%845 : tensor<1x40x4096xbf16>) outs(%846 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %848 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_796 = arith.constant 2 : i32
    %849 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%847 : tensor<1x40x4096xf32>) outs(%848 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_796 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_797 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %850 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%849 : tensor<1x40x4096xf32>) outs(%cst_797 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %851 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %852 = tosa.add %850, %851 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %853 = tosa.rsqrt %852 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %854 = tosa.mul %847, %853 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %855 = tensor.empty() : tensor<1x40x4096xbf16>
    %856 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%854 : tensor<1x40x4096xf32>) outs(%855 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %857 = tosa.reshape %extracted_slice_10 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %858 = tosa.mul %857, %856 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %859 = tensor.empty() : tensor<4096x11008xbf16>
    %860 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_144 : tensor<11008x4096xbf16>) outs(%859 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %861 = tosa.reshape %858 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_798 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %862 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%861, %860 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_798 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %863 = tosa.reshape %862 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %864 = tensor.empty() : tensor<1x40x11008xbf16>
    %865 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%863 : tensor<1x40x11008xbf16>) outs(%864 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %866 = tensor.empty() : tensor<4096x11008xbf16>
    %867 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_146 : tensor<11008x4096xbf16>) outs(%866 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %868 = tosa.reshape %858 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_799 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %869 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%868, %867 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_799 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %870 = tosa.reshape %869 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %871 = tosa.mul %865, %870 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %872 = tensor.empty() : tensor<11008x4096xbf16>
    %873 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_148 : tensor<4096x11008xbf16>) outs(%872 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %874 = tosa.reshape %871 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_800 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %875 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%874, %873 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_800 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %876 = tosa.reshape %875 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %877 = tosa.add %845, %876 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %878 = tensor.empty() : tensor<1x40x4096xf32>
    %879 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%877 : tensor<1x40x4096xbf16>) outs(%878 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %880 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_801 = arith.constant 2 : i32
    %881 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%879 : tensor<1x40x4096xf32>) outs(%880 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_801 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_802 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %882 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%881 : tensor<1x40x4096xf32>) outs(%cst_802 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %883 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %884 = tosa.add %882, %883 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %885 = tosa.rsqrt %884 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %886 = tosa.mul %879, %885 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %887 = tensor.empty() : tensor<1x40x4096xbf16>
    %888 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%886 : tensor<1x40x4096xf32>) outs(%887 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %889 = tosa.reshape %extracted_slice_11 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %890 = tosa.mul %889, %888 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %891 = tensor.empty() : tensor<4096x4096xbf16>
    %892 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_150 : tensor<4096x4096xbf16>) outs(%891 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %893 = tosa.reshape %890 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_803 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %894 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%893, %892 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_803 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %895 = tosa.reshape %894 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %896 = tensor.empty() : tensor<4096x4096xbf16>
    %897 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_152 : tensor<4096x4096xbf16>) outs(%896 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %898 = tosa.reshape %890 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_804 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %899 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%898, %897 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_804 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %900 = tosa.reshape %899 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %901 = tensor.empty() : tensor<4096x4096xbf16>
    %902 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_154 : tensor<4096x4096xbf16>) outs(%901 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %903 = tosa.reshape %890 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_805 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %904 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%903, %902 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_805 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %905 = tosa.reshape %904 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %906 = tosa.reshape %895 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %907 = tensor.empty() : tensor<1x32x40x128xbf16>
    %908 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%906 : tensor<1x40x32x128xbf16>) outs(%907 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %909 = tosa.reshape %900 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %910 = tensor.empty() : tensor<1x32x40x128xbf16>
    %911 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%909 : tensor<1x40x32x128xbf16>) outs(%910 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %912 = tosa.reshape %905 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %913 = tensor.empty() : tensor<1x32x40x128xbf16>
    %914 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%912 : tensor<1x40x32x128xbf16>) outs(%913 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_806 = tensor.extract_slice %expanded_540[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_807 = tensor.extract_slice %extracted_slice_806[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_808 = tensor.extract_slice %extracted_slice_807[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_809 = tensor.extract_slice %expanded_542[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_810 = tensor.extract_slice %extracted_slice_809[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_811 = tensor.extract_slice %extracted_slice_810[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %915 = tensor.empty() : tensor<1x40x128xbf16>
    %916  = tensor.collapse_shape %extracted_slice_808 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %917 = tensor.empty() : tensor<40x128xbf16>
    %918 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%916 : tensor<1x40x128xbf16>) outs(%917 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %919 = tensor.empty() : tensor<1x40x128xbf16>
    %920  = tensor.collapse_shape %extracted_slice_811 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %921 = tensor.empty() : tensor<40x128xbf16>
    %922 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%920 : tensor<1x40x128xbf16>) outs(%921 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %923 = tensor.empty() : tensor<1x40x128xbf16>
    %924 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%923 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %918[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %925 = tosa.reshape %924 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %926 = tensor.empty() : tensor<1x40x128xbf16>
    %927 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%926 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %922[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %928 = tosa.reshape %927 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %929 = tosa.mul %908, %925 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_812 = tensor.extract_slice %908[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_813 = tensor.extract_slice %908[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %930 = tensor.empty() : tensor<1x32x40x64xbf16>
    %931 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_813 : tensor<1x32x40x64xbf16>) outs(%930 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %932 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_814 = tensor.insert_slice %931 into %932[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_815 = tensor.insert_slice %extracted_slice_812 into %inserted_slice_814[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %933 = tosa.mul %inserted_slice_815, %928 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %934 = tosa.add %929, %933 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %935 = tosa.mul %911, %925 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_816 = tensor.extract_slice %911[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_817 = tensor.extract_slice %911[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %936 = tensor.empty() : tensor<1x32x40x64xbf16>
    %937 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_817 : tensor<1x32x40x64xbf16>) outs(%936 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %938 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_818 = tensor.insert_slice %937 into %938[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_819 = tensor.insert_slice %extracted_slice_816 into %inserted_slice_818[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %939 = tosa.mul %inserted_slice_819, %928 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %940 = tosa.add %935, %939 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %941 = tensor.empty() : tensor<1x32x128x40xbf16>
    %942 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%940 : tensor<1x32x40x128xbf16>) outs(%941 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %943 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %944 = tosa.add %934, %943 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %945 = tosa.reshape %944 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %946 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %947 = tosa.add %942, %946 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %948 = tosa.reshape %947 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %949 = tosa.matmul %945, %948 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %950 = tosa.reshape %949 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %951 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %952 = tosa.reciprocal %951 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %953 = tosa.mul %950, %952 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %954 = tosa.add %953, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %955 = tensor.empty() : tensor<1x32x40x40xf32>
    %956 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%954 : tensor<1x32x40x40xbf16>) outs(%955 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %957 = tensor.empty() : tensor<1x32x40x1xf32>
    %958 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%957 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %959 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%956 : tensor<1x32x40x40xf32>) outs(%957 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %960 = tensor.empty() : tensor<1x32x40x40xf32>
    %961 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%956, %959 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%960 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %962 = tensor.empty() : tensor<1x32x40x1xf32>
    %963 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%962 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %964 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%961 : tensor<1x32x40x40xf32>) outs(%963 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %965 = tensor.empty() : tensor<1x32x40x40xf32>
    %966 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%961, %964 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%965 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %967 = tensor.empty() : tensor<1x32x40x40xbf16>
    %968 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%966 : tensor<1x32x40x40xf32>) outs(%967 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %969 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %970 = tosa.add %968, %969 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %971 = tosa.reshape %970 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %972 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %973 = tosa.add %914, %972 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %974 = tosa.reshape %973 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %975 = tosa.matmul %971, %974 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %976 = tosa.reshape %975 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %977 = tensor.empty() : tensor<1x40x32x128xbf16>
    %978 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%976 : tensor<1x32x40x128xbf16>) outs(%977 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %979 = tosa.identity %978 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %980 = tosa.reshape %979 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %981 = tensor.empty() : tensor<4096x4096xbf16>
    %982 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_156 : tensor<4096x4096xbf16>) outs(%981 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %983 = tosa.reshape %980 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_820 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %984 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%983, %982 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_820 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %985 = tosa.reshape %984 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %986 = tosa.add %877, %985 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %987 = tensor.empty() : tensor<1x40x4096xf32>
    %988 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%986 : tensor<1x40x4096xbf16>) outs(%987 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %989 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_821 = arith.constant 2 : i32
    %990 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%988 : tensor<1x40x4096xf32>) outs(%989 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_821 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_822 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %991 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%990 : tensor<1x40x4096xf32>) outs(%cst_822 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %992 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %993 = tosa.add %991, %992 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %994 = tosa.rsqrt %993 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %995 = tosa.mul %988, %994 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %996 = tensor.empty() : tensor<1x40x4096xbf16>
    %997 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%995 : tensor<1x40x4096xf32>) outs(%996 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %998 = tosa.reshape %extracted_slice_12 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %999 = tosa.mul %998, %997 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1000 = tensor.empty() : tensor<4096x11008xbf16>
    %1001 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_158 : tensor<11008x4096xbf16>) outs(%1000 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1002 = tosa.reshape %999 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_823 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1003 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1002, %1001 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_823 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1004 = tosa.reshape %1003 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1005 = tensor.empty() : tensor<1x40x11008xbf16>
    %1006 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1004 : tensor<1x40x11008xbf16>) outs(%1005 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %1007 = tensor.empty() : tensor<4096x11008xbf16>
    %1008 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_160 : tensor<11008x4096xbf16>) outs(%1007 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1009 = tosa.reshape %999 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_824 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1010 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1009, %1008 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_824 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1011 = tosa.reshape %1010 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1012 = tosa.mul %1006, %1011 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1013 = tensor.empty() : tensor<11008x4096xbf16>
    %1014 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_162 : tensor<4096x11008xbf16>) outs(%1013 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %1015 = tosa.reshape %1012 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_825 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1016 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1015, %1014 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_825 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1017 = tosa.reshape %1016 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1018 = tosa.add %986, %1017 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1019 = tensor.empty() : tensor<1x40x4096xf32>
    %1020 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1018 : tensor<1x40x4096xbf16>) outs(%1019 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1021 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_826 = arith.constant 2 : i32
    %1022 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1020 : tensor<1x40x4096xf32>) outs(%1021 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_826 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_827 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1023 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1022 : tensor<1x40x4096xf32>) outs(%cst_827 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1024 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1025 = tosa.add %1023, %1024 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1026 = tosa.rsqrt %1025 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1027 = tosa.mul %1020, %1026 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1028 = tensor.empty() : tensor<1x40x4096xbf16>
    %1029 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1027 : tensor<1x40x4096xf32>) outs(%1028 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1030 = tosa.reshape %extracted_slice_13 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1031 = tosa.mul %1030, %1029 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1032 = tensor.empty() : tensor<4096x4096xbf16>
    %1033 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_164 : tensor<4096x4096xbf16>) outs(%1032 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1034 = tosa.reshape %1031 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_828 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1035 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1034, %1033 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_828 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1036 = tosa.reshape %1035 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1037 = tensor.empty() : tensor<4096x4096xbf16>
    %1038 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_166 : tensor<4096x4096xbf16>) outs(%1037 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1039 = tosa.reshape %1031 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_829 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1040 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1039, %1038 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_829 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1041 = tosa.reshape %1040 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1042 = tensor.empty() : tensor<4096x4096xbf16>
    %1043 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_168 : tensor<4096x4096xbf16>) outs(%1042 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1044 = tosa.reshape %1031 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_830 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1045 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1044, %1043 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_830 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1046 = tosa.reshape %1045 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1047 = tosa.reshape %1036 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1048 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1049 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1047 : tensor<1x40x32x128xbf16>) outs(%1048 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1050 = tosa.reshape %1041 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1051 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1052 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1050 : tensor<1x40x32x128xbf16>) outs(%1051 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1053 = tosa.reshape %1046 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1054 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1055 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1053 : tensor<1x40x32x128xbf16>) outs(%1054 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_831 = tensor.extract_slice %expanded_544[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_832 = tensor.extract_slice %extracted_slice_831[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_833 = tensor.extract_slice %extracted_slice_832[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_834 = tensor.extract_slice %expanded_546[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_835 = tensor.extract_slice %extracted_slice_834[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_836 = tensor.extract_slice %extracted_slice_835[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %1056 = tensor.empty() : tensor<1x40x128xbf16>
    %1057  = tensor.collapse_shape %extracted_slice_833 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1058 = tensor.empty() : tensor<40x128xbf16>
    %1059 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1057 : tensor<1x40x128xbf16>) outs(%1058 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1060 = tensor.empty() : tensor<1x40x128xbf16>
    %1061  = tensor.collapse_shape %extracted_slice_836 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1062 = tensor.empty() : tensor<40x128xbf16>
    %1063 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1061 : tensor<1x40x128xbf16>) outs(%1062 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1064 = tensor.empty() : tensor<1x40x128xbf16>
    %1065 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1064 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1059[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1066 = tosa.reshape %1065 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1067 = tensor.empty() : tensor<1x40x128xbf16>
    %1068 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1067 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1063[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1069 = tosa.reshape %1068 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1070 = tosa.mul %1049, %1066 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_837 = tensor.extract_slice %1049[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_838 = tensor.extract_slice %1049[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1071 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1072 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_838 : tensor<1x32x40x64xbf16>) outs(%1071 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1073 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_839 = tensor.insert_slice %1072 into %1073[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_840 = tensor.insert_slice %extracted_slice_837 into %inserted_slice_839[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1074 = tosa.mul %inserted_slice_840, %1069 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1075 = tosa.add %1070, %1074 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1076 = tosa.mul %1052, %1066 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_841 = tensor.extract_slice %1052[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_842 = tensor.extract_slice %1052[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1077 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1078 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_842 : tensor<1x32x40x64xbf16>) outs(%1077 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1079 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_843 = tensor.insert_slice %1078 into %1079[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_844 = tensor.insert_slice %extracted_slice_841 into %inserted_slice_843[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1080 = tosa.mul %inserted_slice_844, %1069 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1081 = tosa.add %1076, %1080 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1082 = tensor.empty() : tensor<1x32x128x40xbf16>
    %1083 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1081 : tensor<1x32x40x128xbf16>) outs(%1082 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %1084 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1085 = tosa.add %1075, %1084 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1086 = tosa.reshape %1085 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1087 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %1088 = tosa.add %1083, %1087 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %1089 = tosa.reshape %1088 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %1090 = tosa.matmul %1086, %1089 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %1091 = tosa.reshape %1090 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1092 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1093 = tosa.reciprocal %1092 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1094 = tosa.mul %1091, %1093 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1095 = tosa.add %1094, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1096 = tensor.empty() : tensor<1x32x40x40xf32>
    %1097 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1095 : tensor<1x32x40x40xbf16>) outs(%1096 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1098 = tensor.empty() : tensor<1x32x40x1xf32>
    %1099 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1098 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1100 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1097 : tensor<1x32x40x40xf32>) outs(%1098 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1101 = tensor.empty() : tensor<1x32x40x40xf32>
    %1102 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1097, %1100 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1101 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %1103 = tensor.empty() : tensor<1x32x40x1xf32>
    %1104 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1103 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1105 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1102 : tensor<1x32x40x40xf32>) outs(%1104 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1106 = tensor.empty() : tensor<1x32x40x40xf32>
    %1107 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1102, %1105 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1106 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1108 = tensor.empty() : tensor<1x32x40x40xbf16>
    %1109 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1107 : tensor<1x32x40x40xf32>) outs(%1108 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %1110 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1111 = tosa.add %1109, %1110 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1112 = tosa.reshape %1111 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %1113 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1114 = tosa.add %1055, %1113 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1115 = tosa.reshape %1114 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1116 = tosa.matmul %1112, %1115 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1117 = tosa.reshape %1116 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1118 = tensor.empty() : tensor<1x40x32x128xbf16>
    %1119 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1117 : tensor<1x32x40x128xbf16>) outs(%1118 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %1120 = tosa.identity %1119 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %1121 = tosa.reshape %1120 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %1122 = tensor.empty() : tensor<4096x4096xbf16>
    %1123 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_170 : tensor<4096x4096xbf16>) outs(%1122 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1124 = tosa.reshape %1121 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_845 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1125 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1124, %1123 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_845 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1126 = tosa.reshape %1125 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1127 = tosa.add %1018, %1126 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1128 = tensor.empty() : tensor<1x40x4096xf32>
    %1129 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1127 : tensor<1x40x4096xbf16>) outs(%1128 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1130 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_846 = arith.constant 2 : i32
    %1131 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1129 : tensor<1x40x4096xf32>) outs(%1130 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_846 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_847 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1132 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1131 : tensor<1x40x4096xf32>) outs(%cst_847 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1133 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1134 = tosa.add %1132, %1133 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1135 = tosa.rsqrt %1134 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1136 = tosa.mul %1129, %1135 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1137 = tensor.empty() : tensor<1x40x4096xbf16>
    %1138 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1136 : tensor<1x40x4096xf32>) outs(%1137 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1139 = tosa.reshape %extracted_slice_14 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1140 = tosa.mul %1139, %1138 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1141 = tensor.empty() : tensor<4096x11008xbf16>
    %1142 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_172 : tensor<11008x4096xbf16>) outs(%1141 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1143 = tosa.reshape %1140 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_848 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1144 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1143, %1142 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_848 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1145 = tosa.reshape %1144 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1146 = tensor.empty() : tensor<1x40x11008xbf16>
    %1147 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1145 : tensor<1x40x11008xbf16>) outs(%1146 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %1148 = tensor.empty() : tensor<4096x11008xbf16>
    %1149 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_174 : tensor<11008x4096xbf16>) outs(%1148 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1150 = tosa.reshape %1140 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_849 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1151 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1150, %1149 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_849 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1152 = tosa.reshape %1151 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1153 = tosa.mul %1147, %1152 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1154 = tensor.empty() : tensor<11008x4096xbf16>
    %1155 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_176 : tensor<4096x11008xbf16>) outs(%1154 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %1156 = tosa.reshape %1153 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_850 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1157 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1156, %1155 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_850 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1158 = tosa.reshape %1157 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1159 = tosa.add %1127, %1158 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1160 = tensor.empty() : tensor<1x40x4096xf32>
    %1161 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1159 : tensor<1x40x4096xbf16>) outs(%1160 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1162 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_851 = arith.constant 2 : i32
    %1163 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1161 : tensor<1x40x4096xf32>) outs(%1162 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_851 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_852 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1164 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1163 : tensor<1x40x4096xf32>) outs(%cst_852 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1165 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1166 = tosa.add %1164, %1165 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1167 = tosa.rsqrt %1166 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1168 = tosa.mul %1161, %1167 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1169 = tensor.empty() : tensor<1x40x4096xbf16>
    %1170 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1168 : tensor<1x40x4096xf32>) outs(%1169 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1171 = tosa.reshape %extracted_slice_15 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1172 = tosa.mul %1171, %1170 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1173 = tensor.empty() : tensor<4096x4096xbf16>
    %1174 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_178 : tensor<4096x4096xbf16>) outs(%1173 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1175 = tosa.reshape %1172 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_853 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1176 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1175, %1174 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_853 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1177 = tosa.reshape %1176 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1178 = tensor.empty() : tensor<4096x4096xbf16>
    %1179 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_180 : tensor<4096x4096xbf16>) outs(%1178 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1180 = tosa.reshape %1172 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_854 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1181 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1180, %1179 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_854 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1182 = tosa.reshape %1181 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1183 = tensor.empty() : tensor<4096x4096xbf16>
    %1184 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_182 : tensor<4096x4096xbf16>) outs(%1183 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1185 = tosa.reshape %1172 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_855 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1186 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1185, %1184 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_855 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1187 = tosa.reshape %1186 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1188 = tosa.reshape %1177 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1189 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1190 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1188 : tensor<1x40x32x128xbf16>) outs(%1189 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1191 = tosa.reshape %1182 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1192 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1193 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1191 : tensor<1x40x32x128xbf16>) outs(%1192 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1194 = tosa.reshape %1187 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1195 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1196 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1194 : tensor<1x40x32x128xbf16>) outs(%1195 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_856 = tensor.extract_slice %expanded_548[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_857 = tensor.extract_slice %extracted_slice_856[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_858 = tensor.extract_slice %extracted_slice_857[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_859 = tensor.extract_slice %expanded_550[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_860 = tensor.extract_slice %extracted_slice_859[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_861 = tensor.extract_slice %extracted_slice_860[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %1197 = tensor.empty() : tensor<1x40x128xbf16>
    %1198  = tensor.collapse_shape %extracted_slice_858 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1199 = tensor.empty() : tensor<40x128xbf16>
    %1200 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1198 : tensor<1x40x128xbf16>) outs(%1199 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1201 = tensor.empty() : tensor<1x40x128xbf16>
    %1202  = tensor.collapse_shape %extracted_slice_861 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1203 = tensor.empty() : tensor<40x128xbf16>
    %1204 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1202 : tensor<1x40x128xbf16>) outs(%1203 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1205 = tensor.empty() : tensor<1x40x128xbf16>
    %1206 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1205 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1200[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1207 = tosa.reshape %1206 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1208 = tensor.empty() : tensor<1x40x128xbf16>
    %1209 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1208 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1204[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1210 = tosa.reshape %1209 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1211 = tosa.mul %1190, %1207 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_862 = tensor.extract_slice %1190[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_863 = tensor.extract_slice %1190[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1212 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1213 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_863 : tensor<1x32x40x64xbf16>) outs(%1212 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1214 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_864 = tensor.insert_slice %1213 into %1214[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_865 = tensor.insert_slice %extracted_slice_862 into %inserted_slice_864[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1215 = tosa.mul %inserted_slice_865, %1210 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1216 = tosa.add %1211, %1215 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1217 = tosa.mul %1193, %1207 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_866 = tensor.extract_slice %1193[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_867 = tensor.extract_slice %1193[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1218 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1219 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_867 : tensor<1x32x40x64xbf16>) outs(%1218 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1220 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_868 = tensor.insert_slice %1219 into %1220[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_869 = tensor.insert_slice %extracted_slice_866 into %inserted_slice_868[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1221 = tosa.mul %inserted_slice_869, %1210 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1222 = tosa.add %1217, %1221 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1223 = tensor.empty() : tensor<1x32x128x40xbf16>
    %1224 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1222 : tensor<1x32x40x128xbf16>) outs(%1223 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %1225 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1226 = tosa.add %1216, %1225 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1227 = tosa.reshape %1226 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1228 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %1229 = tosa.add %1224, %1228 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %1230 = tosa.reshape %1229 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %1231 = tosa.matmul %1227, %1230 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %1232 = tosa.reshape %1231 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1233 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1234 = tosa.reciprocal %1233 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1235 = tosa.mul %1232, %1234 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1236 = tosa.add %1235, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1237 = tensor.empty() : tensor<1x32x40x40xf32>
    %1238 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1236 : tensor<1x32x40x40xbf16>) outs(%1237 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1239 = tensor.empty() : tensor<1x32x40x1xf32>
    %1240 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1239 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1241 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1238 : tensor<1x32x40x40xf32>) outs(%1239 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1242 = tensor.empty() : tensor<1x32x40x40xf32>
    %1243 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1238, %1241 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1242 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %1244 = tensor.empty() : tensor<1x32x40x1xf32>
    %1245 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1244 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1246 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1243 : tensor<1x32x40x40xf32>) outs(%1245 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1247 = tensor.empty() : tensor<1x32x40x40xf32>
    %1248 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1243, %1246 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1247 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1249 = tensor.empty() : tensor<1x32x40x40xbf16>
    %1250 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1248 : tensor<1x32x40x40xf32>) outs(%1249 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %1251 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1252 = tosa.add %1250, %1251 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1253 = tosa.reshape %1252 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %1254 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1255 = tosa.add %1196, %1254 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1256 = tosa.reshape %1255 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1257 = tosa.matmul %1253, %1256 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1258 = tosa.reshape %1257 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1259 = tensor.empty() : tensor<1x40x32x128xbf16>
    %1260 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1258 : tensor<1x32x40x128xbf16>) outs(%1259 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %1261 = tosa.identity %1260 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %1262 = tosa.reshape %1261 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %1263 = tensor.empty() : tensor<4096x4096xbf16>
    %1264 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_184 : tensor<4096x4096xbf16>) outs(%1263 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1265 = tosa.reshape %1262 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_870 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1266 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1265, %1264 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_870 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1267 = tosa.reshape %1266 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1268 = tosa.add %1159, %1267 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1269 = tensor.empty() : tensor<1x40x4096xf32>
    %1270 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1268 : tensor<1x40x4096xbf16>) outs(%1269 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1271 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_871 = arith.constant 2 : i32
    %1272 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1270 : tensor<1x40x4096xf32>) outs(%1271 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_871 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_872 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1273 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1272 : tensor<1x40x4096xf32>) outs(%cst_872 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1274 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1275 = tosa.add %1273, %1274 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1276 = tosa.rsqrt %1275 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1277 = tosa.mul %1270, %1276 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1278 = tensor.empty() : tensor<1x40x4096xbf16>
    %1279 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1277 : tensor<1x40x4096xf32>) outs(%1278 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1280 = tosa.reshape %extracted_slice_16 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1281 = tosa.mul %1280, %1279 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1282 = tensor.empty() : tensor<4096x11008xbf16>
    %1283 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_186 : tensor<11008x4096xbf16>) outs(%1282 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1284 = tosa.reshape %1281 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_873 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1285 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1284, %1283 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_873 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1286 = tosa.reshape %1285 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1287 = tensor.empty() : tensor<1x40x11008xbf16>
    %1288 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1286 : tensor<1x40x11008xbf16>) outs(%1287 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %1289 = tensor.empty() : tensor<4096x11008xbf16>
    %1290 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_188 : tensor<11008x4096xbf16>) outs(%1289 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1291 = tosa.reshape %1281 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_874 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1292 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1291, %1290 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_874 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1293 = tosa.reshape %1292 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1294 = tosa.mul %1288, %1293 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1295 = tensor.empty() : tensor<11008x4096xbf16>
    %1296 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_190 : tensor<4096x11008xbf16>) outs(%1295 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %1297 = tosa.reshape %1294 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_875 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1298 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1297, %1296 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_875 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1299 = tosa.reshape %1298 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1300 = tosa.add %1268, %1299 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1301 = tensor.empty() : tensor<1x40x4096xf32>
    %1302 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1300 : tensor<1x40x4096xbf16>) outs(%1301 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1303 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_876 = arith.constant 2 : i32
    %1304 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1302 : tensor<1x40x4096xf32>) outs(%1303 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_876 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_877 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1305 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1304 : tensor<1x40x4096xf32>) outs(%cst_877 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1306 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1307 = tosa.add %1305, %1306 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1308 = tosa.rsqrt %1307 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1309 = tosa.mul %1302, %1308 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1310 = tensor.empty() : tensor<1x40x4096xbf16>
    %1311 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1309 : tensor<1x40x4096xf32>) outs(%1310 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1312 = tosa.reshape %extracted_slice_17 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1313 = tosa.mul %1312, %1311 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1314 = tensor.empty() : tensor<4096x4096xbf16>
    %1315 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_192 : tensor<4096x4096xbf16>) outs(%1314 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1316 = tosa.reshape %1313 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_878 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1317 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1316, %1315 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_878 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1318 = tosa.reshape %1317 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1319 = tensor.empty() : tensor<4096x4096xbf16>
    %1320 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_194 : tensor<4096x4096xbf16>) outs(%1319 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1321 = tosa.reshape %1313 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_879 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1322 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1321, %1320 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_879 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1323 = tosa.reshape %1322 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1324 = tensor.empty() : tensor<4096x4096xbf16>
    %1325 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_196 : tensor<4096x4096xbf16>) outs(%1324 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1326 = tosa.reshape %1313 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_880 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1327 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1326, %1325 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_880 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1328 = tosa.reshape %1327 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1329 = tosa.reshape %1318 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1330 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1331 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1329 : tensor<1x40x32x128xbf16>) outs(%1330 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1332 = tosa.reshape %1323 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1333 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1334 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1332 : tensor<1x40x32x128xbf16>) outs(%1333 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1335 = tosa.reshape %1328 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1336 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1337 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1335 : tensor<1x40x32x128xbf16>) outs(%1336 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_881 = tensor.extract_slice %expanded_552[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_882 = tensor.extract_slice %extracted_slice_881[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_883 = tensor.extract_slice %extracted_slice_882[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_884 = tensor.extract_slice %expanded_554[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_885 = tensor.extract_slice %extracted_slice_884[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_886 = tensor.extract_slice %extracted_slice_885[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %1338 = tensor.empty() : tensor<1x40x128xbf16>
    %1339  = tensor.collapse_shape %extracted_slice_883 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1340 = tensor.empty() : tensor<40x128xbf16>
    %1341 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1339 : tensor<1x40x128xbf16>) outs(%1340 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1342 = tensor.empty() : tensor<1x40x128xbf16>
    %1343  = tensor.collapse_shape %extracted_slice_886 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1344 = tensor.empty() : tensor<40x128xbf16>
    %1345 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1343 : tensor<1x40x128xbf16>) outs(%1344 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1346 = tensor.empty() : tensor<1x40x128xbf16>
    %1347 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1346 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1341[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1348 = tosa.reshape %1347 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1349 = tensor.empty() : tensor<1x40x128xbf16>
    %1350 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1349 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1345[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1351 = tosa.reshape %1350 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1352 = tosa.mul %1331, %1348 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_887 = tensor.extract_slice %1331[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_888 = tensor.extract_slice %1331[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1353 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1354 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_888 : tensor<1x32x40x64xbf16>) outs(%1353 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1355 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_889 = tensor.insert_slice %1354 into %1355[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_890 = tensor.insert_slice %extracted_slice_887 into %inserted_slice_889[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1356 = tosa.mul %inserted_slice_890, %1351 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1357 = tosa.add %1352, %1356 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1358 = tosa.mul %1334, %1348 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_891 = tensor.extract_slice %1334[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_892 = tensor.extract_slice %1334[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1359 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1360 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_892 : tensor<1x32x40x64xbf16>) outs(%1359 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1361 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_893 = tensor.insert_slice %1360 into %1361[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_894 = tensor.insert_slice %extracted_slice_891 into %inserted_slice_893[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1362 = tosa.mul %inserted_slice_894, %1351 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1363 = tosa.add %1358, %1362 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1364 = tensor.empty() : tensor<1x32x128x40xbf16>
    %1365 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1363 : tensor<1x32x40x128xbf16>) outs(%1364 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %1366 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1367 = tosa.add %1357, %1366 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1368 = tosa.reshape %1367 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1369 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %1370 = tosa.add %1365, %1369 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %1371 = tosa.reshape %1370 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %1372 = tosa.matmul %1368, %1371 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %1373 = tosa.reshape %1372 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1374 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1375 = tosa.reciprocal %1374 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1376 = tosa.mul %1373, %1375 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1377 = tosa.add %1376, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1378 = tensor.empty() : tensor<1x32x40x40xf32>
    %1379 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1377 : tensor<1x32x40x40xbf16>) outs(%1378 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1380 = tensor.empty() : tensor<1x32x40x1xf32>
    %1381 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1380 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1382 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1379 : tensor<1x32x40x40xf32>) outs(%1380 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1383 = tensor.empty() : tensor<1x32x40x40xf32>
    %1384 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1379, %1382 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1383 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %1385 = tensor.empty() : tensor<1x32x40x1xf32>
    %1386 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1385 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1387 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1384 : tensor<1x32x40x40xf32>) outs(%1386 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1388 = tensor.empty() : tensor<1x32x40x40xf32>
    %1389 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1384, %1387 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1388 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1390 = tensor.empty() : tensor<1x32x40x40xbf16>
    %1391 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1389 : tensor<1x32x40x40xf32>) outs(%1390 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %1392 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1393 = tosa.add %1391, %1392 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1394 = tosa.reshape %1393 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %1395 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1396 = tosa.add %1337, %1395 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1397 = tosa.reshape %1396 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1398 = tosa.matmul %1394, %1397 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1399 = tosa.reshape %1398 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1400 = tensor.empty() : tensor<1x40x32x128xbf16>
    %1401 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1399 : tensor<1x32x40x128xbf16>) outs(%1400 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %1402 = tosa.identity %1401 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %1403 = tosa.reshape %1402 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %1404 = tensor.empty() : tensor<4096x4096xbf16>
    %1405 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_198 : tensor<4096x4096xbf16>) outs(%1404 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1406 = tosa.reshape %1403 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_895 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1407 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1406, %1405 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_895 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1408 = tosa.reshape %1407 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1409 = tosa.add %1300, %1408 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1410 = tensor.empty() : tensor<1x40x4096xf32>
    %1411 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1409 : tensor<1x40x4096xbf16>) outs(%1410 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1412 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_896 = arith.constant 2 : i32
    %1413 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1411 : tensor<1x40x4096xf32>) outs(%1412 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_896 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_897 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1414 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1413 : tensor<1x40x4096xf32>) outs(%cst_897 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1415 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1416 = tosa.add %1414, %1415 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1417 = tosa.rsqrt %1416 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1418 = tosa.mul %1411, %1417 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1419 = tensor.empty() : tensor<1x40x4096xbf16>
    %1420 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1418 : tensor<1x40x4096xf32>) outs(%1419 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1421 = tosa.reshape %extracted_slice_18 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1422 = tosa.mul %1421, %1420 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1423 = tensor.empty() : tensor<4096x11008xbf16>
    %1424 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_200 : tensor<11008x4096xbf16>) outs(%1423 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1425 = tosa.reshape %1422 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_898 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1426 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1425, %1424 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_898 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1427 = tosa.reshape %1426 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1428 = tensor.empty() : tensor<1x40x11008xbf16>
    %1429 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1427 : tensor<1x40x11008xbf16>) outs(%1428 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %1430 = tensor.empty() : tensor<4096x11008xbf16>
    %1431 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_202 : tensor<11008x4096xbf16>) outs(%1430 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1432 = tosa.reshape %1422 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_899 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1433 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1432, %1431 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_899 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1434 = tosa.reshape %1433 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1435 = tosa.mul %1429, %1434 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1436 = tensor.empty() : tensor<11008x4096xbf16>
    %1437 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_204 : tensor<4096x11008xbf16>) outs(%1436 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %1438 = tosa.reshape %1435 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_900 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1439 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1438, %1437 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_900 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1440 = tosa.reshape %1439 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1441 = tosa.add %1409, %1440 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1442 = tensor.empty() : tensor<1x40x4096xf32>
    %1443 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1441 : tensor<1x40x4096xbf16>) outs(%1442 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1444 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_901 = arith.constant 2 : i32
    %1445 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1443 : tensor<1x40x4096xf32>) outs(%1444 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_901 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_902 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1446 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1445 : tensor<1x40x4096xf32>) outs(%cst_902 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1447 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1448 = tosa.add %1446, %1447 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1449 = tosa.rsqrt %1448 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1450 = tosa.mul %1443, %1449 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1451 = tensor.empty() : tensor<1x40x4096xbf16>
    %1452 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1450 : tensor<1x40x4096xf32>) outs(%1451 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1453 = tosa.reshape %extracted_slice_19 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1454 = tosa.mul %1453, %1452 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1455 = tensor.empty() : tensor<4096x4096xbf16>
    %1456 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_206 : tensor<4096x4096xbf16>) outs(%1455 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1457 = tosa.reshape %1454 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_903 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1458 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1457, %1456 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_903 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1459 = tosa.reshape %1458 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1460 = tensor.empty() : tensor<4096x4096xbf16>
    %1461 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_208 : tensor<4096x4096xbf16>) outs(%1460 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1462 = tosa.reshape %1454 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_904 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1463 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1462, %1461 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_904 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1464 = tosa.reshape %1463 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1465 = tensor.empty() : tensor<4096x4096xbf16>
    %1466 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_210 : tensor<4096x4096xbf16>) outs(%1465 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1467 = tosa.reshape %1454 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_905 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1468 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1467, %1466 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_905 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1469 = tosa.reshape %1468 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1470 = tosa.reshape %1459 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1471 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1472 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1470 : tensor<1x40x32x128xbf16>) outs(%1471 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1473 = tosa.reshape %1464 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1474 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1475 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1473 : tensor<1x40x32x128xbf16>) outs(%1474 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1476 = tosa.reshape %1469 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1477 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1478 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1476 : tensor<1x40x32x128xbf16>) outs(%1477 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_906 = tensor.extract_slice %expanded_556[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_907 = tensor.extract_slice %extracted_slice_906[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_908 = tensor.extract_slice %extracted_slice_907[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_909 = tensor.extract_slice %expanded_558[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_910 = tensor.extract_slice %extracted_slice_909[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_911 = tensor.extract_slice %extracted_slice_910[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %1479 = tensor.empty() : tensor<1x40x128xbf16>
    %1480  = tensor.collapse_shape %extracted_slice_908 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1481 = tensor.empty() : tensor<40x128xbf16>
    %1482 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1480 : tensor<1x40x128xbf16>) outs(%1481 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1483 = tensor.empty() : tensor<1x40x128xbf16>
    %1484  = tensor.collapse_shape %extracted_slice_911 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1485 = tensor.empty() : tensor<40x128xbf16>
    %1486 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1484 : tensor<1x40x128xbf16>) outs(%1485 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1487 = tensor.empty() : tensor<1x40x128xbf16>
    %1488 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1487 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1482[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1489 = tosa.reshape %1488 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1490 = tensor.empty() : tensor<1x40x128xbf16>
    %1491 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1490 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1486[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1492 = tosa.reshape %1491 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1493 = tosa.mul %1472, %1489 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_912 = tensor.extract_slice %1472[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_913 = tensor.extract_slice %1472[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1494 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1495 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_913 : tensor<1x32x40x64xbf16>) outs(%1494 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1496 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_914 = tensor.insert_slice %1495 into %1496[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_915 = tensor.insert_slice %extracted_slice_912 into %inserted_slice_914[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1497 = tosa.mul %inserted_slice_915, %1492 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1498 = tosa.add %1493, %1497 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1499 = tosa.mul %1475, %1489 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_916 = tensor.extract_slice %1475[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_917 = tensor.extract_slice %1475[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1500 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1501 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_917 : tensor<1x32x40x64xbf16>) outs(%1500 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1502 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_918 = tensor.insert_slice %1501 into %1502[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_919 = tensor.insert_slice %extracted_slice_916 into %inserted_slice_918[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1503 = tosa.mul %inserted_slice_919, %1492 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1504 = tosa.add %1499, %1503 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1505 = tensor.empty() : tensor<1x32x128x40xbf16>
    %1506 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1504 : tensor<1x32x40x128xbf16>) outs(%1505 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %1507 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1508 = tosa.add %1498, %1507 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1509 = tosa.reshape %1508 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1510 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %1511 = tosa.add %1506, %1510 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %1512 = tosa.reshape %1511 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %1513 = tosa.matmul %1509, %1512 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %1514 = tosa.reshape %1513 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1515 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1516 = tosa.reciprocal %1515 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1517 = tosa.mul %1514, %1516 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1518 = tosa.add %1517, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1519 = tensor.empty() : tensor<1x32x40x40xf32>
    %1520 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1518 : tensor<1x32x40x40xbf16>) outs(%1519 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1521 = tensor.empty() : tensor<1x32x40x1xf32>
    %1522 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1521 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1523 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1520 : tensor<1x32x40x40xf32>) outs(%1521 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1524 = tensor.empty() : tensor<1x32x40x40xf32>
    %1525 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1520, %1523 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1524 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %1526 = tensor.empty() : tensor<1x32x40x1xf32>
    %1527 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1526 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1528 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1525 : tensor<1x32x40x40xf32>) outs(%1527 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1529 = tensor.empty() : tensor<1x32x40x40xf32>
    %1530 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1525, %1528 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1529 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1531 = tensor.empty() : tensor<1x32x40x40xbf16>
    %1532 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1530 : tensor<1x32x40x40xf32>) outs(%1531 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %1533 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1534 = tosa.add %1532, %1533 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1535 = tosa.reshape %1534 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %1536 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1537 = tosa.add %1478, %1536 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1538 = tosa.reshape %1537 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1539 = tosa.matmul %1535, %1538 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1540 = tosa.reshape %1539 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1541 = tensor.empty() : tensor<1x40x32x128xbf16>
    %1542 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1540 : tensor<1x32x40x128xbf16>) outs(%1541 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %1543 = tosa.identity %1542 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %1544 = tosa.reshape %1543 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %1545 = tensor.empty() : tensor<4096x4096xbf16>
    %1546 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_212 : tensor<4096x4096xbf16>) outs(%1545 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1547 = tosa.reshape %1544 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_920 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1548 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1547, %1546 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_920 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1549 = tosa.reshape %1548 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1550 = tosa.add %1441, %1549 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1551 = tensor.empty() : tensor<1x40x4096xf32>
    %1552 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1550 : tensor<1x40x4096xbf16>) outs(%1551 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1553 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_921 = arith.constant 2 : i32
    %1554 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1552 : tensor<1x40x4096xf32>) outs(%1553 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_921 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_922 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1555 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1554 : tensor<1x40x4096xf32>) outs(%cst_922 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1556 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1557 = tosa.add %1555, %1556 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1558 = tosa.rsqrt %1557 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1559 = tosa.mul %1552, %1558 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1560 = tensor.empty() : tensor<1x40x4096xbf16>
    %1561 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1559 : tensor<1x40x4096xf32>) outs(%1560 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1562 = tosa.reshape %extracted_slice_20 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1563 = tosa.mul %1562, %1561 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1564 = tensor.empty() : tensor<4096x11008xbf16>
    %1565 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_214 : tensor<11008x4096xbf16>) outs(%1564 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1566 = tosa.reshape %1563 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_923 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1567 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1566, %1565 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_923 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1568 = tosa.reshape %1567 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1569 = tensor.empty() : tensor<1x40x11008xbf16>
    %1570 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1568 : tensor<1x40x11008xbf16>) outs(%1569 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %1571 = tensor.empty() : tensor<4096x11008xbf16>
    %1572 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_216 : tensor<11008x4096xbf16>) outs(%1571 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1573 = tosa.reshape %1563 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_924 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1574 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1573, %1572 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_924 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1575 = tosa.reshape %1574 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1576 = tosa.mul %1570, %1575 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1577 = tensor.empty() : tensor<11008x4096xbf16>
    %1578 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_218 : tensor<4096x11008xbf16>) outs(%1577 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %1579 = tosa.reshape %1576 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_925 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1580 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1579, %1578 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_925 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1581 = tosa.reshape %1580 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1582 = tosa.add %1550, %1581 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1583 = tensor.empty() : tensor<1x40x4096xf32>
    %1584 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1582 : tensor<1x40x4096xbf16>) outs(%1583 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1585 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_926 = arith.constant 2 : i32
    %1586 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1584 : tensor<1x40x4096xf32>) outs(%1585 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_926 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_927 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1587 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1586 : tensor<1x40x4096xf32>) outs(%cst_927 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1588 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1589 = tosa.add %1587, %1588 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1590 = tosa.rsqrt %1589 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1591 = tosa.mul %1584, %1590 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1592 = tensor.empty() : tensor<1x40x4096xbf16>
    %1593 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1591 : tensor<1x40x4096xf32>) outs(%1592 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1594 = tosa.reshape %extracted_slice_21 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1595 = tosa.mul %1594, %1593 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1596 = tensor.empty() : tensor<4096x4096xbf16>
    %1597 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_220 : tensor<4096x4096xbf16>) outs(%1596 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1598 = tosa.reshape %1595 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_928 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1599 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1598, %1597 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_928 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1600 = tosa.reshape %1599 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1601 = tensor.empty() : tensor<4096x4096xbf16>
    %1602 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_222 : tensor<4096x4096xbf16>) outs(%1601 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1603 = tosa.reshape %1595 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_929 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1604 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1603, %1602 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_929 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1605 = tosa.reshape %1604 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1606 = tensor.empty() : tensor<4096x4096xbf16>
    %1607 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_224 : tensor<4096x4096xbf16>) outs(%1606 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1608 = tosa.reshape %1595 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_930 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1609 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1608, %1607 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_930 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1610 = tosa.reshape %1609 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1611 = tosa.reshape %1600 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1612 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1613 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1611 : tensor<1x40x32x128xbf16>) outs(%1612 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1614 = tosa.reshape %1605 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1615 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1616 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1614 : tensor<1x40x32x128xbf16>) outs(%1615 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1617 = tosa.reshape %1610 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1618 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1619 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1617 : tensor<1x40x32x128xbf16>) outs(%1618 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_931 = tensor.extract_slice %expanded_560[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_932 = tensor.extract_slice %extracted_slice_931[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_933 = tensor.extract_slice %extracted_slice_932[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_934 = tensor.extract_slice %expanded_562[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_935 = tensor.extract_slice %extracted_slice_934[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_936 = tensor.extract_slice %extracted_slice_935[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %1620 = tensor.empty() : tensor<1x40x128xbf16>
    %1621  = tensor.collapse_shape %extracted_slice_933 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1622 = tensor.empty() : tensor<40x128xbf16>
    %1623 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1621 : tensor<1x40x128xbf16>) outs(%1622 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1624 = tensor.empty() : tensor<1x40x128xbf16>
    %1625  = tensor.collapse_shape %extracted_slice_936 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1626 = tensor.empty() : tensor<40x128xbf16>
    %1627 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1625 : tensor<1x40x128xbf16>) outs(%1626 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1628 = tensor.empty() : tensor<1x40x128xbf16>
    %1629 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1628 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1623[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1630 = tosa.reshape %1629 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1631 = tensor.empty() : tensor<1x40x128xbf16>
    %1632 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1631 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1627[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1633 = tosa.reshape %1632 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1634 = tosa.mul %1613, %1630 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_937 = tensor.extract_slice %1613[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_938 = tensor.extract_slice %1613[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1635 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1636 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_938 : tensor<1x32x40x64xbf16>) outs(%1635 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1637 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_939 = tensor.insert_slice %1636 into %1637[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_940 = tensor.insert_slice %extracted_slice_937 into %inserted_slice_939[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1638 = tosa.mul %inserted_slice_940, %1633 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1639 = tosa.add %1634, %1638 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1640 = tosa.mul %1616, %1630 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_941 = tensor.extract_slice %1616[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_942 = tensor.extract_slice %1616[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1641 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1642 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_942 : tensor<1x32x40x64xbf16>) outs(%1641 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1643 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_943 = tensor.insert_slice %1642 into %1643[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_944 = tensor.insert_slice %extracted_slice_941 into %inserted_slice_943[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1644 = tosa.mul %inserted_slice_944, %1633 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1645 = tosa.add %1640, %1644 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1646 = tensor.empty() : tensor<1x32x128x40xbf16>
    %1647 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1645 : tensor<1x32x40x128xbf16>) outs(%1646 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %1648 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1649 = tosa.add %1639, %1648 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1650 = tosa.reshape %1649 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1651 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %1652 = tosa.add %1647, %1651 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %1653 = tosa.reshape %1652 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %1654 = tosa.matmul %1650, %1653 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %1655 = tosa.reshape %1654 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1656 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1657 = tosa.reciprocal %1656 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1658 = tosa.mul %1655, %1657 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1659 = tosa.add %1658, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1660 = tensor.empty() : tensor<1x32x40x40xf32>
    %1661 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1659 : tensor<1x32x40x40xbf16>) outs(%1660 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1662 = tensor.empty() : tensor<1x32x40x1xf32>
    %1663 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1662 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1664 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1661 : tensor<1x32x40x40xf32>) outs(%1662 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1665 = tensor.empty() : tensor<1x32x40x40xf32>
    %1666 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1661, %1664 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1665 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %1667 = tensor.empty() : tensor<1x32x40x1xf32>
    %1668 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1667 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1669 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1666 : tensor<1x32x40x40xf32>) outs(%1668 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1670 = tensor.empty() : tensor<1x32x40x40xf32>
    %1671 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1666, %1669 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1670 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1672 = tensor.empty() : tensor<1x32x40x40xbf16>
    %1673 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1671 : tensor<1x32x40x40xf32>) outs(%1672 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %1674 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1675 = tosa.add %1673, %1674 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1676 = tosa.reshape %1675 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %1677 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1678 = tosa.add %1619, %1677 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1679 = tosa.reshape %1678 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1680 = tosa.matmul %1676, %1679 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1681 = tosa.reshape %1680 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1682 = tensor.empty() : tensor<1x40x32x128xbf16>
    %1683 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1681 : tensor<1x32x40x128xbf16>) outs(%1682 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %1684 = tosa.identity %1683 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %1685 = tosa.reshape %1684 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %1686 = tensor.empty() : tensor<4096x4096xbf16>
    %1687 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_226 : tensor<4096x4096xbf16>) outs(%1686 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1688 = tosa.reshape %1685 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_945 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1689 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1688, %1687 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_945 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1690 = tosa.reshape %1689 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1691 = tosa.add %1582, %1690 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1692 = tensor.empty() : tensor<1x40x4096xf32>
    %1693 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1691 : tensor<1x40x4096xbf16>) outs(%1692 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1694 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_946 = arith.constant 2 : i32
    %1695 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1693 : tensor<1x40x4096xf32>) outs(%1694 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_946 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_947 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1696 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1695 : tensor<1x40x4096xf32>) outs(%cst_947 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1697 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1698 = tosa.add %1696, %1697 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1699 = tosa.rsqrt %1698 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1700 = tosa.mul %1693, %1699 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1701 = tensor.empty() : tensor<1x40x4096xbf16>
    %1702 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1700 : tensor<1x40x4096xf32>) outs(%1701 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1703 = tosa.reshape %extracted_slice_22 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1704 = tosa.mul %1703, %1702 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1705 = tensor.empty() : tensor<4096x11008xbf16>
    %1706 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_228 : tensor<11008x4096xbf16>) outs(%1705 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1707 = tosa.reshape %1704 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_948 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1708 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1707, %1706 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_948 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1709 = tosa.reshape %1708 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1710 = tensor.empty() : tensor<1x40x11008xbf16>
    %1711 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1709 : tensor<1x40x11008xbf16>) outs(%1710 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %1712 = tensor.empty() : tensor<4096x11008xbf16>
    %1713 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_230 : tensor<11008x4096xbf16>) outs(%1712 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1714 = tosa.reshape %1704 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_949 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1715 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1714, %1713 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_949 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1716 = tosa.reshape %1715 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1717 = tosa.mul %1711, %1716 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1718 = tensor.empty() : tensor<11008x4096xbf16>
    %1719 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_232 : tensor<4096x11008xbf16>) outs(%1718 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %1720 = tosa.reshape %1717 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_950 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1721 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1720, %1719 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_950 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1722 = tosa.reshape %1721 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1723 = tosa.add %1691, %1722 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1724 = tensor.empty() : tensor<1x40x4096xf32>
    %1725 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1723 : tensor<1x40x4096xbf16>) outs(%1724 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1726 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_951 = arith.constant 2 : i32
    %1727 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1725 : tensor<1x40x4096xf32>) outs(%1726 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_951 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_952 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1728 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1727 : tensor<1x40x4096xf32>) outs(%cst_952 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1729 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1730 = tosa.add %1728, %1729 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1731 = tosa.rsqrt %1730 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1732 = tosa.mul %1725, %1731 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1733 = tensor.empty() : tensor<1x40x4096xbf16>
    %1734 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1732 : tensor<1x40x4096xf32>) outs(%1733 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1735 = tosa.reshape %extracted_slice_23 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1736 = tosa.mul %1735, %1734 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1737 = tensor.empty() : tensor<4096x4096xbf16>
    %1738 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_234 : tensor<4096x4096xbf16>) outs(%1737 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1739 = tosa.reshape %1736 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_953 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1740 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1739, %1738 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_953 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1741 = tosa.reshape %1740 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1742 = tensor.empty() : tensor<4096x4096xbf16>
    %1743 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_236 : tensor<4096x4096xbf16>) outs(%1742 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1744 = tosa.reshape %1736 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_954 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1745 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1744, %1743 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_954 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1746 = tosa.reshape %1745 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1747 = tensor.empty() : tensor<4096x4096xbf16>
    %1748 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_238 : tensor<4096x4096xbf16>) outs(%1747 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1749 = tosa.reshape %1736 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_955 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1750 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1749, %1748 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_955 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1751 = tosa.reshape %1750 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1752 = tosa.reshape %1741 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1753 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1754 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1752 : tensor<1x40x32x128xbf16>) outs(%1753 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1755 = tosa.reshape %1746 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1756 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1757 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1755 : tensor<1x40x32x128xbf16>) outs(%1756 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1758 = tosa.reshape %1751 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1759 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1760 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1758 : tensor<1x40x32x128xbf16>) outs(%1759 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_956 = tensor.extract_slice %expanded_564[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_957 = tensor.extract_slice %extracted_slice_956[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_958 = tensor.extract_slice %extracted_slice_957[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_959 = tensor.extract_slice %expanded_566[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_960 = tensor.extract_slice %extracted_slice_959[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_961 = tensor.extract_slice %extracted_slice_960[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %1761 = tensor.empty() : tensor<1x40x128xbf16>
    %1762  = tensor.collapse_shape %extracted_slice_958 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1763 = tensor.empty() : tensor<40x128xbf16>
    %1764 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1762 : tensor<1x40x128xbf16>) outs(%1763 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1765 = tensor.empty() : tensor<1x40x128xbf16>
    %1766  = tensor.collapse_shape %extracted_slice_961 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1767 = tensor.empty() : tensor<40x128xbf16>
    %1768 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1766 : tensor<1x40x128xbf16>) outs(%1767 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1769 = tensor.empty() : tensor<1x40x128xbf16>
    %1770 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1769 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1764[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1771 = tosa.reshape %1770 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1772 = tensor.empty() : tensor<1x40x128xbf16>
    %1773 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1772 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1768[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1774 = tosa.reshape %1773 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1775 = tosa.mul %1754, %1771 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_962 = tensor.extract_slice %1754[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_963 = tensor.extract_slice %1754[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1776 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1777 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_963 : tensor<1x32x40x64xbf16>) outs(%1776 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1778 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_964 = tensor.insert_slice %1777 into %1778[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_965 = tensor.insert_slice %extracted_slice_962 into %inserted_slice_964[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1779 = tosa.mul %inserted_slice_965, %1774 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1780 = tosa.add %1775, %1779 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1781 = tosa.mul %1757, %1771 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_966 = tensor.extract_slice %1757[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_967 = tensor.extract_slice %1757[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1782 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1783 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_967 : tensor<1x32x40x64xbf16>) outs(%1782 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1784 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_968 = tensor.insert_slice %1783 into %1784[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_969 = tensor.insert_slice %extracted_slice_966 into %inserted_slice_968[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1785 = tosa.mul %inserted_slice_969, %1774 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1786 = tosa.add %1781, %1785 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1787 = tensor.empty() : tensor<1x32x128x40xbf16>
    %1788 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1786 : tensor<1x32x40x128xbf16>) outs(%1787 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %1789 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1790 = tosa.add %1780, %1789 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1791 = tosa.reshape %1790 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1792 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %1793 = tosa.add %1788, %1792 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %1794 = tosa.reshape %1793 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %1795 = tosa.matmul %1791, %1794 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %1796 = tosa.reshape %1795 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1797 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1798 = tosa.reciprocal %1797 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1799 = tosa.mul %1796, %1798 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1800 = tosa.add %1799, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1801 = tensor.empty() : tensor<1x32x40x40xf32>
    %1802 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1800 : tensor<1x32x40x40xbf16>) outs(%1801 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1803 = tensor.empty() : tensor<1x32x40x1xf32>
    %1804 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1803 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1805 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1802 : tensor<1x32x40x40xf32>) outs(%1803 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1806 = tensor.empty() : tensor<1x32x40x40xf32>
    %1807 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1802, %1805 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1806 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %1808 = tensor.empty() : tensor<1x32x40x1xf32>
    %1809 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1808 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1810 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1807 : tensor<1x32x40x40xf32>) outs(%1809 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1811 = tensor.empty() : tensor<1x32x40x40xf32>
    %1812 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1807, %1810 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1811 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1813 = tensor.empty() : tensor<1x32x40x40xbf16>
    %1814 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1812 : tensor<1x32x40x40xf32>) outs(%1813 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %1815 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1816 = tosa.add %1814, %1815 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1817 = tosa.reshape %1816 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %1818 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1819 = tosa.add %1760, %1818 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1820 = tosa.reshape %1819 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1821 = tosa.matmul %1817, %1820 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1822 = tosa.reshape %1821 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1823 = tensor.empty() : tensor<1x40x32x128xbf16>
    %1824 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1822 : tensor<1x32x40x128xbf16>) outs(%1823 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %1825 = tosa.identity %1824 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %1826 = tosa.reshape %1825 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %1827 = tensor.empty() : tensor<4096x4096xbf16>
    %1828 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_240 : tensor<4096x4096xbf16>) outs(%1827 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1829 = tosa.reshape %1826 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_970 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1830 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1829, %1828 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_970 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1831 = tosa.reshape %1830 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1832 = tosa.add %1723, %1831 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1833 = tensor.empty() : tensor<1x40x4096xf32>
    %1834 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1832 : tensor<1x40x4096xbf16>) outs(%1833 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1835 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_971 = arith.constant 2 : i32
    %1836 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1834 : tensor<1x40x4096xf32>) outs(%1835 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_971 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_972 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1837 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1836 : tensor<1x40x4096xf32>) outs(%cst_972 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1838 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1839 = tosa.add %1837, %1838 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1840 = tosa.rsqrt %1839 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1841 = tosa.mul %1834, %1840 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1842 = tensor.empty() : tensor<1x40x4096xbf16>
    %1843 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1841 : tensor<1x40x4096xf32>) outs(%1842 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1844 = tosa.reshape %extracted_slice_24 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1845 = tosa.mul %1844, %1843 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1846 = tensor.empty() : tensor<4096x11008xbf16>
    %1847 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_242 : tensor<11008x4096xbf16>) outs(%1846 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1848 = tosa.reshape %1845 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_973 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1849 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1848, %1847 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_973 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1850 = tosa.reshape %1849 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1851 = tensor.empty() : tensor<1x40x11008xbf16>
    %1852 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1850 : tensor<1x40x11008xbf16>) outs(%1851 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %1853 = tensor.empty() : tensor<4096x11008xbf16>
    %1854 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_244 : tensor<11008x4096xbf16>) outs(%1853 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1855 = tosa.reshape %1845 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_974 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1856 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1855, %1854 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_974 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1857 = tosa.reshape %1856 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1858 = tosa.mul %1852, %1857 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1859 = tensor.empty() : tensor<11008x4096xbf16>
    %1860 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_246 : tensor<4096x11008xbf16>) outs(%1859 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %1861 = tosa.reshape %1858 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_975 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1862 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1861, %1860 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_975 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1863 = tosa.reshape %1862 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1864 = tosa.add %1832, %1863 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1865 = tensor.empty() : tensor<1x40x4096xf32>
    %1866 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1864 : tensor<1x40x4096xbf16>) outs(%1865 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1867 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_976 = arith.constant 2 : i32
    %1868 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1866 : tensor<1x40x4096xf32>) outs(%1867 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_976 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_977 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1869 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1868 : tensor<1x40x4096xf32>) outs(%cst_977 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1870 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1871 = tosa.add %1869, %1870 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1872 = tosa.rsqrt %1871 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1873 = tosa.mul %1866, %1872 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1874 = tensor.empty() : tensor<1x40x4096xbf16>
    %1875 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1873 : tensor<1x40x4096xf32>) outs(%1874 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1876 = tosa.reshape %extracted_slice_25 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1877 = tosa.mul %1876, %1875 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1878 = tensor.empty() : tensor<4096x4096xbf16>
    %1879 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_248 : tensor<4096x4096xbf16>) outs(%1878 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1880 = tosa.reshape %1877 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_978 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1881 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1880, %1879 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_978 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1882 = tosa.reshape %1881 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1883 = tensor.empty() : tensor<4096x4096xbf16>
    %1884 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_250 : tensor<4096x4096xbf16>) outs(%1883 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1885 = tosa.reshape %1877 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_979 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1886 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1885, %1884 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_979 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1887 = tosa.reshape %1886 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1888 = tensor.empty() : tensor<4096x4096xbf16>
    %1889 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_252 : tensor<4096x4096xbf16>) outs(%1888 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1890 = tosa.reshape %1877 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_980 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1891 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1890, %1889 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_980 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1892 = tosa.reshape %1891 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1893 = tosa.reshape %1882 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1894 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1895 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1893 : tensor<1x40x32x128xbf16>) outs(%1894 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1896 = tosa.reshape %1887 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1897 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1898 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1896 : tensor<1x40x32x128xbf16>) outs(%1897 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %1899 = tosa.reshape %1892 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %1900 = tensor.empty() : tensor<1x32x40x128xbf16>
    %1901 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1899 : tensor<1x40x32x128xbf16>) outs(%1900 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_981 = tensor.extract_slice %expanded_568[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_982 = tensor.extract_slice %extracted_slice_981[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_983 = tensor.extract_slice %extracted_slice_982[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_984 = tensor.extract_slice %expanded_570[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_985 = tensor.extract_slice %extracted_slice_984[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_986 = tensor.extract_slice %extracted_slice_985[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %1902 = tensor.empty() : tensor<1x40x128xbf16>
    %1903  = tensor.collapse_shape %extracted_slice_983 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1904 = tensor.empty() : tensor<40x128xbf16>
    %1905 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1903 : tensor<1x40x128xbf16>) outs(%1904 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1906 = tensor.empty() : tensor<1x40x128xbf16>
    %1907  = tensor.collapse_shape %extracted_slice_986 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %1908 = tensor.empty() : tensor<40x128xbf16>
    %1909 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1907 : tensor<1x40x128xbf16>) outs(%1908 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %1910 = tensor.empty() : tensor<1x40x128xbf16>
    %1911 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1910 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1905[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1912 = tosa.reshape %1911 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1913 = tensor.empty() : tensor<1x40x128xbf16>
    %1914 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%1913 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %1909[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %1915 = tosa.reshape %1914 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %1916 = tosa.mul %1895, %1912 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_987 = tensor.extract_slice %1895[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_988 = tensor.extract_slice %1895[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1917 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1918 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_988 : tensor<1x32x40x64xbf16>) outs(%1917 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1919 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_989 = tensor.insert_slice %1918 into %1919[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_990 = tensor.insert_slice %extracted_slice_987 into %inserted_slice_989[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1920 = tosa.mul %inserted_slice_990, %1915 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1921 = tosa.add %1916, %1920 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1922 = tosa.mul %1898, %1912 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_991 = tensor.extract_slice %1898[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_992 = tensor.extract_slice %1898[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %1923 = tensor.empty() : tensor<1x32x40x64xbf16>
    %1924 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_992 : tensor<1x32x40x64xbf16>) outs(%1923 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %1925 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_993 = tensor.insert_slice %1924 into %1925[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_994 = tensor.insert_slice %extracted_slice_991 into %inserted_slice_993[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %1926 = tosa.mul %inserted_slice_994, %1915 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1927 = tosa.add %1922, %1926 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1928 = tensor.empty() : tensor<1x32x128x40xbf16>
    %1929 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1927 : tensor<1x32x40x128xbf16>) outs(%1928 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %1930 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1931 = tosa.add %1921, %1930 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1932 = tosa.reshape %1931 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1933 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %1934 = tosa.add %1929, %1933 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %1935 = tosa.reshape %1934 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %1936 = tosa.matmul %1932, %1935 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %1937 = tosa.reshape %1936 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1938 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1939 = tosa.reciprocal %1938 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1940 = tosa.mul %1937, %1939 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1941 = tosa.add %1940, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1942 = tensor.empty() : tensor<1x32x40x40xf32>
    %1943 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1941 : tensor<1x32x40x40xbf16>) outs(%1942 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1944 = tensor.empty() : tensor<1x32x40x1xf32>
    %1945 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1944 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1946 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1943 : tensor<1x32x40x40xf32>) outs(%1944 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1947 = tensor.empty() : tensor<1x32x40x40xf32>
    %1948 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1943, %1946 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1947 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %1949 = tensor.empty() : tensor<1x32x40x1xf32>
    %1950 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1949 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %1951 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1948 : tensor<1x32x40x40xf32>) outs(%1950 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %1952 = tensor.empty() : tensor<1x32x40x40xf32>
    %1953 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1948, %1951 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%1952 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %1954 = tensor.empty() : tensor<1x32x40x40xbf16>
    %1955 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1953 : tensor<1x32x40x40xf32>) outs(%1954 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %1956 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %1957 = tosa.add %1955, %1956 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %1958 = tosa.reshape %1957 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %1959 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %1960 = tosa.add %1901, %1959 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1961 = tosa.reshape %1960 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1962 = tosa.matmul %1958, %1961 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %1963 = tosa.reshape %1962 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %1964 = tensor.empty() : tensor<1x40x32x128xbf16>
    %1965 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1963 : tensor<1x32x40x128xbf16>) outs(%1964 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %1966 = tosa.identity %1965 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %1967 = tosa.reshape %1966 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %1968 = tensor.empty() : tensor<4096x4096xbf16>
    %1969 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_254 : tensor<4096x4096xbf16>) outs(%1968 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %1970 = tosa.reshape %1967 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_995 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %1971 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1970, %1969 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_995 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %1972 = tosa.reshape %1971 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1973 = tosa.add %1864, %1972 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1974 = tensor.empty() : tensor<1x40x4096xf32>
    %1975 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1973 : tensor<1x40x4096xbf16>) outs(%1974 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %1976 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_996 = arith.constant 2 : i32
    %1977 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1975 : tensor<1x40x4096xf32>) outs(%1976 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_996 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_997 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %1978 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1977 : tensor<1x40x4096xf32>) outs(%cst_997 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %1979 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1980 = tosa.add %1978, %1979 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1981 = tosa.rsqrt %1980 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1982 = tosa.mul %1975, %1981 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %1983 = tensor.empty() : tensor<1x40x4096xbf16>
    %1984 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1982 : tensor<1x40x4096xf32>) outs(%1983 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %1985 = tosa.reshape %extracted_slice_26 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %1986 = tosa.mul %1985, %1984 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %1987 = tensor.empty() : tensor<4096x11008xbf16>
    %1988 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_256 : tensor<11008x4096xbf16>) outs(%1987 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1989 = tosa.reshape %1986 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_998 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1990 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1989, %1988 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_998 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1991 = tosa.reshape %1990 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1992 = tensor.empty() : tensor<1x40x11008xbf16>
    %1993 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1991 : tensor<1x40x11008xbf16>) outs(%1992 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %1994 = tensor.empty() : tensor<4096x11008xbf16>
    %1995 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_258 : tensor<11008x4096xbf16>) outs(%1994 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %1996 = tosa.reshape %1986 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_999 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %1997 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1996, %1995 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_999 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %1998 = tosa.reshape %1997 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %1999 = tosa.mul %1993, %1998 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2000 = tensor.empty() : tensor<11008x4096xbf16>
    %2001 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_260 : tensor<4096x11008xbf16>) outs(%2000 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %2002 = tosa.reshape %1999 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1000 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2003 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2002, %2001 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1000 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2004 = tosa.reshape %2003 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2005 = tosa.add %1973, %2004 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2006 = tensor.empty() : tensor<1x40x4096xf32>
    %2007 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2005 : tensor<1x40x4096xbf16>) outs(%2006 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2008 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1001 = arith.constant 2 : i32
    %2009 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2007 : tensor<1x40x4096xf32>) outs(%2008 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1001 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1002 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2010 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2009 : tensor<1x40x4096xf32>) outs(%cst_1002 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2011 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2012 = tosa.add %2010, %2011 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2013 = tosa.rsqrt %2012 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2014 = tosa.mul %2007, %2013 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2015 = tensor.empty() : tensor<1x40x4096xbf16>
    %2016 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2014 : tensor<1x40x4096xf32>) outs(%2015 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2017 = tosa.reshape %extracted_slice_27 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2018 = tosa.mul %2017, %2016 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2019 = tensor.empty() : tensor<4096x4096xbf16>
    %2020 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_262 : tensor<4096x4096xbf16>) outs(%2019 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2021 = tosa.reshape %2018 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1003 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2022 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2021, %2020 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1003 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2023 = tosa.reshape %2022 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2024 = tensor.empty() : tensor<4096x4096xbf16>
    %2025 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_264 : tensor<4096x4096xbf16>) outs(%2024 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2026 = tosa.reshape %2018 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1004 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2027 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2026, %2025 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1004 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2028 = tosa.reshape %2027 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2029 = tensor.empty() : tensor<4096x4096xbf16>
    %2030 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_266 : tensor<4096x4096xbf16>) outs(%2029 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2031 = tosa.reshape %2018 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1005 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2032 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2031, %2030 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1005 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2033 = tosa.reshape %2032 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2034 = tosa.reshape %2023 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2035 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2036 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2034 : tensor<1x40x32x128xbf16>) outs(%2035 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2037 = tosa.reshape %2028 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2038 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2039 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2037 : tensor<1x40x32x128xbf16>) outs(%2038 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2040 = tosa.reshape %2033 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2041 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2042 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2040 : tensor<1x40x32x128xbf16>) outs(%2041 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1006 = tensor.extract_slice %expanded_572[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1007 = tensor.extract_slice %extracted_slice_1006[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1008 = tensor.extract_slice %extracted_slice_1007[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1009 = tensor.extract_slice %expanded_574[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1010 = tensor.extract_slice %extracted_slice_1009[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1011 = tensor.extract_slice %extracted_slice_1010[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %2043 = tensor.empty() : tensor<1x40x128xbf16>
    %2044  = tensor.collapse_shape %extracted_slice_1008 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2045 = tensor.empty() : tensor<40x128xbf16>
    %2046 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2044 : tensor<1x40x128xbf16>) outs(%2045 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2047 = tensor.empty() : tensor<1x40x128xbf16>
    %2048  = tensor.collapse_shape %extracted_slice_1011 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2049 = tensor.empty() : tensor<40x128xbf16>
    %2050 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2048 : tensor<1x40x128xbf16>) outs(%2049 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2051 = tensor.empty() : tensor<1x40x128xbf16>
    %2052 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2051 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2046[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2053 = tosa.reshape %2052 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2054 = tensor.empty() : tensor<1x40x128xbf16>
    %2055 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2054 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2050[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2056 = tosa.reshape %2055 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2057 = tosa.mul %2036, %2053 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1012 = tensor.extract_slice %2036[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1013 = tensor.extract_slice %2036[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2058 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2059 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1013 : tensor<1x32x40x64xbf16>) outs(%2058 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2060 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1014 = tensor.insert_slice %2059 into %2060[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1015 = tensor.insert_slice %extracted_slice_1012 into %inserted_slice_1014[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2061 = tosa.mul %inserted_slice_1015, %2056 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2062 = tosa.add %2057, %2061 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2063 = tosa.mul %2039, %2053 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1016 = tensor.extract_slice %2039[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1017 = tensor.extract_slice %2039[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2064 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2065 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1017 : tensor<1x32x40x64xbf16>) outs(%2064 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2066 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1018 = tensor.insert_slice %2065 into %2066[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1019 = tensor.insert_slice %extracted_slice_1016 into %inserted_slice_1018[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2067 = tosa.mul %inserted_slice_1019, %2056 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2068 = tosa.add %2063, %2067 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2069 = tensor.empty() : tensor<1x32x128x40xbf16>
    %2070 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2068 : tensor<1x32x40x128xbf16>) outs(%2069 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %2071 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2072 = tosa.add %2062, %2071 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2073 = tosa.reshape %2072 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2074 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %2075 = tosa.add %2070, %2074 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %2076 = tosa.reshape %2075 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %2077 = tosa.matmul %2073, %2076 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %2078 = tosa.reshape %2077 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2079 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2080 = tosa.reciprocal %2079 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2081 = tosa.mul %2078, %2080 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2082 = tosa.add %2081, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2083 = tensor.empty() : tensor<1x32x40x40xf32>
    %2084 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2082 : tensor<1x32x40x40xbf16>) outs(%2083 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2085 = tensor.empty() : tensor<1x32x40x1xf32>
    %2086 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2085 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2087 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2084 : tensor<1x32x40x40xf32>) outs(%2085 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2088 = tensor.empty() : tensor<1x32x40x40xf32>
    %2089 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2084, %2087 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2088 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %2090 = tensor.empty() : tensor<1x32x40x1xf32>
    %2091 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2090 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2092 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2089 : tensor<1x32x40x40xf32>) outs(%2091 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2093 = tensor.empty() : tensor<1x32x40x40xf32>
    %2094 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2089, %2092 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2093 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2095 = tensor.empty() : tensor<1x32x40x40xbf16>
    %2096 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2094 : tensor<1x32x40x40xf32>) outs(%2095 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %2097 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2098 = tosa.add %2096, %2097 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2099 = tosa.reshape %2098 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %2100 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2101 = tosa.add %2042, %2100 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2102 = tosa.reshape %2101 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2103 = tosa.matmul %2099, %2102 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2104 = tosa.reshape %2103 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2105 = tensor.empty() : tensor<1x40x32x128xbf16>
    %2106 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2104 : tensor<1x32x40x128xbf16>) outs(%2105 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %2107 = tosa.identity %2106 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %2108 = tosa.reshape %2107 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %2109 = tensor.empty() : tensor<4096x4096xbf16>
    %2110 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_268 : tensor<4096x4096xbf16>) outs(%2109 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2111 = tosa.reshape %2108 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1020 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2112 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2111, %2110 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1020 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2113 = tosa.reshape %2112 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2114 = tosa.add %2005, %2113 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2115 = tensor.empty() : tensor<1x40x4096xf32>
    %2116 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2114 : tensor<1x40x4096xbf16>) outs(%2115 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2117 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1021 = arith.constant 2 : i32
    %2118 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2116 : tensor<1x40x4096xf32>) outs(%2117 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1021 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1022 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2119 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2118 : tensor<1x40x4096xf32>) outs(%cst_1022 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2120 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2121 = tosa.add %2119, %2120 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2122 = tosa.rsqrt %2121 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2123 = tosa.mul %2116, %2122 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2124 = tensor.empty() : tensor<1x40x4096xbf16>
    %2125 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2123 : tensor<1x40x4096xf32>) outs(%2124 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2126 = tosa.reshape %extracted_slice_28 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2127 = tosa.mul %2126, %2125 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2128 = tensor.empty() : tensor<4096x11008xbf16>
    %2129 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_270 : tensor<11008x4096xbf16>) outs(%2128 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2130 = tosa.reshape %2127 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1023 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2131 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2130, %2129 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1023 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2132 = tosa.reshape %2131 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2133 = tensor.empty() : tensor<1x40x11008xbf16>
    %2134 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2132 : tensor<1x40x11008xbf16>) outs(%2133 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %2135 = tensor.empty() : tensor<4096x11008xbf16>
    %2136 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_272 : tensor<11008x4096xbf16>) outs(%2135 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2137 = tosa.reshape %2127 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1024 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2138 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2137, %2136 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1024 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2139 = tosa.reshape %2138 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2140 = tosa.mul %2134, %2139 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2141 = tensor.empty() : tensor<11008x4096xbf16>
    %2142 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_274 : tensor<4096x11008xbf16>) outs(%2141 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %2143 = tosa.reshape %2140 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1025 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2144 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2143, %2142 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1025 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2145 = tosa.reshape %2144 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2146 = tosa.add %2114, %2145 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2147 = tensor.empty() : tensor<1x40x4096xf32>
    %2148 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2146 : tensor<1x40x4096xbf16>) outs(%2147 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2149 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1026 = arith.constant 2 : i32
    %2150 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2148 : tensor<1x40x4096xf32>) outs(%2149 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1026 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1027 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2151 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2150 : tensor<1x40x4096xf32>) outs(%cst_1027 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2152 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2153 = tosa.add %2151, %2152 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2154 = tosa.rsqrt %2153 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2155 = tosa.mul %2148, %2154 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2156 = tensor.empty() : tensor<1x40x4096xbf16>
    %2157 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2155 : tensor<1x40x4096xf32>) outs(%2156 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2158 = tosa.reshape %extracted_slice_29 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2159 = tosa.mul %2158, %2157 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2160 = tensor.empty() : tensor<4096x4096xbf16>
    %2161 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_276 : tensor<4096x4096xbf16>) outs(%2160 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2162 = tosa.reshape %2159 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1028 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2163 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2162, %2161 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1028 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2164 = tosa.reshape %2163 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2165 = tensor.empty() : tensor<4096x4096xbf16>
    %2166 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_278 : tensor<4096x4096xbf16>) outs(%2165 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2167 = tosa.reshape %2159 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1029 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2168 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2167, %2166 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1029 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2169 = tosa.reshape %2168 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2170 = tensor.empty() : tensor<4096x4096xbf16>
    %2171 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_280 : tensor<4096x4096xbf16>) outs(%2170 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2172 = tosa.reshape %2159 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1030 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2173 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2172, %2171 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1030 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2174 = tosa.reshape %2173 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2175 = tosa.reshape %2164 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2176 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2177 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2175 : tensor<1x40x32x128xbf16>) outs(%2176 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2178 = tosa.reshape %2169 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2179 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2180 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2178 : tensor<1x40x32x128xbf16>) outs(%2179 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2181 = tosa.reshape %2174 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2182 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2183 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2181 : tensor<1x40x32x128xbf16>) outs(%2182 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1031 = tensor.extract_slice %expanded_576[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1032 = tensor.extract_slice %extracted_slice_1031[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1033 = tensor.extract_slice %extracted_slice_1032[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1034 = tensor.extract_slice %expanded_578[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1035 = tensor.extract_slice %extracted_slice_1034[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1036 = tensor.extract_slice %extracted_slice_1035[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %2184 = tensor.empty() : tensor<1x40x128xbf16>
    %2185  = tensor.collapse_shape %extracted_slice_1033 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2186 = tensor.empty() : tensor<40x128xbf16>
    %2187 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2185 : tensor<1x40x128xbf16>) outs(%2186 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2188 = tensor.empty() : tensor<1x40x128xbf16>
    %2189  = tensor.collapse_shape %extracted_slice_1036 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2190 = tensor.empty() : tensor<40x128xbf16>
    %2191 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2189 : tensor<1x40x128xbf16>) outs(%2190 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2192 = tensor.empty() : tensor<1x40x128xbf16>
    %2193 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2192 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2187[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2194 = tosa.reshape %2193 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2195 = tensor.empty() : tensor<1x40x128xbf16>
    %2196 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2195 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2191[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2197 = tosa.reshape %2196 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2198 = tosa.mul %2177, %2194 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1037 = tensor.extract_slice %2177[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1038 = tensor.extract_slice %2177[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2199 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2200 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1038 : tensor<1x32x40x64xbf16>) outs(%2199 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2201 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1039 = tensor.insert_slice %2200 into %2201[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1040 = tensor.insert_slice %extracted_slice_1037 into %inserted_slice_1039[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2202 = tosa.mul %inserted_slice_1040, %2197 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2203 = tosa.add %2198, %2202 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2204 = tosa.mul %2180, %2194 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1041 = tensor.extract_slice %2180[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1042 = tensor.extract_slice %2180[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2205 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2206 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1042 : tensor<1x32x40x64xbf16>) outs(%2205 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2207 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1043 = tensor.insert_slice %2206 into %2207[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1044 = tensor.insert_slice %extracted_slice_1041 into %inserted_slice_1043[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2208 = tosa.mul %inserted_slice_1044, %2197 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2209 = tosa.add %2204, %2208 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2210 = tensor.empty() : tensor<1x32x128x40xbf16>
    %2211 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2209 : tensor<1x32x40x128xbf16>) outs(%2210 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %2212 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2213 = tosa.add %2203, %2212 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2214 = tosa.reshape %2213 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2215 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %2216 = tosa.add %2211, %2215 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %2217 = tosa.reshape %2216 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %2218 = tosa.matmul %2214, %2217 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %2219 = tosa.reshape %2218 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2220 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2221 = tosa.reciprocal %2220 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2222 = tosa.mul %2219, %2221 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2223 = tosa.add %2222, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2224 = tensor.empty() : tensor<1x32x40x40xf32>
    %2225 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2223 : tensor<1x32x40x40xbf16>) outs(%2224 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2226 = tensor.empty() : tensor<1x32x40x1xf32>
    %2227 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2226 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2228 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2225 : tensor<1x32x40x40xf32>) outs(%2226 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2229 = tensor.empty() : tensor<1x32x40x40xf32>
    %2230 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2225, %2228 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2229 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %2231 = tensor.empty() : tensor<1x32x40x1xf32>
    %2232 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2231 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2233 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2230 : tensor<1x32x40x40xf32>) outs(%2232 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2234 = tensor.empty() : tensor<1x32x40x40xf32>
    %2235 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2230, %2233 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2234 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2236 = tensor.empty() : tensor<1x32x40x40xbf16>
    %2237 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2235 : tensor<1x32x40x40xf32>) outs(%2236 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %2238 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2239 = tosa.add %2237, %2238 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2240 = tosa.reshape %2239 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %2241 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2242 = tosa.add %2183, %2241 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2243 = tosa.reshape %2242 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2244 = tosa.matmul %2240, %2243 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2245 = tosa.reshape %2244 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2246 = tensor.empty() : tensor<1x40x32x128xbf16>
    %2247 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2245 : tensor<1x32x40x128xbf16>) outs(%2246 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %2248 = tosa.identity %2247 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %2249 = tosa.reshape %2248 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %2250 = tensor.empty() : tensor<4096x4096xbf16>
    %2251 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_282 : tensor<4096x4096xbf16>) outs(%2250 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2252 = tosa.reshape %2249 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1045 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2253 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2252, %2251 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1045 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2254 = tosa.reshape %2253 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2255 = tosa.add %2146, %2254 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2256 = tensor.empty() : tensor<1x40x4096xf32>
    %2257 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2255 : tensor<1x40x4096xbf16>) outs(%2256 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2258 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1046 = arith.constant 2 : i32
    %2259 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2257 : tensor<1x40x4096xf32>) outs(%2258 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1046 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1047 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2260 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2259 : tensor<1x40x4096xf32>) outs(%cst_1047 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2261 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2262 = tosa.add %2260, %2261 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2263 = tosa.rsqrt %2262 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2264 = tosa.mul %2257, %2263 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2265 = tensor.empty() : tensor<1x40x4096xbf16>
    %2266 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2264 : tensor<1x40x4096xf32>) outs(%2265 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2267 = tosa.reshape %extracted_slice_30 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2268 = tosa.mul %2267, %2266 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2269 = tensor.empty() : tensor<4096x11008xbf16>
    %2270 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_284 : tensor<11008x4096xbf16>) outs(%2269 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2271 = tosa.reshape %2268 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1048 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2272 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2271, %2270 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1048 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2273 = tosa.reshape %2272 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2274 = tensor.empty() : tensor<1x40x11008xbf16>
    %2275 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2273 : tensor<1x40x11008xbf16>) outs(%2274 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %2276 = tensor.empty() : tensor<4096x11008xbf16>
    %2277 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_286 : tensor<11008x4096xbf16>) outs(%2276 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2278 = tosa.reshape %2268 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1049 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2279 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2278, %2277 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1049 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2280 = tosa.reshape %2279 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2281 = tosa.mul %2275, %2280 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2282 = tensor.empty() : tensor<11008x4096xbf16>
    %2283 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_288 : tensor<4096x11008xbf16>) outs(%2282 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %2284 = tosa.reshape %2281 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1050 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2285 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2284, %2283 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1050 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2286 = tosa.reshape %2285 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2287 = tosa.add %2255, %2286 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2288 = tensor.empty() : tensor<1x40x4096xf32>
    %2289 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2287 : tensor<1x40x4096xbf16>) outs(%2288 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2290 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1051 = arith.constant 2 : i32
    %2291 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2289 : tensor<1x40x4096xf32>) outs(%2290 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1051 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1052 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2292 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2291 : tensor<1x40x4096xf32>) outs(%cst_1052 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2293 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2294 = tosa.add %2292, %2293 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2295 = tosa.rsqrt %2294 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2296 = tosa.mul %2289, %2295 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2297 = tensor.empty() : tensor<1x40x4096xbf16>
    %2298 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2296 : tensor<1x40x4096xf32>) outs(%2297 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2299 = tosa.reshape %extracted_slice_31 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2300 = tosa.mul %2299, %2298 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2301 = tensor.empty() : tensor<4096x4096xbf16>
    %2302 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_290 : tensor<4096x4096xbf16>) outs(%2301 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2303 = tosa.reshape %2300 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1053 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2304 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2303, %2302 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1053 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2305 = tosa.reshape %2304 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2306 = tensor.empty() : tensor<4096x4096xbf16>
    %2307 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_292 : tensor<4096x4096xbf16>) outs(%2306 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2308 = tosa.reshape %2300 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1054 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2309 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2308, %2307 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1054 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2310 = tosa.reshape %2309 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2311 = tensor.empty() : tensor<4096x4096xbf16>
    %2312 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_294 : tensor<4096x4096xbf16>) outs(%2311 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2313 = tosa.reshape %2300 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1055 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2314 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2313, %2312 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1055 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2315 = tosa.reshape %2314 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2316 = tosa.reshape %2305 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2317 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2318 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2316 : tensor<1x40x32x128xbf16>) outs(%2317 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2319 = tosa.reshape %2310 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2320 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2321 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2319 : tensor<1x40x32x128xbf16>) outs(%2320 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2322 = tosa.reshape %2315 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2323 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2324 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2322 : tensor<1x40x32x128xbf16>) outs(%2323 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1056 = tensor.extract_slice %expanded_580[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1057 = tensor.extract_slice %extracted_slice_1056[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1058 = tensor.extract_slice %extracted_slice_1057[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1059 = tensor.extract_slice %expanded_582[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1060 = tensor.extract_slice %extracted_slice_1059[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1061 = tensor.extract_slice %extracted_slice_1060[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %2325 = tensor.empty() : tensor<1x40x128xbf16>
    %2326  = tensor.collapse_shape %extracted_slice_1058 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2327 = tensor.empty() : tensor<40x128xbf16>
    %2328 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2326 : tensor<1x40x128xbf16>) outs(%2327 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2329 = tensor.empty() : tensor<1x40x128xbf16>
    %2330  = tensor.collapse_shape %extracted_slice_1061 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2331 = tensor.empty() : tensor<40x128xbf16>
    %2332 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2330 : tensor<1x40x128xbf16>) outs(%2331 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2333 = tensor.empty() : tensor<1x40x128xbf16>
    %2334 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2333 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2328[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2335 = tosa.reshape %2334 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2336 = tensor.empty() : tensor<1x40x128xbf16>
    %2337 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2336 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2332[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2338 = tosa.reshape %2337 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2339 = tosa.mul %2318, %2335 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1062 = tensor.extract_slice %2318[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1063 = tensor.extract_slice %2318[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2340 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2341 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1063 : tensor<1x32x40x64xbf16>) outs(%2340 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2342 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1064 = tensor.insert_slice %2341 into %2342[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1065 = tensor.insert_slice %extracted_slice_1062 into %inserted_slice_1064[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2343 = tosa.mul %inserted_slice_1065, %2338 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2344 = tosa.add %2339, %2343 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2345 = tosa.mul %2321, %2335 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1066 = tensor.extract_slice %2321[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1067 = tensor.extract_slice %2321[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2346 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2347 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1067 : tensor<1x32x40x64xbf16>) outs(%2346 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2348 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1068 = tensor.insert_slice %2347 into %2348[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1069 = tensor.insert_slice %extracted_slice_1066 into %inserted_slice_1068[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2349 = tosa.mul %inserted_slice_1069, %2338 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2350 = tosa.add %2345, %2349 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2351 = tensor.empty() : tensor<1x32x128x40xbf16>
    %2352 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2350 : tensor<1x32x40x128xbf16>) outs(%2351 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %2353 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2354 = tosa.add %2344, %2353 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2355 = tosa.reshape %2354 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2356 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %2357 = tosa.add %2352, %2356 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %2358 = tosa.reshape %2357 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %2359 = tosa.matmul %2355, %2358 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %2360 = tosa.reshape %2359 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2361 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2362 = tosa.reciprocal %2361 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2363 = tosa.mul %2360, %2362 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2364 = tosa.add %2363, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2365 = tensor.empty() : tensor<1x32x40x40xf32>
    %2366 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2364 : tensor<1x32x40x40xbf16>) outs(%2365 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2367 = tensor.empty() : tensor<1x32x40x1xf32>
    %2368 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2367 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2369 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2366 : tensor<1x32x40x40xf32>) outs(%2367 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2370 = tensor.empty() : tensor<1x32x40x40xf32>
    %2371 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2366, %2369 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2370 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %2372 = tensor.empty() : tensor<1x32x40x1xf32>
    %2373 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2372 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2374 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2371 : tensor<1x32x40x40xf32>) outs(%2373 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2375 = tensor.empty() : tensor<1x32x40x40xf32>
    %2376 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2371, %2374 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2375 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2377 = tensor.empty() : tensor<1x32x40x40xbf16>
    %2378 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2376 : tensor<1x32x40x40xf32>) outs(%2377 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %2379 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2380 = tosa.add %2378, %2379 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2381 = tosa.reshape %2380 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %2382 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2383 = tosa.add %2324, %2382 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2384 = tosa.reshape %2383 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2385 = tosa.matmul %2381, %2384 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2386 = tosa.reshape %2385 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2387 = tensor.empty() : tensor<1x40x32x128xbf16>
    %2388 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2386 : tensor<1x32x40x128xbf16>) outs(%2387 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %2389 = tosa.identity %2388 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %2390 = tosa.reshape %2389 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %2391 = tensor.empty() : tensor<4096x4096xbf16>
    %2392 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_296 : tensor<4096x4096xbf16>) outs(%2391 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2393 = tosa.reshape %2390 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1070 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2394 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2393, %2392 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1070 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2395 = tosa.reshape %2394 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2396 = tosa.add %2287, %2395 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2397 = tensor.empty() : tensor<1x40x4096xf32>
    %2398 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2396 : tensor<1x40x4096xbf16>) outs(%2397 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2399 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1071 = arith.constant 2 : i32
    %2400 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2398 : tensor<1x40x4096xf32>) outs(%2399 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1071 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1072 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2401 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2400 : tensor<1x40x4096xf32>) outs(%cst_1072 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2402 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2403 = tosa.add %2401, %2402 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2404 = tosa.rsqrt %2403 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2405 = tosa.mul %2398, %2404 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2406 = tensor.empty() : tensor<1x40x4096xbf16>
    %2407 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2405 : tensor<1x40x4096xf32>) outs(%2406 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2408 = tosa.reshape %extracted_slice_32 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2409 = tosa.mul %2408, %2407 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2410 = tensor.empty() : tensor<4096x11008xbf16>
    %2411 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_298 : tensor<11008x4096xbf16>) outs(%2410 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2412 = tosa.reshape %2409 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1073 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2413 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2412, %2411 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1073 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2414 = tosa.reshape %2413 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2415 = tensor.empty() : tensor<1x40x11008xbf16>
    %2416 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2414 : tensor<1x40x11008xbf16>) outs(%2415 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %2417 = tensor.empty() : tensor<4096x11008xbf16>
    %2418 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_300 : tensor<11008x4096xbf16>) outs(%2417 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2419 = tosa.reshape %2409 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1074 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2420 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2419, %2418 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1074 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2421 = tosa.reshape %2420 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2422 = tosa.mul %2416, %2421 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2423 = tensor.empty() : tensor<11008x4096xbf16>
    %2424 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_302 : tensor<4096x11008xbf16>) outs(%2423 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %2425 = tosa.reshape %2422 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1075 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2426 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2425, %2424 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1075 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2427 = tosa.reshape %2426 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2428 = tosa.add %2396, %2427 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2429 = tensor.empty() : tensor<1x40x4096xf32>
    %2430 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2428 : tensor<1x40x4096xbf16>) outs(%2429 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2431 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1076 = arith.constant 2 : i32
    %2432 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2430 : tensor<1x40x4096xf32>) outs(%2431 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1076 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1077 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2433 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2432 : tensor<1x40x4096xf32>) outs(%cst_1077 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2434 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2435 = tosa.add %2433, %2434 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2436 = tosa.rsqrt %2435 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2437 = tosa.mul %2430, %2436 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2438 = tensor.empty() : tensor<1x40x4096xbf16>
    %2439 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2437 : tensor<1x40x4096xf32>) outs(%2438 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2440 = tosa.reshape %extracted_slice_33 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2441 = tosa.mul %2440, %2439 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2442 = tensor.empty() : tensor<4096x4096xbf16>
    %2443 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_304 : tensor<4096x4096xbf16>) outs(%2442 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2444 = tosa.reshape %2441 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1078 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2445 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2444, %2443 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1078 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2446 = tosa.reshape %2445 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2447 = tensor.empty() : tensor<4096x4096xbf16>
    %2448 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_306 : tensor<4096x4096xbf16>) outs(%2447 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2449 = tosa.reshape %2441 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1079 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2450 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2449, %2448 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1079 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2451 = tosa.reshape %2450 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2452 = tensor.empty() : tensor<4096x4096xbf16>
    %2453 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_308 : tensor<4096x4096xbf16>) outs(%2452 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2454 = tosa.reshape %2441 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1080 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2455 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2454, %2453 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1080 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2456 = tosa.reshape %2455 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2457 = tosa.reshape %2446 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2458 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2459 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2457 : tensor<1x40x32x128xbf16>) outs(%2458 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2460 = tosa.reshape %2451 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2461 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2462 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2460 : tensor<1x40x32x128xbf16>) outs(%2461 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2463 = tosa.reshape %2456 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2464 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2465 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2463 : tensor<1x40x32x128xbf16>) outs(%2464 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1081 = tensor.extract_slice %expanded_584[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1082 = tensor.extract_slice %extracted_slice_1081[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1083 = tensor.extract_slice %extracted_slice_1082[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1084 = tensor.extract_slice %expanded_586[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1085 = tensor.extract_slice %extracted_slice_1084[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1086 = tensor.extract_slice %extracted_slice_1085[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %2466 = tensor.empty() : tensor<1x40x128xbf16>
    %2467  = tensor.collapse_shape %extracted_slice_1083 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2468 = tensor.empty() : tensor<40x128xbf16>
    %2469 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2467 : tensor<1x40x128xbf16>) outs(%2468 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2470 = tensor.empty() : tensor<1x40x128xbf16>
    %2471  = tensor.collapse_shape %extracted_slice_1086 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2472 = tensor.empty() : tensor<40x128xbf16>
    %2473 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2471 : tensor<1x40x128xbf16>) outs(%2472 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2474 = tensor.empty() : tensor<1x40x128xbf16>
    %2475 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2474 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2469[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2476 = tosa.reshape %2475 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2477 = tensor.empty() : tensor<1x40x128xbf16>
    %2478 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2477 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2473[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2479 = tosa.reshape %2478 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2480 = tosa.mul %2459, %2476 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1087 = tensor.extract_slice %2459[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1088 = tensor.extract_slice %2459[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2481 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2482 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1088 : tensor<1x32x40x64xbf16>) outs(%2481 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2483 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1089 = tensor.insert_slice %2482 into %2483[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1090 = tensor.insert_slice %extracted_slice_1087 into %inserted_slice_1089[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2484 = tosa.mul %inserted_slice_1090, %2479 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2485 = tosa.add %2480, %2484 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2486 = tosa.mul %2462, %2476 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1091 = tensor.extract_slice %2462[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1092 = tensor.extract_slice %2462[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2487 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2488 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1092 : tensor<1x32x40x64xbf16>) outs(%2487 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2489 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1093 = tensor.insert_slice %2488 into %2489[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1094 = tensor.insert_slice %extracted_slice_1091 into %inserted_slice_1093[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2490 = tosa.mul %inserted_slice_1094, %2479 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2491 = tosa.add %2486, %2490 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2492 = tensor.empty() : tensor<1x32x128x40xbf16>
    %2493 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2491 : tensor<1x32x40x128xbf16>) outs(%2492 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %2494 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2495 = tosa.add %2485, %2494 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2496 = tosa.reshape %2495 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2497 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %2498 = tosa.add %2493, %2497 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %2499 = tosa.reshape %2498 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %2500 = tosa.matmul %2496, %2499 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %2501 = tosa.reshape %2500 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2502 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2503 = tosa.reciprocal %2502 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2504 = tosa.mul %2501, %2503 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2505 = tosa.add %2504, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2506 = tensor.empty() : tensor<1x32x40x40xf32>
    %2507 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2505 : tensor<1x32x40x40xbf16>) outs(%2506 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2508 = tensor.empty() : tensor<1x32x40x1xf32>
    %2509 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2508 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2510 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2507 : tensor<1x32x40x40xf32>) outs(%2508 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2511 = tensor.empty() : tensor<1x32x40x40xf32>
    %2512 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2507, %2510 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2511 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %2513 = tensor.empty() : tensor<1x32x40x1xf32>
    %2514 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2513 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2515 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2512 : tensor<1x32x40x40xf32>) outs(%2514 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2516 = tensor.empty() : tensor<1x32x40x40xf32>
    %2517 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2512, %2515 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2516 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2518 = tensor.empty() : tensor<1x32x40x40xbf16>
    %2519 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2517 : tensor<1x32x40x40xf32>) outs(%2518 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %2520 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2521 = tosa.add %2519, %2520 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2522 = tosa.reshape %2521 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %2523 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2524 = tosa.add %2465, %2523 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2525 = tosa.reshape %2524 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2526 = tosa.matmul %2522, %2525 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2527 = tosa.reshape %2526 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2528 = tensor.empty() : tensor<1x40x32x128xbf16>
    %2529 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2527 : tensor<1x32x40x128xbf16>) outs(%2528 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %2530 = tosa.identity %2529 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %2531 = tosa.reshape %2530 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %2532 = tensor.empty() : tensor<4096x4096xbf16>
    %2533 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_310 : tensor<4096x4096xbf16>) outs(%2532 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2534 = tosa.reshape %2531 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1095 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2535 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2534, %2533 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1095 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2536 = tosa.reshape %2535 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2537 = tosa.add %2428, %2536 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2538 = tensor.empty() : tensor<1x40x4096xf32>
    %2539 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2537 : tensor<1x40x4096xbf16>) outs(%2538 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2540 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1096 = arith.constant 2 : i32
    %2541 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2539 : tensor<1x40x4096xf32>) outs(%2540 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1096 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1097 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2542 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2541 : tensor<1x40x4096xf32>) outs(%cst_1097 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2543 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2544 = tosa.add %2542, %2543 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2545 = tosa.rsqrt %2544 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2546 = tosa.mul %2539, %2545 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2547 = tensor.empty() : tensor<1x40x4096xbf16>
    %2548 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2546 : tensor<1x40x4096xf32>) outs(%2547 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2549 = tosa.reshape %extracted_slice_34 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2550 = tosa.mul %2549, %2548 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2551 = tensor.empty() : tensor<4096x11008xbf16>
    %2552 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_312 : tensor<11008x4096xbf16>) outs(%2551 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2553 = tosa.reshape %2550 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1098 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2554 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2553, %2552 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1098 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2555 = tosa.reshape %2554 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2556 = tensor.empty() : tensor<1x40x11008xbf16>
    %2557 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2555 : tensor<1x40x11008xbf16>) outs(%2556 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %2558 = tensor.empty() : tensor<4096x11008xbf16>
    %2559 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_314 : tensor<11008x4096xbf16>) outs(%2558 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2560 = tosa.reshape %2550 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1099 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2561 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2560, %2559 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1099 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2562 = tosa.reshape %2561 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2563 = tosa.mul %2557, %2562 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2564 = tensor.empty() : tensor<11008x4096xbf16>
    %2565 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_316 : tensor<4096x11008xbf16>) outs(%2564 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %2566 = tosa.reshape %2563 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1100 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2567 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2566, %2565 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1100 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2568 = tosa.reshape %2567 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2569 = tosa.add %2537, %2568 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2570 = tensor.empty() : tensor<1x40x4096xf32>
    %2571 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2569 : tensor<1x40x4096xbf16>) outs(%2570 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2572 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1101 = arith.constant 2 : i32
    %2573 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2571 : tensor<1x40x4096xf32>) outs(%2572 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1101 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1102 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2574 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2573 : tensor<1x40x4096xf32>) outs(%cst_1102 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2575 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2576 = tosa.add %2574, %2575 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2577 = tosa.rsqrt %2576 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2578 = tosa.mul %2571, %2577 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2579 = tensor.empty() : tensor<1x40x4096xbf16>
    %2580 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2578 : tensor<1x40x4096xf32>) outs(%2579 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2581 = tosa.reshape %extracted_slice_35 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2582 = tosa.mul %2581, %2580 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2583 = tensor.empty() : tensor<4096x4096xbf16>
    %2584 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_318 : tensor<4096x4096xbf16>) outs(%2583 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2585 = tosa.reshape %2582 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1103 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2586 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2585, %2584 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1103 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2587 = tosa.reshape %2586 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2588 = tensor.empty() : tensor<4096x4096xbf16>
    %2589 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_320 : tensor<4096x4096xbf16>) outs(%2588 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2590 = tosa.reshape %2582 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1104 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2591 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2590, %2589 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1104 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2592 = tosa.reshape %2591 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2593 = tensor.empty() : tensor<4096x4096xbf16>
    %2594 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_322 : tensor<4096x4096xbf16>) outs(%2593 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2595 = tosa.reshape %2582 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1105 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2596 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2595, %2594 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1105 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2597 = tosa.reshape %2596 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2598 = tosa.reshape %2587 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2599 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2600 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2598 : tensor<1x40x32x128xbf16>) outs(%2599 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2601 = tosa.reshape %2592 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2602 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2603 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2601 : tensor<1x40x32x128xbf16>) outs(%2602 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2604 = tosa.reshape %2597 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2605 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2606 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2604 : tensor<1x40x32x128xbf16>) outs(%2605 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1106 = tensor.extract_slice %expanded_588[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1107 = tensor.extract_slice %extracted_slice_1106[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1108 = tensor.extract_slice %extracted_slice_1107[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1109 = tensor.extract_slice %expanded_590[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1110 = tensor.extract_slice %extracted_slice_1109[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1111 = tensor.extract_slice %extracted_slice_1110[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %2607 = tensor.empty() : tensor<1x40x128xbf16>
    %2608  = tensor.collapse_shape %extracted_slice_1108 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2609 = tensor.empty() : tensor<40x128xbf16>
    %2610 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2608 : tensor<1x40x128xbf16>) outs(%2609 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2611 = tensor.empty() : tensor<1x40x128xbf16>
    %2612  = tensor.collapse_shape %extracted_slice_1111 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2613 = tensor.empty() : tensor<40x128xbf16>
    %2614 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2612 : tensor<1x40x128xbf16>) outs(%2613 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2615 = tensor.empty() : tensor<1x40x128xbf16>
    %2616 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2615 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2610[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2617 = tosa.reshape %2616 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2618 = tensor.empty() : tensor<1x40x128xbf16>
    %2619 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2618 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2614[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2620 = tosa.reshape %2619 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2621 = tosa.mul %2600, %2617 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1112 = tensor.extract_slice %2600[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1113 = tensor.extract_slice %2600[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2622 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2623 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1113 : tensor<1x32x40x64xbf16>) outs(%2622 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2624 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1114 = tensor.insert_slice %2623 into %2624[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1115 = tensor.insert_slice %extracted_slice_1112 into %inserted_slice_1114[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2625 = tosa.mul %inserted_slice_1115, %2620 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2626 = tosa.add %2621, %2625 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2627 = tosa.mul %2603, %2617 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1116 = tensor.extract_slice %2603[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1117 = tensor.extract_slice %2603[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2628 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2629 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1117 : tensor<1x32x40x64xbf16>) outs(%2628 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2630 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1118 = tensor.insert_slice %2629 into %2630[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1119 = tensor.insert_slice %extracted_slice_1116 into %inserted_slice_1118[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2631 = tosa.mul %inserted_slice_1119, %2620 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2632 = tosa.add %2627, %2631 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2633 = tensor.empty() : tensor<1x32x128x40xbf16>
    %2634 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2632 : tensor<1x32x40x128xbf16>) outs(%2633 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %2635 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2636 = tosa.add %2626, %2635 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2637 = tosa.reshape %2636 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2638 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %2639 = tosa.add %2634, %2638 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %2640 = tosa.reshape %2639 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %2641 = tosa.matmul %2637, %2640 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %2642 = tosa.reshape %2641 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2643 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2644 = tosa.reciprocal %2643 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2645 = tosa.mul %2642, %2644 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2646 = tosa.add %2645, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2647 = tensor.empty() : tensor<1x32x40x40xf32>
    %2648 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2646 : tensor<1x32x40x40xbf16>) outs(%2647 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2649 = tensor.empty() : tensor<1x32x40x1xf32>
    %2650 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2649 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2651 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2648 : tensor<1x32x40x40xf32>) outs(%2649 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2652 = tensor.empty() : tensor<1x32x40x40xf32>
    %2653 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2648, %2651 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2652 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %2654 = tensor.empty() : tensor<1x32x40x1xf32>
    %2655 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2654 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2656 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2653 : tensor<1x32x40x40xf32>) outs(%2655 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2657 = tensor.empty() : tensor<1x32x40x40xf32>
    %2658 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2653, %2656 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2657 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2659 = tensor.empty() : tensor<1x32x40x40xbf16>
    %2660 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2658 : tensor<1x32x40x40xf32>) outs(%2659 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %2661 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2662 = tosa.add %2660, %2661 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2663 = tosa.reshape %2662 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %2664 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2665 = tosa.add %2606, %2664 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2666 = tosa.reshape %2665 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2667 = tosa.matmul %2663, %2666 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2668 = tosa.reshape %2667 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2669 = tensor.empty() : tensor<1x40x32x128xbf16>
    %2670 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2668 : tensor<1x32x40x128xbf16>) outs(%2669 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %2671 = tosa.identity %2670 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %2672 = tosa.reshape %2671 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %2673 = tensor.empty() : tensor<4096x4096xbf16>
    %2674 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_324 : tensor<4096x4096xbf16>) outs(%2673 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2675 = tosa.reshape %2672 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1120 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2676 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2675, %2674 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1120 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2677 = tosa.reshape %2676 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2678 = tosa.add %2569, %2677 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2679 = tensor.empty() : tensor<1x40x4096xf32>
    %2680 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2678 : tensor<1x40x4096xbf16>) outs(%2679 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2681 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1121 = arith.constant 2 : i32
    %2682 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2680 : tensor<1x40x4096xf32>) outs(%2681 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1121 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1122 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2683 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2682 : tensor<1x40x4096xf32>) outs(%cst_1122 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2684 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2685 = tosa.add %2683, %2684 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2686 = tosa.rsqrt %2685 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2687 = tosa.mul %2680, %2686 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2688 = tensor.empty() : tensor<1x40x4096xbf16>
    %2689 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2687 : tensor<1x40x4096xf32>) outs(%2688 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2690 = tosa.reshape %extracted_slice_36 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2691 = tosa.mul %2690, %2689 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2692 = tensor.empty() : tensor<4096x11008xbf16>
    %2693 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_326 : tensor<11008x4096xbf16>) outs(%2692 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2694 = tosa.reshape %2691 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1123 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2695 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2694, %2693 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1123 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2696 = tosa.reshape %2695 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2697 = tensor.empty() : tensor<1x40x11008xbf16>
    %2698 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2696 : tensor<1x40x11008xbf16>) outs(%2697 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %2699 = tensor.empty() : tensor<4096x11008xbf16>
    %2700 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_328 : tensor<11008x4096xbf16>) outs(%2699 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2701 = tosa.reshape %2691 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1124 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2702 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2701, %2700 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1124 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2703 = tosa.reshape %2702 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2704 = tosa.mul %2698, %2703 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2705 = tensor.empty() : tensor<11008x4096xbf16>
    %2706 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_330 : tensor<4096x11008xbf16>) outs(%2705 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %2707 = tosa.reshape %2704 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1125 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2708 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2707, %2706 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1125 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2709 = tosa.reshape %2708 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2710 = tosa.add %2678, %2709 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2711 = tensor.empty() : tensor<1x40x4096xf32>
    %2712 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2710 : tensor<1x40x4096xbf16>) outs(%2711 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2713 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1126 = arith.constant 2 : i32
    %2714 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2712 : tensor<1x40x4096xf32>) outs(%2713 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1126 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1127 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2715 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2714 : tensor<1x40x4096xf32>) outs(%cst_1127 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2716 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2717 = tosa.add %2715, %2716 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2718 = tosa.rsqrt %2717 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2719 = tosa.mul %2712, %2718 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2720 = tensor.empty() : tensor<1x40x4096xbf16>
    %2721 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2719 : tensor<1x40x4096xf32>) outs(%2720 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2722 = tosa.reshape %extracted_slice_37 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2723 = tosa.mul %2722, %2721 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2724 = tensor.empty() : tensor<4096x4096xbf16>
    %2725 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_332 : tensor<4096x4096xbf16>) outs(%2724 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2726 = tosa.reshape %2723 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1128 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2727 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2726, %2725 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1128 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2728 = tosa.reshape %2727 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2729 = tensor.empty() : tensor<4096x4096xbf16>
    %2730 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_334 : tensor<4096x4096xbf16>) outs(%2729 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2731 = tosa.reshape %2723 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1129 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2732 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2731, %2730 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1129 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2733 = tosa.reshape %2732 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2734 = tensor.empty() : tensor<4096x4096xbf16>
    %2735 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_336 : tensor<4096x4096xbf16>) outs(%2734 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2736 = tosa.reshape %2723 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1130 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2737 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2736, %2735 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1130 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2738 = tosa.reshape %2737 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2739 = tosa.reshape %2728 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2740 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2741 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2739 : tensor<1x40x32x128xbf16>) outs(%2740 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2742 = tosa.reshape %2733 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2743 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2744 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2742 : tensor<1x40x32x128xbf16>) outs(%2743 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2745 = tosa.reshape %2738 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2746 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2747 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2745 : tensor<1x40x32x128xbf16>) outs(%2746 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1131 = tensor.extract_slice %expanded_592[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1132 = tensor.extract_slice %extracted_slice_1131[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1133 = tensor.extract_slice %extracted_slice_1132[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1134 = tensor.extract_slice %expanded_594[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1135 = tensor.extract_slice %extracted_slice_1134[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1136 = tensor.extract_slice %extracted_slice_1135[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %2748 = tensor.empty() : tensor<1x40x128xbf16>
    %2749  = tensor.collapse_shape %extracted_slice_1133 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2750 = tensor.empty() : tensor<40x128xbf16>
    %2751 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2749 : tensor<1x40x128xbf16>) outs(%2750 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2752 = tensor.empty() : tensor<1x40x128xbf16>
    %2753  = tensor.collapse_shape %extracted_slice_1136 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2754 = tensor.empty() : tensor<40x128xbf16>
    %2755 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2753 : tensor<1x40x128xbf16>) outs(%2754 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2756 = tensor.empty() : tensor<1x40x128xbf16>
    %2757 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2756 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2751[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2758 = tosa.reshape %2757 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2759 = tensor.empty() : tensor<1x40x128xbf16>
    %2760 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2759 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2755[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2761 = tosa.reshape %2760 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2762 = tosa.mul %2741, %2758 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1137 = tensor.extract_slice %2741[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1138 = tensor.extract_slice %2741[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2763 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2764 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1138 : tensor<1x32x40x64xbf16>) outs(%2763 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2765 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1139 = tensor.insert_slice %2764 into %2765[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1140 = tensor.insert_slice %extracted_slice_1137 into %inserted_slice_1139[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2766 = tosa.mul %inserted_slice_1140, %2761 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2767 = tosa.add %2762, %2766 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2768 = tosa.mul %2744, %2758 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1141 = tensor.extract_slice %2744[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1142 = tensor.extract_slice %2744[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2769 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2770 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1142 : tensor<1x32x40x64xbf16>) outs(%2769 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2771 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1143 = tensor.insert_slice %2770 into %2771[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1144 = tensor.insert_slice %extracted_slice_1141 into %inserted_slice_1143[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2772 = tosa.mul %inserted_slice_1144, %2761 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2773 = tosa.add %2768, %2772 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2774 = tensor.empty() : tensor<1x32x128x40xbf16>
    %2775 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2773 : tensor<1x32x40x128xbf16>) outs(%2774 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %2776 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2777 = tosa.add %2767, %2776 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2778 = tosa.reshape %2777 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2779 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %2780 = tosa.add %2775, %2779 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %2781 = tosa.reshape %2780 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %2782 = tosa.matmul %2778, %2781 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %2783 = tosa.reshape %2782 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2784 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2785 = tosa.reciprocal %2784 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2786 = tosa.mul %2783, %2785 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2787 = tosa.add %2786, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2788 = tensor.empty() : tensor<1x32x40x40xf32>
    %2789 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2787 : tensor<1x32x40x40xbf16>) outs(%2788 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2790 = tensor.empty() : tensor<1x32x40x1xf32>
    %2791 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2790 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2792 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2789 : tensor<1x32x40x40xf32>) outs(%2790 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2793 = tensor.empty() : tensor<1x32x40x40xf32>
    %2794 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2789, %2792 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2793 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %2795 = tensor.empty() : tensor<1x32x40x1xf32>
    %2796 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2795 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2797 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2794 : tensor<1x32x40x40xf32>) outs(%2796 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2798 = tensor.empty() : tensor<1x32x40x40xf32>
    %2799 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2794, %2797 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2798 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2800 = tensor.empty() : tensor<1x32x40x40xbf16>
    %2801 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2799 : tensor<1x32x40x40xf32>) outs(%2800 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %2802 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2803 = tosa.add %2801, %2802 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2804 = tosa.reshape %2803 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %2805 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2806 = tosa.add %2747, %2805 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2807 = tosa.reshape %2806 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2808 = tosa.matmul %2804, %2807 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2809 = tosa.reshape %2808 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2810 = tensor.empty() : tensor<1x40x32x128xbf16>
    %2811 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2809 : tensor<1x32x40x128xbf16>) outs(%2810 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %2812 = tosa.identity %2811 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %2813 = tosa.reshape %2812 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %2814 = tensor.empty() : tensor<4096x4096xbf16>
    %2815 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_338 : tensor<4096x4096xbf16>) outs(%2814 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2816 = tosa.reshape %2813 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1145 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2817 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2816, %2815 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1145 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2818 = tosa.reshape %2817 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2819 = tosa.add %2710, %2818 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2820 = tensor.empty() : tensor<1x40x4096xf32>
    %2821 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2819 : tensor<1x40x4096xbf16>) outs(%2820 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2822 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1146 = arith.constant 2 : i32
    %2823 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2821 : tensor<1x40x4096xf32>) outs(%2822 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1146 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1147 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2824 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2823 : tensor<1x40x4096xf32>) outs(%cst_1147 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2825 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2826 = tosa.add %2824, %2825 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2827 = tosa.rsqrt %2826 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2828 = tosa.mul %2821, %2827 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2829 = tensor.empty() : tensor<1x40x4096xbf16>
    %2830 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2828 : tensor<1x40x4096xf32>) outs(%2829 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2831 = tosa.reshape %extracted_slice_38 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2832 = tosa.mul %2831, %2830 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2833 = tensor.empty() : tensor<4096x11008xbf16>
    %2834 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_340 : tensor<11008x4096xbf16>) outs(%2833 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2835 = tosa.reshape %2832 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1148 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2836 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2835, %2834 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1148 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2837 = tosa.reshape %2836 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2838 = tensor.empty() : tensor<1x40x11008xbf16>
    %2839 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2837 : tensor<1x40x11008xbf16>) outs(%2838 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %2840 = tensor.empty() : tensor<4096x11008xbf16>
    %2841 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_342 : tensor<11008x4096xbf16>) outs(%2840 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2842 = tosa.reshape %2832 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1149 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2843 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2842, %2841 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1149 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2844 = tosa.reshape %2843 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2845 = tosa.mul %2839, %2844 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2846 = tensor.empty() : tensor<11008x4096xbf16>
    %2847 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_344 : tensor<4096x11008xbf16>) outs(%2846 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %2848 = tosa.reshape %2845 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1150 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2849 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2848, %2847 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1150 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2850 = tosa.reshape %2849 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2851 = tosa.add %2819, %2850 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2852 = tensor.empty() : tensor<1x40x4096xf32>
    %2853 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2851 : tensor<1x40x4096xbf16>) outs(%2852 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2854 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1151 = arith.constant 2 : i32
    %2855 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2853 : tensor<1x40x4096xf32>) outs(%2854 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1151 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1152 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2856 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2855 : tensor<1x40x4096xf32>) outs(%cst_1152 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2857 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2858 = tosa.add %2856, %2857 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2859 = tosa.rsqrt %2858 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2860 = tosa.mul %2853, %2859 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2861 = tensor.empty() : tensor<1x40x4096xbf16>
    %2862 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2860 : tensor<1x40x4096xf32>) outs(%2861 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2863 = tosa.reshape %extracted_slice_39 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2864 = tosa.mul %2863, %2862 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2865 = tensor.empty() : tensor<4096x4096xbf16>
    %2866 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_346 : tensor<4096x4096xbf16>) outs(%2865 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2867 = tosa.reshape %2864 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1153 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2868 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2867, %2866 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1153 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2869 = tosa.reshape %2868 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2870 = tensor.empty() : tensor<4096x4096xbf16>
    %2871 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_348 : tensor<4096x4096xbf16>) outs(%2870 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2872 = tosa.reshape %2864 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1154 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2873 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2872, %2871 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1154 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2874 = tosa.reshape %2873 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2875 = tensor.empty() : tensor<4096x4096xbf16>
    %2876 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_350 : tensor<4096x4096xbf16>) outs(%2875 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2877 = tosa.reshape %2864 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1155 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2878 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2877, %2876 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1155 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2879 = tosa.reshape %2878 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2880 = tosa.reshape %2869 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2881 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2882 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2880 : tensor<1x40x32x128xbf16>) outs(%2881 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2883 = tosa.reshape %2874 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2884 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2885 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2883 : tensor<1x40x32x128xbf16>) outs(%2884 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %2886 = tosa.reshape %2879 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %2887 = tensor.empty() : tensor<1x32x40x128xbf16>
    %2888 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2886 : tensor<1x40x32x128xbf16>) outs(%2887 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1156 = tensor.extract_slice %expanded_596[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1157 = tensor.extract_slice %extracted_slice_1156[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1158 = tensor.extract_slice %extracted_slice_1157[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1159 = tensor.extract_slice %expanded_598[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1160 = tensor.extract_slice %extracted_slice_1159[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1161 = tensor.extract_slice %extracted_slice_1160[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %2889 = tensor.empty() : tensor<1x40x128xbf16>
    %2890  = tensor.collapse_shape %extracted_slice_1158 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2891 = tensor.empty() : tensor<40x128xbf16>
    %2892 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2890 : tensor<1x40x128xbf16>) outs(%2891 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2893 = tensor.empty() : tensor<1x40x128xbf16>
    %2894  = tensor.collapse_shape %extracted_slice_1161 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %2895 = tensor.empty() : tensor<40x128xbf16>
    %2896 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2894 : tensor<1x40x128xbf16>) outs(%2895 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %2897 = tensor.empty() : tensor<1x40x128xbf16>
    %2898 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2897 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2892[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2899 = tosa.reshape %2898 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2900 = tensor.empty() : tensor<1x40x128xbf16>
    %2901 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%2900 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %2896[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %2902 = tosa.reshape %2901 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %2903 = tosa.mul %2882, %2899 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1162 = tensor.extract_slice %2882[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1163 = tensor.extract_slice %2882[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2904 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2905 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1163 : tensor<1x32x40x64xbf16>) outs(%2904 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2906 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1164 = tensor.insert_slice %2905 into %2906[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1165 = tensor.insert_slice %extracted_slice_1162 into %inserted_slice_1164[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2907 = tosa.mul %inserted_slice_1165, %2902 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2908 = tosa.add %2903, %2907 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2909 = tosa.mul %2885, %2899 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1166 = tensor.extract_slice %2885[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1167 = tensor.extract_slice %2885[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %2910 = tensor.empty() : tensor<1x32x40x64xbf16>
    %2911 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1167 : tensor<1x32x40x64xbf16>) outs(%2910 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %2912 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1168 = tensor.insert_slice %2911 into %2912[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1169 = tensor.insert_slice %extracted_slice_1166 into %inserted_slice_1168[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %2913 = tosa.mul %inserted_slice_1169, %2902 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2914 = tosa.add %2909, %2913 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2915 = tensor.empty() : tensor<1x32x128x40xbf16>
    %2916 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2914 : tensor<1x32x40x128xbf16>) outs(%2915 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %2917 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2918 = tosa.add %2908, %2917 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2919 = tosa.reshape %2918 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2920 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %2921 = tosa.add %2916, %2920 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %2922 = tosa.reshape %2921 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %2923 = tosa.matmul %2919, %2922 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %2924 = tosa.reshape %2923 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2925 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2926 = tosa.reciprocal %2925 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2927 = tosa.mul %2924, %2926 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2928 = tosa.add %2927, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2929 = tensor.empty() : tensor<1x32x40x40xf32>
    %2930 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2928 : tensor<1x32x40x40xbf16>) outs(%2929 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2931 = tensor.empty() : tensor<1x32x40x1xf32>
    %2932 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2931 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2933 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2930 : tensor<1x32x40x40xf32>) outs(%2931 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2934 = tensor.empty() : tensor<1x32x40x40xf32>
    %2935 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2930, %2933 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2934 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %2936 = tensor.empty() : tensor<1x32x40x1xf32>
    %2937 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2936 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %2938 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2935 : tensor<1x32x40x40xf32>) outs(%2937 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %2939 = tensor.empty() : tensor<1x32x40x40xf32>
    %2940 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2935, %2938 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%2939 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %2941 = tensor.empty() : tensor<1x32x40x40xbf16>
    %2942 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2940 : tensor<1x32x40x40xf32>) outs(%2941 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %2943 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %2944 = tosa.add %2942, %2943 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %2945 = tosa.reshape %2944 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %2946 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %2947 = tosa.add %2888, %2946 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2948 = tosa.reshape %2947 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2949 = tosa.matmul %2945, %2948 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %2950 = tosa.reshape %2949 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %2951 = tensor.empty() : tensor<1x40x32x128xbf16>
    %2952 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2950 : tensor<1x32x40x128xbf16>) outs(%2951 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %2953 = tosa.identity %2952 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %2954 = tosa.reshape %2953 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %2955 = tensor.empty() : tensor<4096x4096xbf16>
    %2956 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_352 : tensor<4096x4096xbf16>) outs(%2955 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2957 = tosa.reshape %2954 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1170 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2958 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2957, %2956 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1170 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2959 = tosa.reshape %2958 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2960 = tosa.add %2851, %2959 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2961 = tensor.empty() : tensor<1x40x4096xf32>
    %2962 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2960 : tensor<1x40x4096xbf16>) outs(%2961 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2963 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1171 = arith.constant 2 : i32
    %2964 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2962 : tensor<1x40x4096xf32>) outs(%2963 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1171 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1172 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2965 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2964 : tensor<1x40x4096xf32>) outs(%cst_1172 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2966 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2967 = tosa.add %2965, %2966 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2968 = tosa.rsqrt %2967 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2969 = tosa.mul %2962, %2968 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %2970 = tensor.empty() : tensor<1x40x4096xbf16>
    %2971 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2969 : tensor<1x40x4096xf32>) outs(%2970 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %2972 = tosa.reshape %extracted_slice_40 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %2973 = tosa.mul %2972, %2971 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2974 = tensor.empty() : tensor<4096x11008xbf16>
    %2975 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_354 : tensor<11008x4096xbf16>) outs(%2974 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2976 = tosa.reshape %2973 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1173 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2977 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2976, %2975 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1173 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2978 = tosa.reshape %2977 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2979 = tensor.empty() : tensor<1x40x11008xbf16>
    %2980 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2978 : tensor<1x40x11008xbf16>) outs(%2979 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %2981 = tensor.empty() : tensor<4096x11008xbf16>
    %2982 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_356 : tensor<11008x4096xbf16>) outs(%2981 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %2983 = tosa.reshape %2973 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1174 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %2984 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2983, %2982 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1174 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %2985 = tosa.reshape %2984 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2986 = tosa.mul %2980, %2985 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %2987 = tensor.empty() : tensor<11008x4096xbf16>
    %2988 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_358 : tensor<4096x11008xbf16>) outs(%2987 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %2989 = tosa.reshape %2986 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1175 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %2990 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2989, %2988 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1175 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %2991 = tosa.reshape %2990 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2992 = tosa.add %2960, %2991 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %2993 = tensor.empty() : tensor<1x40x4096xf32>
    %2994 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2992 : tensor<1x40x4096xbf16>) outs(%2993 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %2995 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1176 = arith.constant 2 : i32
    %2996 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2994 : tensor<1x40x4096xf32>) outs(%2995 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1176 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1177 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %2997 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2996 : tensor<1x40x4096xf32>) outs(%cst_1177 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %2998 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2999 = tosa.add %2997, %2998 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3000 = tosa.rsqrt %2999 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3001 = tosa.mul %2994, %3000 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3002 = tensor.empty() : tensor<1x40x4096xbf16>
    %3003 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3001 : tensor<1x40x4096xf32>) outs(%3002 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3004 = tosa.reshape %extracted_slice_41 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3005 = tosa.mul %3004, %3003 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3006 = tensor.empty() : tensor<4096x4096xbf16>
    %3007 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_360 : tensor<4096x4096xbf16>) outs(%3006 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3008 = tosa.reshape %3005 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1178 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3009 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3008, %3007 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1178 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3010 = tosa.reshape %3009 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3011 = tensor.empty() : tensor<4096x4096xbf16>
    %3012 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_362 : tensor<4096x4096xbf16>) outs(%3011 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3013 = tosa.reshape %3005 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1179 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3014 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3013, %3012 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1179 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3015 = tosa.reshape %3014 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3016 = tensor.empty() : tensor<4096x4096xbf16>
    %3017 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_364 : tensor<4096x4096xbf16>) outs(%3016 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3018 = tosa.reshape %3005 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1180 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3019 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3018, %3017 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1180 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3020 = tosa.reshape %3019 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3021 = tosa.reshape %3010 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3022 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3023 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3021 : tensor<1x40x32x128xbf16>) outs(%3022 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3024 = tosa.reshape %3015 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3025 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3026 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3024 : tensor<1x40x32x128xbf16>) outs(%3025 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3027 = tosa.reshape %3020 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3028 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3029 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3027 : tensor<1x40x32x128xbf16>) outs(%3028 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1181 = tensor.extract_slice %expanded_600[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1182 = tensor.extract_slice %extracted_slice_1181[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1183 = tensor.extract_slice %extracted_slice_1182[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1184 = tensor.extract_slice %expanded_602[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1185 = tensor.extract_slice %extracted_slice_1184[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1186 = tensor.extract_slice %extracted_slice_1185[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %3030 = tensor.empty() : tensor<1x40x128xbf16>
    %3031  = tensor.collapse_shape %extracted_slice_1183 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3032 = tensor.empty() : tensor<40x128xbf16>
    %3033 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3031 : tensor<1x40x128xbf16>) outs(%3032 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3034 = tensor.empty() : tensor<1x40x128xbf16>
    %3035  = tensor.collapse_shape %extracted_slice_1186 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3036 = tensor.empty() : tensor<40x128xbf16>
    %3037 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3035 : tensor<1x40x128xbf16>) outs(%3036 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3038 = tensor.empty() : tensor<1x40x128xbf16>
    %3039 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3038 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3033[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3040 = tosa.reshape %3039 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3041 = tensor.empty() : tensor<1x40x128xbf16>
    %3042 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3041 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3037[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3043 = tosa.reshape %3042 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3044 = tosa.mul %3023, %3040 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1187 = tensor.extract_slice %3023[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1188 = tensor.extract_slice %3023[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3045 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3046 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1188 : tensor<1x32x40x64xbf16>) outs(%3045 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3047 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1189 = tensor.insert_slice %3046 into %3047[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1190 = tensor.insert_slice %extracted_slice_1187 into %inserted_slice_1189[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3048 = tosa.mul %inserted_slice_1190, %3043 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3049 = tosa.add %3044, %3048 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3050 = tosa.mul %3026, %3040 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1191 = tensor.extract_slice %3026[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1192 = tensor.extract_slice %3026[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3051 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3052 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1192 : tensor<1x32x40x64xbf16>) outs(%3051 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3053 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1193 = tensor.insert_slice %3052 into %3053[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1194 = tensor.insert_slice %extracted_slice_1191 into %inserted_slice_1193[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3054 = tosa.mul %inserted_slice_1194, %3043 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3055 = tosa.add %3050, %3054 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3056 = tensor.empty() : tensor<1x32x128x40xbf16>
    %3057 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3055 : tensor<1x32x40x128xbf16>) outs(%3056 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %3058 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3059 = tosa.add %3049, %3058 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3060 = tosa.reshape %3059 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3061 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %3062 = tosa.add %3057, %3061 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %3063 = tosa.reshape %3062 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %3064 = tosa.matmul %3060, %3063 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %3065 = tosa.reshape %3064 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3066 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3067 = tosa.reciprocal %3066 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3068 = tosa.mul %3065, %3067 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3069 = tosa.add %3068, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3070 = tensor.empty() : tensor<1x32x40x40xf32>
    %3071 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3069 : tensor<1x32x40x40xbf16>) outs(%3070 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3072 = tensor.empty() : tensor<1x32x40x1xf32>
    %3073 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3072 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3074 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3071 : tensor<1x32x40x40xf32>) outs(%3072 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3075 = tensor.empty() : tensor<1x32x40x40xf32>
    %3076 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3071, %3074 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3075 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %3077 = tensor.empty() : tensor<1x32x40x1xf32>
    %3078 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3077 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3079 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3076 : tensor<1x32x40x40xf32>) outs(%3078 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3080 = tensor.empty() : tensor<1x32x40x40xf32>
    %3081 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3076, %3079 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3080 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3082 = tensor.empty() : tensor<1x32x40x40xbf16>
    %3083 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3081 : tensor<1x32x40x40xf32>) outs(%3082 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %3084 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3085 = tosa.add %3083, %3084 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3086 = tosa.reshape %3085 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %3087 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3088 = tosa.add %3029, %3087 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3089 = tosa.reshape %3088 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3090 = tosa.matmul %3086, %3089 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3091 = tosa.reshape %3090 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3092 = tensor.empty() : tensor<1x40x32x128xbf16>
    %3093 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3091 : tensor<1x32x40x128xbf16>) outs(%3092 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %3094 = tosa.identity %3093 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %3095 = tosa.reshape %3094 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %3096 = tensor.empty() : tensor<4096x4096xbf16>
    %3097 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_366 : tensor<4096x4096xbf16>) outs(%3096 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3098 = tosa.reshape %3095 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1195 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3099 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3098, %3097 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1195 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3100 = tosa.reshape %3099 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3101 = tosa.add %2992, %3100 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3102 = tensor.empty() : tensor<1x40x4096xf32>
    %3103 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3101 : tensor<1x40x4096xbf16>) outs(%3102 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3104 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1196 = arith.constant 2 : i32
    %3105 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3103 : tensor<1x40x4096xf32>) outs(%3104 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1196 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1197 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3106 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3105 : tensor<1x40x4096xf32>) outs(%cst_1197 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3107 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3108 = tosa.add %3106, %3107 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3109 = tosa.rsqrt %3108 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3110 = tosa.mul %3103, %3109 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3111 = tensor.empty() : tensor<1x40x4096xbf16>
    %3112 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3110 : tensor<1x40x4096xf32>) outs(%3111 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3113 = tosa.reshape %extracted_slice_42 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3114 = tosa.mul %3113, %3112 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3115 = tensor.empty() : tensor<4096x11008xbf16>
    %3116 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_368 : tensor<11008x4096xbf16>) outs(%3115 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3117 = tosa.reshape %3114 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1198 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3118 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3117, %3116 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1198 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3119 = tosa.reshape %3118 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3120 = tensor.empty() : tensor<1x40x11008xbf16>
    %3121 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3119 : tensor<1x40x11008xbf16>) outs(%3120 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %3122 = tensor.empty() : tensor<4096x11008xbf16>
    %3123 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_370 : tensor<11008x4096xbf16>) outs(%3122 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3124 = tosa.reshape %3114 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1199 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3125 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3124, %3123 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1199 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3126 = tosa.reshape %3125 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3127 = tosa.mul %3121, %3126 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3128 = tensor.empty() : tensor<11008x4096xbf16>
    %3129 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_372 : tensor<4096x11008xbf16>) outs(%3128 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %3130 = tosa.reshape %3127 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1200 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3131 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3130, %3129 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1200 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3132 = tosa.reshape %3131 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3133 = tosa.add %3101, %3132 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3134 = tensor.empty() : tensor<1x40x4096xf32>
    %3135 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3133 : tensor<1x40x4096xbf16>) outs(%3134 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3136 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1201 = arith.constant 2 : i32
    %3137 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3135 : tensor<1x40x4096xf32>) outs(%3136 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1201 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1202 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3138 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3137 : tensor<1x40x4096xf32>) outs(%cst_1202 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3139 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3140 = tosa.add %3138, %3139 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3141 = tosa.rsqrt %3140 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3142 = tosa.mul %3135, %3141 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3143 = tensor.empty() : tensor<1x40x4096xbf16>
    %3144 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3142 : tensor<1x40x4096xf32>) outs(%3143 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3145 = tosa.reshape %extracted_slice_43 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3146 = tosa.mul %3145, %3144 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3147 = tensor.empty() : tensor<4096x4096xbf16>
    %3148 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_374 : tensor<4096x4096xbf16>) outs(%3147 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3149 = tosa.reshape %3146 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1203 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3150 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3149, %3148 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1203 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3151 = tosa.reshape %3150 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3152 = tensor.empty() : tensor<4096x4096xbf16>
    %3153 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_376 : tensor<4096x4096xbf16>) outs(%3152 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3154 = tosa.reshape %3146 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1204 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3155 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3154, %3153 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1204 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3156 = tosa.reshape %3155 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3157 = tensor.empty() : tensor<4096x4096xbf16>
    %3158 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_378 : tensor<4096x4096xbf16>) outs(%3157 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3159 = tosa.reshape %3146 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1205 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3160 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3159, %3158 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1205 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3161 = tosa.reshape %3160 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3162 = tosa.reshape %3151 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3163 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3164 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3162 : tensor<1x40x32x128xbf16>) outs(%3163 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3165 = tosa.reshape %3156 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3166 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3167 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3165 : tensor<1x40x32x128xbf16>) outs(%3166 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3168 = tosa.reshape %3161 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3169 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3170 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3168 : tensor<1x40x32x128xbf16>) outs(%3169 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1206 = tensor.extract_slice %expanded_604[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1207 = tensor.extract_slice %extracted_slice_1206[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1208 = tensor.extract_slice %extracted_slice_1207[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1209 = tensor.extract_slice %expanded_606[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1210 = tensor.extract_slice %extracted_slice_1209[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1211 = tensor.extract_slice %extracted_slice_1210[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %3171 = tensor.empty() : tensor<1x40x128xbf16>
    %3172  = tensor.collapse_shape %extracted_slice_1208 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3173 = tensor.empty() : tensor<40x128xbf16>
    %3174 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3172 : tensor<1x40x128xbf16>) outs(%3173 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3175 = tensor.empty() : tensor<1x40x128xbf16>
    %3176  = tensor.collapse_shape %extracted_slice_1211 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3177 = tensor.empty() : tensor<40x128xbf16>
    %3178 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3176 : tensor<1x40x128xbf16>) outs(%3177 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3179 = tensor.empty() : tensor<1x40x128xbf16>
    %3180 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3179 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3174[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3181 = tosa.reshape %3180 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3182 = tensor.empty() : tensor<1x40x128xbf16>
    %3183 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3182 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3178[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3184 = tosa.reshape %3183 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3185 = tosa.mul %3164, %3181 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1212 = tensor.extract_slice %3164[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1213 = tensor.extract_slice %3164[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3186 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3187 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1213 : tensor<1x32x40x64xbf16>) outs(%3186 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3188 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1214 = tensor.insert_slice %3187 into %3188[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1215 = tensor.insert_slice %extracted_slice_1212 into %inserted_slice_1214[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3189 = tosa.mul %inserted_slice_1215, %3184 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3190 = tosa.add %3185, %3189 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3191 = tosa.mul %3167, %3181 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1216 = tensor.extract_slice %3167[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1217 = tensor.extract_slice %3167[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3192 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3193 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1217 : tensor<1x32x40x64xbf16>) outs(%3192 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3194 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1218 = tensor.insert_slice %3193 into %3194[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1219 = tensor.insert_slice %extracted_slice_1216 into %inserted_slice_1218[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3195 = tosa.mul %inserted_slice_1219, %3184 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3196 = tosa.add %3191, %3195 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3197 = tensor.empty() : tensor<1x32x128x40xbf16>
    %3198 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3196 : tensor<1x32x40x128xbf16>) outs(%3197 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %3199 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3200 = tosa.add %3190, %3199 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3201 = tosa.reshape %3200 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3202 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %3203 = tosa.add %3198, %3202 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %3204 = tosa.reshape %3203 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %3205 = tosa.matmul %3201, %3204 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %3206 = tosa.reshape %3205 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3207 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3208 = tosa.reciprocal %3207 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3209 = tosa.mul %3206, %3208 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3210 = tosa.add %3209, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3211 = tensor.empty() : tensor<1x32x40x40xf32>
    %3212 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3210 : tensor<1x32x40x40xbf16>) outs(%3211 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3213 = tensor.empty() : tensor<1x32x40x1xf32>
    %3214 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3213 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3215 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3212 : tensor<1x32x40x40xf32>) outs(%3213 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3216 = tensor.empty() : tensor<1x32x40x40xf32>
    %3217 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3212, %3215 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3216 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %3218 = tensor.empty() : tensor<1x32x40x1xf32>
    %3219 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3218 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3220 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3217 : tensor<1x32x40x40xf32>) outs(%3219 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3221 = tensor.empty() : tensor<1x32x40x40xf32>
    %3222 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3217, %3220 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3221 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3223 = tensor.empty() : tensor<1x32x40x40xbf16>
    %3224 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3222 : tensor<1x32x40x40xf32>) outs(%3223 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %3225 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3226 = tosa.add %3224, %3225 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3227 = tosa.reshape %3226 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %3228 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3229 = tosa.add %3170, %3228 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3230 = tosa.reshape %3229 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3231 = tosa.matmul %3227, %3230 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3232 = tosa.reshape %3231 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3233 = tensor.empty() : tensor<1x40x32x128xbf16>
    %3234 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3232 : tensor<1x32x40x128xbf16>) outs(%3233 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %3235 = tosa.identity %3234 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %3236 = tosa.reshape %3235 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %3237 = tensor.empty() : tensor<4096x4096xbf16>
    %3238 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_380 : tensor<4096x4096xbf16>) outs(%3237 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3239 = tosa.reshape %3236 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1220 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3240 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3239, %3238 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1220 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3241 = tosa.reshape %3240 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3242 = tosa.add %3133, %3241 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3243 = tensor.empty() : tensor<1x40x4096xf32>
    %3244 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3242 : tensor<1x40x4096xbf16>) outs(%3243 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3245 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1221 = arith.constant 2 : i32
    %3246 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3244 : tensor<1x40x4096xf32>) outs(%3245 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1221 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1222 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3247 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3246 : tensor<1x40x4096xf32>) outs(%cst_1222 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3248 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3249 = tosa.add %3247, %3248 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3250 = tosa.rsqrt %3249 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3251 = tosa.mul %3244, %3250 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3252 = tensor.empty() : tensor<1x40x4096xbf16>
    %3253 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3251 : tensor<1x40x4096xf32>) outs(%3252 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3254 = tosa.reshape %extracted_slice_44 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3255 = tosa.mul %3254, %3253 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3256 = tensor.empty() : tensor<4096x11008xbf16>
    %3257 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_382 : tensor<11008x4096xbf16>) outs(%3256 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3258 = tosa.reshape %3255 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1223 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3259 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3258, %3257 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1223 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3260 = tosa.reshape %3259 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3261 = tensor.empty() : tensor<1x40x11008xbf16>
    %3262 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3260 : tensor<1x40x11008xbf16>) outs(%3261 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %3263 = tensor.empty() : tensor<4096x11008xbf16>
    %3264 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_384 : tensor<11008x4096xbf16>) outs(%3263 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3265 = tosa.reshape %3255 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1224 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3266 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3265, %3264 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1224 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3267 = tosa.reshape %3266 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3268 = tosa.mul %3262, %3267 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3269 = tensor.empty() : tensor<11008x4096xbf16>
    %3270 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_386 : tensor<4096x11008xbf16>) outs(%3269 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %3271 = tosa.reshape %3268 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1225 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3272 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3271, %3270 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1225 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3273 = tosa.reshape %3272 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3274 = tosa.add %3242, %3273 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3275 = tensor.empty() : tensor<1x40x4096xf32>
    %3276 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3274 : tensor<1x40x4096xbf16>) outs(%3275 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3277 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1226 = arith.constant 2 : i32
    %3278 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3276 : tensor<1x40x4096xf32>) outs(%3277 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1226 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1227 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3279 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3278 : tensor<1x40x4096xf32>) outs(%cst_1227 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3280 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3281 = tosa.add %3279, %3280 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3282 = tosa.rsqrt %3281 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3283 = tosa.mul %3276, %3282 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3284 = tensor.empty() : tensor<1x40x4096xbf16>
    %3285 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3283 : tensor<1x40x4096xf32>) outs(%3284 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3286 = tosa.reshape %extracted_slice_45 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3287 = tosa.mul %3286, %3285 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3288 = tensor.empty() : tensor<4096x4096xbf16>
    %3289 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_388 : tensor<4096x4096xbf16>) outs(%3288 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3290 = tosa.reshape %3287 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1228 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3291 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3290, %3289 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1228 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3292 = tosa.reshape %3291 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3293 = tensor.empty() : tensor<4096x4096xbf16>
    %3294 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_390 : tensor<4096x4096xbf16>) outs(%3293 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3295 = tosa.reshape %3287 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1229 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3296 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3295, %3294 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1229 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3297 = tosa.reshape %3296 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3298 = tensor.empty() : tensor<4096x4096xbf16>
    %3299 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_392 : tensor<4096x4096xbf16>) outs(%3298 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3300 = tosa.reshape %3287 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1230 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3301 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3300, %3299 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1230 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3302 = tosa.reshape %3301 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3303 = tosa.reshape %3292 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3304 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3305 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3303 : tensor<1x40x32x128xbf16>) outs(%3304 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3306 = tosa.reshape %3297 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3307 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3308 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3306 : tensor<1x40x32x128xbf16>) outs(%3307 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3309 = tosa.reshape %3302 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3310 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3311 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3309 : tensor<1x40x32x128xbf16>) outs(%3310 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1231 = tensor.extract_slice %expanded_608[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1232 = tensor.extract_slice %extracted_slice_1231[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1233 = tensor.extract_slice %extracted_slice_1232[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1234 = tensor.extract_slice %expanded_610[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1235 = tensor.extract_slice %extracted_slice_1234[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1236 = tensor.extract_slice %extracted_slice_1235[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %3312 = tensor.empty() : tensor<1x40x128xbf16>
    %3313  = tensor.collapse_shape %extracted_slice_1233 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3314 = tensor.empty() : tensor<40x128xbf16>
    %3315 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3313 : tensor<1x40x128xbf16>) outs(%3314 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3316 = tensor.empty() : tensor<1x40x128xbf16>
    %3317  = tensor.collapse_shape %extracted_slice_1236 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3318 = tensor.empty() : tensor<40x128xbf16>
    %3319 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3317 : tensor<1x40x128xbf16>) outs(%3318 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3320 = tensor.empty() : tensor<1x40x128xbf16>
    %3321 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3320 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3315[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3322 = tosa.reshape %3321 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3323 = tensor.empty() : tensor<1x40x128xbf16>
    %3324 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3323 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3319[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3325 = tosa.reshape %3324 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3326 = tosa.mul %3305, %3322 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1237 = tensor.extract_slice %3305[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1238 = tensor.extract_slice %3305[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3327 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3328 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1238 : tensor<1x32x40x64xbf16>) outs(%3327 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3329 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1239 = tensor.insert_slice %3328 into %3329[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1240 = tensor.insert_slice %extracted_slice_1237 into %inserted_slice_1239[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3330 = tosa.mul %inserted_slice_1240, %3325 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3331 = tosa.add %3326, %3330 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3332 = tosa.mul %3308, %3322 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1241 = tensor.extract_slice %3308[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1242 = tensor.extract_slice %3308[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3333 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3334 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1242 : tensor<1x32x40x64xbf16>) outs(%3333 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3335 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1243 = tensor.insert_slice %3334 into %3335[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1244 = tensor.insert_slice %extracted_slice_1241 into %inserted_slice_1243[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3336 = tosa.mul %inserted_slice_1244, %3325 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3337 = tosa.add %3332, %3336 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3338 = tensor.empty() : tensor<1x32x128x40xbf16>
    %3339 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3337 : tensor<1x32x40x128xbf16>) outs(%3338 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %3340 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3341 = tosa.add %3331, %3340 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3342 = tosa.reshape %3341 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3343 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %3344 = tosa.add %3339, %3343 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %3345 = tosa.reshape %3344 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %3346 = tosa.matmul %3342, %3345 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %3347 = tosa.reshape %3346 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3348 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3349 = tosa.reciprocal %3348 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3350 = tosa.mul %3347, %3349 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3351 = tosa.add %3350, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3352 = tensor.empty() : tensor<1x32x40x40xf32>
    %3353 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3351 : tensor<1x32x40x40xbf16>) outs(%3352 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3354 = tensor.empty() : tensor<1x32x40x1xf32>
    %3355 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3354 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3356 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3353 : tensor<1x32x40x40xf32>) outs(%3354 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3357 = tensor.empty() : tensor<1x32x40x40xf32>
    %3358 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3353, %3356 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3357 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %3359 = tensor.empty() : tensor<1x32x40x1xf32>
    %3360 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3359 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3361 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3358 : tensor<1x32x40x40xf32>) outs(%3360 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3362 = tensor.empty() : tensor<1x32x40x40xf32>
    %3363 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3358, %3361 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3362 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3364 = tensor.empty() : tensor<1x32x40x40xbf16>
    %3365 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3363 : tensor<1x32x40x40xf32>) outs(%3364 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %3366 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3367 = tosa.add %3365, %3366 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3368 = tosa.reshape %3367 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %3369 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3370 = tosa.add %3311, %3369 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3371 = tosa.reshape %3370 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3372 = tosa.matmul %3368, %3371 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3373 = tosa.reshape %3372 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3374 = tensor.empty() : tensor<1x40x32x128xbf16>
    %3375 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3373 : tensor<1x32x40x128xbf16>) outs(%3374 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %3376 = tosa.identity %3375 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %3377 = tosa.reshape %3376 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %3378 = tensor.empty() : tensor<4096x4096xbf16>
    %3379 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_394 : tensor<4096x4096xbf16>) outs(%3378 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3380 = tosa.reshape %3377 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1245 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3381 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3380, %3379 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1245 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3382 = tosa.reshape %3381 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3383 = tosa.add %3274, %3382 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3384 = tensor.empty() : tensor<1x40x4096xf32>
    %3385 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3383 : tensor<1x40x4096xbf16>) outs(%3384 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3386 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1246 = arith.constant 2 : i32
    %3387 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3385 : tensor<1x40x4096xf32>) outs(%3386 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1246 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1247 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3388 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3387 : tensor<1x40x4096xf32>) outs(%cst_1247 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3389 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3390 = tosa.add %3388, %3389 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3391 = tosa.rsqrt %3390 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3392 = tosa.mul %3385, %3391 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3393 = tensor.empty() : tensor<1x40x4096xbf16>
    %3394 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3392 : tensor<1x40x4096xf32>) outs(%3393 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3395 = tosa.reshape %extracted_slice_46 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3396 = tosa.mul %3395, %3394 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3397 = tensor.empty() : tensor<4096x11008xbf16>
    %3398 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_396 : tensor<11008x4096xbf16>) outs(%3397 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3399 = tosa.reshape %3396 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1248 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3400 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3399, %3398 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1248 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3401 = tosa.reshape %3400 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3402 = tensor.empty() : tensor<1x40x11008xbf16>
    %3403 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3401 : tensor<1x40x11008xbf16>) outs(%3402 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %3404 = tensor.empty() : tensor<4096x11008xbf16>
    %3405 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_398 : tensor<11008x4096xbf16>) outs(%3404 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3406 = tosa.reshape %3396 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1249 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3407 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3406, %3405 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1249 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3408 = tosa.reshape %3407 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3409 = tosa.mul %3403, %3408 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3410 = tensor.empty() : tensor<11008x4096xbf16>
    %3411 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_400 : tensor<4096x11008xbf16>) outs(%3410 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %3412 = tosa.reshape %3409 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1250 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3413 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3412, %3411 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1250 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3414 = tosa.reshape %3413 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3415 = tosa.add %3383, %3414 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3416 = tensor.empty() : tensor<1x40x4096xf32>
    %3417 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3415 : tensor<1x40x4096xbf16>) outs(%3416 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3418 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1251 = arith.constant 2 : i32
    %3419 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3417 : tensor<1x40x4096xf32>) outs(%3418 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1251 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1252 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3420 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3419 : tensor<1x40x4096xf32>) outs(%cst_1252 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3421 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3422 = tosa.add %3420, %3421 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3423 = tosa.rsqrt %3422 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3424 = tosa.mul %3417, %3423 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3425 = tensor.empty() : tensor<1x40x4096xbf16>
    %3426 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3424 : tensor<1x40x4096xf32>) outs(%3425 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3427 = tosa.reshape %extracted_slice_47 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3428 = tosa.mul %3427, %3426 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3429 = tensor.empty() : tensor<4096x4096xbf16>
    %3430 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_402 : tensor<4096x4096xbf16>) outs(%3429 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3431 = tosa.reshape %3428 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1253 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3432 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3431, %3430 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1253 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3433 = tosa.reshape %3432 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3434 = tensor.empty() : tensor<4096x4096xbf16>
    %3435 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_404 : tensor<4096x4096xbf16>) outs(%3434 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3436 = tosa.reshape %3428 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1254 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3437 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3436, %3435 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1254 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3438 = tosa.reshape %3437 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3439 = tensor.empty() : tensor<4096x4096xbf16>
    %3440 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_406 : tensor<4096x4096xbf16>) outs(%3439 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3441 = tosa.reshape %3428 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1255 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3442 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3441, %3440 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1255 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3443 = tosa.reshape %3442 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3444 = tosa.reshape %3433 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3445 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3446 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3444 : tensor<1x40x32x128xbf16>) outs(%3445 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3447 = tosa.reshape %3438 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3448 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3449 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3447 : tensor<1x40x32x128xbf16>) outs(%3448 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3450 = tosa.reshape %3443 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3451 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3452 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3450 : tensor<1x40x32x128xbf16>) outs(%3451 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1256 = tensor.extract_slice %expanded_612[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1257 = tensor.extract_slice %extracted_slice_1256[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1258 = tensor.extract_slice %extracted_slice_1257[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1259 = tensor.extract_slice %expanded_614[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1260 = tensor.extract_slice %extracted_slice_1259[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1261 = tensor.extract_slice %extracted_slice_1260[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %3453 = tensor.empty() : tensor<1x40x128xbf16>
    %3454  = tensor.collapse_shape %extracted_slice_1258 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3455 = tensor.empty() : tensor<40x128xbf16>
    %3456 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3454 : tensor<1x40x128xbf16>) outs(%3455 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3457 = tensor.empty() : tensor<1x40x128xbf16>
    %3458  = tensor.collapse_shape %extracted_slice_1261 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3459 = tensor.empty() : tensor<40x128xbf16>
    %3460 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3458 : tensor<1x40x128xbf16>) outs(%3459 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3461 = tensor.empty() : tensor<1x40x128xbf16>
    %3462 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3461 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3456[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3463 = tosa.reshape %3462 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3464 = tensor.empty() : tensor<1x40x128xbf16>
    %3465 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3464 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3460[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3466 = tosa.reshape %3465 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3467 = tosa.mul %3446, %3463 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1262 = tensor.extract_slice %3446[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1263 = tensor.extract_slice %3446[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3468 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3469 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1263 : tensor<1x32x40x64xbf16>) outs(%3468 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3470 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1264 = tensor.insert_slice %3469 into %3470[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1265 = tensor.insert_slice %extracted_slice_1262 into %inserted_slice_1264[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3471 = tosa.mul %inserted_slice_1265, %3466 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3472 = tosa.add %3467, %3471 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3473 = tosa.mul %3449, %3463 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1266 = tensor.extract_slice %3449[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1267 = tensor.extract_slice %3449[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3474 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3475 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1267 : tensor<1x32x40x64xbf16>) outs(%3474 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3476 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1268 = tensor.insert_slice %3475 into %3476[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1269 = tensor.insert_slice %extracted_slice_1266 into %inserted_slice_1268[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3477 = tosa.mul %inserted_slice_1269, %3466 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3478 = tosa.add %3473, %3477 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3479 = tensor.empty() : tensor<1x32x128x40xbf16>
    %3480 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3478 : tensor<1x32x40x128xbf16>) outs(%3479 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %3481 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3482 = tosa.add %3472, %3481 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3483 = tosa.reshape %3482 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3484 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %3485 = tosa.add %3480, %3484 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %3486 = tosa.reshape %3485 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %3487 = tosa.matmul %3483, %3486 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %3488 = tosa.reshape %3487 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3489 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3490 = tosa.reciprocal %3489 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3491 = tosa.mul %3488, %3490 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3492 = tosa.add %3491, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3493 = tensor.empty() : tensor<1x32x40x40xf32>
    %3494 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3492 : tensor<1x32x40x40xbf16>) outs(%3493 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3495 = tensor.empty() : tensor<1x32x40x1xf32>
    %3496 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3495 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3497 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3494 : tensor<1x32x40x40xf32>) outs(%3495 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3498 = tensor.empty() : tensor<1x32x40x40xf32>
    %3499 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3494, %3497 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3498 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %3500 = tensor.empty() : tensor<1x32x40x1xf32>
    %3501 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3500 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3502 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3499 : tensor<1x32x40x40xf32>) outs(%3501 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3503 = tensor.empty() : tensor<1x32x40x40xf32>
    %3504 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3499, %3502 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3503 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3505 = tensor.empty() : tensor<1x32x40x40xbf16>
    %3506 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3504 : tensor<1x32x40x40xf32>) outs(%3505 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %3507 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3508 = tosa.add %3506, %3507 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3509 = tosa.reshape %3508 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %3510 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3511 = tosa.add %3452, %3510 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3512 = tosa.reshape %3511 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3513 = tosa.matmul %3509, %3512 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3514 = tosa.reshape %3513 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3515 = tensor.empty() : tensor<1x40x32x128xbf16>
    %3516 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3514 : tensor<1x32x40x128xbf16>) outs(%3515 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %3517 = tosa.identity %3516 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %3518 = tosa.reshape %3517 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %3519 = tensor.empty() : tensor<4096x4096xbf16>
    %3520 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_408 : tensor<4096x4096xbf16>) outs(%3519 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3521 = tosa.reshape %3518 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1270 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3522 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3521, %3520 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1270 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3523 = tosa.reshape %3522 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3524 = tosa.add %3415, %3523 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3525 = tensor.empty() : tensor<1x40x4096xf32>
    %3526 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3524 : tensor<1x40x4096xbf16>) outs(%3525 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3527 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1271 = arith.constant 2 : i32
    %3528 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3526 : tensor<1x40x4096xf32>) outs(%3527 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1271 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1272 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3529 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3528 : tensor<1x40x4096xf32>) outs(%cst_1272 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3530 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3531 = tosa.add %3529, %3530 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3532 = tosa.rsqrt %3531 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3533 = tosa.mul %3526, %3532 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3534 = tensor.empty() : tensor<1x40x4096xbf16>
    %3535 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3533 : tensor<1x40x4096xf32>) outs(%3534 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3536 = tosa.reshape %extracted_slice_48 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3537 = tosa.mul %3536, %3535 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3538 = tensor.empty() : tensor<4096x11008xbf16>
    %3539 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_410 : tensor<11008x4096xbf16>) outs(%3538 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3540 = tosa.reshape %3537 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1273 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3541 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3540, %3539 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1273 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3542 = tosa.reshape %3541 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3543 = tensor.empty() : tensor<1x40x11008xbf16>
    %3544 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3542 : tensor<1x40x11008xbf16>) outs(%3543 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %3545 = tensor.empty() : tensor<4096x11008xbf16>
    %3546 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_412 : tensor<11008x4096xbf16>) outs(%3545 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3547 = tosa.reshape %3537 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1274 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3548 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3547, %3546 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1274 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3549 = tosa.reshape %3548 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3550 = tosa.mul %3544, %3549 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3551 = tensor.empty() : tensor<11008x4096xbf16>
    %3552 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_414 : tensor<4096x11008xbf16>) outs(%3551 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %3553 = tosa.reshape %3550 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1275 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3554 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3553, %3552 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1275 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3555 = tosa.reshape %3554 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3556 = tosa.add %3524, %3555 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3557 = tensor.empty() : tensor<1x40x4096xf32>
    %3558 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3556 : tensor<1x40x4096xbf16>) outs(%3557 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3559 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1276 = arith.constant 2 : i32
    %3560 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3558 : tensor<1x40x4096xf32>) outs(%3559 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1276 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1277 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3561 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3560 : tensor<1x40x4096xf32>) outs(%cst_1277 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3562 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3563 = tosa.add %3561, %3562 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3564 = tosa.rsqrt %3563 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3565 = tosa.mul %3558, %3564 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3566 = tensor.empty() : tensor<1x40x4096xbf16>
    %3567 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3565 : tensor<1x40x4096xf32>) outs(%3566 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3568 = tosa.reshape %extracted_slice_49 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3569 = tosa.mul %3568, %3567 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3570 = tensor.empty() : tensor<4096x4096xbf16>
    %3571 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_416 : tensor<4096x4096xbf16>) outs(%3570 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3572 = tosa.reshape %3569 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1278 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3573 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3572, %3571 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1278 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3574 = tosa.reshape %3573 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3575 = tensor.empty() : tensor<4096x4096xbf16>
    %3576 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_418 : tensor<4096x4096xbf16>) outs(%3575 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3577 = tosa.reshape %3569 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1279 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3578 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3577, %3576 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1279 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3579 = tosa.reshape %3578 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3580 = tensor.empty() : tensor<4096x4096xbf16>
    %3581 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_420 : tensor<4096x4096xbf16>) outs(%3580 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3582 = tosa.reshape %3569 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1280 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3583 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3582, %3581 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1280 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3584 = tosa.reshape %3583 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3585 = tosa.reshape %3574 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3586 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3587 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3585 : tensor<1x40x32x128xbf16>) outs(%3586 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3588 = tosa.reshape %3579 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3589 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3590 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3588 : tensor<1x40x32x128xbf16>) outs(%3589 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3591 = tosa.reshape %3584 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3592 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3593 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3591 : tensor<1x40x32x128xbf16>) outs(%3592 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1281 = tensor.extract_slice %expanded_616[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1282 = tensor.extract_slice %extracted_slice_1281[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1283 = tensor.extract_slice %extracted_slice_1282[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1284 = tensor.extract_slice %expanded_618[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1285 = tensor.extract_slice %extracted_slice_1284[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1286 = tensor.extract_slice %extracted_slice_1285[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %3594 = tensor.empty() : tensor<1x40x128xbf16>
    %3595  = tensor.collapse_shape %extracted_slice_1283 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3596 = tensor.empty() : tensor<40x128xbf16>
    %3597 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3595 : tensor<1x40x128xbf16>) outs(%3596 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3598 = tensor.empty() : tensor<1x40x128xbf16>
    %3599  = tensor.collapse_shape %extracted_slice_1286 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3600 = tensor.empty() : tensor<40x128xbf16>
    %3601 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3599 : tensor<1x40x128xbf16>) outs(%3600 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3602 = tensor.empty() : tensor<1x40x128xbf16>
    %3603 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3602 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3597[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3604 = tosa.reshape %3603 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3605 = tensor.empty() : tensor<1x40x128xbf16>
    %3606 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3605 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3601[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3607 = tosa.reshape %3606 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3608 = tosa.mul %3587, %3604 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1287 = tensor.extract_slice %3587[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1288 = tensor.extract_slice %3587[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3609 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3610 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1288 : tensor<1x32x40x64xbf16>) outs(%3609 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3611 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1289 = tensor.insert_slice %3610 into %3611[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1290 = tensor.insert_slice %extracted_slice_1287 into %inserted_slice_1289[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3612 = tosa.mul %inserted_slice_1290, %3607 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3613 = tosa.add %3608, %3612 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3614 = tosa.mul %3590, %3604 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1291 = tensor.extract_slice %3590[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1292 = tensor.extract_slice %3590[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3615 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3616 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1292 : tensor<1x32x40x64xbf16>) outs(%3615 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3617 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1293 = tensor.insert_slice %3616 into %3617[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1294 = tensor.insert_slice %extracted_slice_1291 into %inserted_slice_1293[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3618 = tosa.mul %inserted_slice_1294, %3607 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3619 = tosa.add %3614, %3618 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3620 = tensor.empty() : tensor<1x32x128x40xbf16>
    %3621 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3619 : tensor<1x32x40x128xbf16>) outs(%3620 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %3622 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3623 = tosa.add %3613, %3622 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3624 = tosa.reshape %3623 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3625 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %3626 = tosa.add %3621, %3625 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %3627 = tosa.reshape %3626 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %3628 = tosa.matmul %3624, %3627 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %3629 = tosa.reshape %3628 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3630 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3631 = tosa.reciprocal %3630 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3632 = tosa.mul %3629, %3631 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3633 = tosa.add %3632, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3634 = tensor.empty() : tensor<1x32x40x40xf32>
    %3635 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3633 : tensor<1x32x40x40xbf16>) outs(%3634 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3636 = tensor.empty() : tensor<1x32x40x1xf32>
    %3637 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3636 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3638 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3635 : tensor<1x32x40x40xf32>) outs(%3636 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3639 = tensor.empty() : tensor<1x32x40x40xf32>
    %3640 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3635, %3638 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3639 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %3641 = tensor.empty() : tensor<1x32x40x1xf32>
    %3642 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3641 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3643 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3640 : tensor<1x32x40x40xf32>) outs(%3642 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3644 = tensor.empty() : tensor<1x32x40x40xf32>
    %3645 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3640, %3643 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3644 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3646 = tensor.empty() : tensor<1x32x40x40xbf16>
    %3647 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3645 : tensor<1x32x40x40xf32>) outs(%3646 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %3648 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3649 = tosa.add %3647, %3648 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3650 = tosa.reshape %3649 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %3651 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3652 = tosa.add %3593, %3651 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3653 = tosa.reshape %3652 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3654 = tosa.matmul %3650, %3653 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3655 = tosa.reshape %3654 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3656 = tensor.empty() : tensor<1x40x32x128xbf16>
    %3657 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3655 : tensor<1x32x40x128xbf16>) outs(%3656 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %3658 = tosa.identity %3657 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %3659 = tosa.reshape %3658 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %3660 = tensor.empty() : tensor<4096x4096xbf16>
    %3661 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_422 : tensor<4096x4096xbf16>) outs(%3660 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3662 = tosa.reshape %3659 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1295 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3663 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3662, %3661 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1295 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3664 = tosa.reshape %3663 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3665 = tosa.add %3556, %3664 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3666 = tensor.empty() : tensor<1x40x4096xf32>
    %3667 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3665 : tensor<1x40x4096xbf16>) outs(%3666 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3668 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1296 = arith.constant 2 : i32
    %3669 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3667 : tensor<1x40x4096xf32>) outs(%3668 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1296 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1297 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3670 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3669 : tensor<1x40x4096xf32>) outs(%cst_1297 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3671 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3672 = tosa.add %3670, %3671 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3673 = tosa.rsqrt %3672 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3674 = tosa.mul %3667, %3673 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3675 = tensor.empty() : tensor<1x40x4096xbf16>
    %3676 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3674 : tensor<1x40x4096xf32>) outs(%3675 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3677 = tosa.reshape %extracted_slice_50 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3678 = tosa.mul %3677, %3676 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3679 = tensor.empty() : tensor<4096x11008xbf16>
    %3680 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_424 : tensor<11008x4096xbf16>) outs(%3679 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3681 = tosa.reshape %3678 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1298 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3682 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3681, %3680 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1298 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3683 = tosa.reshape %3682 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3684 = tensor.empty() : tensor<1x40x11008xbf16>
    %3685 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3683 : tensor<1x40x11008xbf16>) outs(%3684 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %3686 = tensor.empty() : tensor<4096x11008xbf16>
    %3687 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_426 : tensor<11008x4096xbf16>) outs(%3686 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3688 = tosa.reshape %3678 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1299 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3689 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3688, %3687 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1299 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3690 = tosa.reshape %3689 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3691 = tosa.mul %3685, %3690 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3692 = tensor.empty() : tensor<11008x4096xbf16>
    %3693 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_428 : tensor<4096x11008xbf16>) outs(%3692 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %3694 = tosa.reshape %3691 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1300 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3695 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3694, %3693 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1300 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3696 = tosa.reshape %3695 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3697 = tosa.add %3665, %3696 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3698 = tensor.empty() : tensor<1x40x4096xf32>
    %3699 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3697 : tensor<1x40x4096xbf16>) outs(%3698 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3700 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1301 = arith.constant 2 : i32
    %3701 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3699 : tensor<1x40x4096xf32>) outs(%3700 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1301 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1302 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3702 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3701 : tensor<1x40x4096xf32>) outs(%cst_1302 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3703 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3704 = tosa.add %3702, %3703 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3705 = tosa.rsqrt %3704 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3706 = tosa.mul %3699, %3705 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3707 = tensor.empty() : tensor<1x40x4096xbf16>
    %3708 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3706 : tensor<1x40x4096xf32>) outs(%3707 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3709 = tosa.reshape %extracted_slice_51 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3710 = tosa.mul %3709, %3708 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3711 = tensor.empty() : tensor<4096x4096xbf16>
    %3712 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_430 : tensor<4096x4096xbf16>) outs(%3711 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3713 = tosa.reshape %3710 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1303 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3714 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3713, %3712 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1303 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3715 = tosa.reshape %3714 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3716 = tensor.empty() : tensor<4096x4096xbf16>
    %3717 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_432 : tensor<4096x4096xbf16>) outs(%3716 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3718 = tosa.reshape %3710 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1304 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3719 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3718, %3717 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1304 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3720 = tosa.reshape %3719 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3721 = tensor.empty() : tensor<4096x4096xbf16>
    %3722 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_434 : tensor<4096x4096xbf16>) outs(%3721 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3723 = tosa.reshape %3710 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1305 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3724 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3723, %3722 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1305 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3725 = tosa.reshape %3724 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3726 = tosa.reshape %3715 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3727 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3728 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3726 : tensor<1x40x32x128xbf16>) outs(%3727 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3729 = tosa.reshape %3720 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3730 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3731 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3729 : tensor<1x40x32x128xbf16>) outs(%3730 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3732 = tosa.reshape %3725 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3733 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3734 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3732 : tensor<1x40x32x128xbf16>) outs(%3733 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1306 = tensor.extract_slice %expanded_620[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1307 = tensor.extract_slice %extracted_slice_1306[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1308 = tensor.extract_slice %extracted_slice_1307[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1309 = tensor.extract_slice %expanded_622[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1310 = tensor.extract_slice %extracted_slice_1309[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1311 = tensor.extract_slice %extracted_slice_1310[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %3735 = tensor.empty() : tensor<1x40x128xbf16>
    %3736  = tensor.collapse_shape %extracted_slice_1308 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3737 = tensor.empty() : tensor<40x128xbf16>
    %3738 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3736 : tensor<1x40x128xbf16>) outs(%3737 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3739 = tensor.empty() : tensor<1x40x128xbf16>
    %3740  = tensor.collapse_shape %extracted_slice_1311 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3741 = tensor.empty() : tensor<40x128xbf16>
    %3742 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3740 : tensor<1x40x128xbf16>) outs(%3741 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3743 = tensor.empty() : tensor<1x40x128xbf16>
    %3744 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3743 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3738[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3745 = tosa.reshape %3744 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3746 = tensor.empty() : tensor<1x40x128xbf16>
    %3747 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3746 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3742[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3748 = tosa.reshape %3747 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3749 = tosa.mul %3728, %3745 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1312 = tensor.extract_slice %3728[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1313 = tensor.extract_slice %3728[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3750 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3751 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1313 : tensor<1x32x40x64xbf16>) outs(%3750 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3752 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1314 = tensor.insert_slice %3751 into %3752[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1315 = tensor.insert_slice %extracted_slice_1312 into %inserted_slice_1314[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3753 = tosa.mul %inserted_slice_1315, %3748 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3754 = tosa.add %3749, %3753 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3755 = tosa.mul %3731, %3745 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1316 = tensor.extract_slice %3731[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1317 = tensor.extract_slice %3731[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3756 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3757 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1317 : tensor<1x32x40x64xbf16>) outs(%3756 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3758 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1318 = tensor.insert_slice %3757 into %3758[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1319 = tensor.insert_slice %extracted_slice_1316 into %inserted_slice_1318[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3759 = tosa.mul %inserted_slice_1319, %3748 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3760 = tosa.add %3755, %3759 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3761 = tensor.empty() : tensor<1x32x128x40xbf16>
    %3762 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3760 : tensor<1x32x40x128xbf16>) outs(%3761 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %3763 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3764 = tosa.add %3754, %3763 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3765 = tosa.reshape %3764 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3766 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %3767 = tosa.add %3762, %3766 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %3768 = tosa.reshape %3767 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %3769 = tosa.matmul %3765, %3768 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %3770 = tosa.reshape %3769 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3771 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3772 = tosa.reciprocal %3771 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3773 = tosa.mul %3770, %3772 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3774 = tosa.add %3773, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3775 = tensor.empty() : tensor<1x32x40x40xf32>
    %3776 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3774 : tensor<1x32x40x40xbf16>) outs(%3775 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3777 = tensor.empty() : tensor<1x32x40x1xf32>
    %3778 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3777 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3779 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3776 : tensor<1x32x40x40xf32>) outs(%3777 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3780 = tensor.empty() : tensor<1x32x40x40xf32>
    %3781 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3776, %3779 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3780 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %3782 = tensor.empty() : tensor<1x32x40x1xf32>
    %3783 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3782 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3784 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3781 : tensor<1x32x40x40xf32>) outs(%3783 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3785 = tensor.empty() : tensor<1x32x40x40xf32>
    %3786 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3781, %3784 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3785 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3787 = tensor.empty() : tensor<1x32x40x40xbf16>
    %3788 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3786 : tensor<1x32x40x40xf32>) outs(%3787 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %3789 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3790 = tosa.add %3788, %3789 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3791 = tosa.reshape %3790 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %3792 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3793 = tosa.add %3734, %3792 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3794 = tosa.reshape %3793 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3795 = tosa.matmul %3791, %3794 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3796 = tosa.reshape %3795 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3797 = tensor.empty() : tensor<1x40x32x128xbf16>
    %3798 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3796 : tensor<1x32x40x128xbf16>) outs(%3797 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %3799 = tosa.identity %3798 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %3800 = tosa.reshape %3799 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %3801 = tensor.empty() : tensor<4096x4096xbf16>
    %3802 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_436 : tensor<4096x4096xbf16>) outs(%3801 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3803 = tosa.reshape %3800 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1320 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3804 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3803, %3802 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1320 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3805 = tosa.reshape %3804 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3806 = tosa.add %3697, %3805 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3807 = tensor.empty() : tensor<1x40x4096xf32>
    %3808 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3806 : tensor<1x40x4096xbf16>) outs(%3807 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3809 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1321 = arith.constant 2 : i32
    %3810 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3808 : tensor<1x40x4096xf32>) outs(%3809 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1321 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1322 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3811 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3810 : tensor<1x40x4096xf32>) outs(%cst_1322 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3812 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3813 = tosa.add %3811, %3812 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3814 = tosa.rsqrt %3813 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3815 = tosa.mul %3808, %3814 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3816 = tensor.empty() : tensor<1x40x4096xbf16>
    %3817 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3815 : tensor<1x40x4096xf32>) outs(%3816 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3818 = tosa.reshape %extracted_slice_52 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3819 = tosa.mul %3818, %3817 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3820 = tensor.empty() : tensor<4096x11008xbf16>
    %3821 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_438 : tensor<11008x4096xbf16>) outs(%3820 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3822 = tosa.reshape %3819 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1323 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3823 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3822, %3821 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1323 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3824 = tosa.reshape %3823 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3825 = tensor.empty() : tensor<1x40x11008xbf16>
    %3826 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3824 : tensor<1x40x11008xbf16>) outs(%3825 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %3827 = tensor.empty() : tensor<4096x11008xbf16>
    %3828 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_440 : tensor<11008x4096xbf16>) outs(%3827 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3829 = tosa.reshape %3819 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1324 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3830 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3829, %3828 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1324 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3831 = tosa.reshape %3830 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3832 = tosa.mul %3826, %3831 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3833 = tensor.empty() : tensor<11008x4096xbf16>
    %3834 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_442 : tensor<4096x11008xbf16>) outs(%3833 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %3835 = tosa.reshape %3832 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1325 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3836 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3835, %3834 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1325 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3837 = tosa.reshape %3836 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3838 = tosa.add %3806, %3837 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3839 = tensor.empty() : tensor<1x40x4096xf32>
    %3840 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3838 : tensor<1x40x4096xbf16>) outs(%3839 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3841 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1326 = arith.constant 2 : i32
    %3842 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3840 : tensor<1x40x4096xf32>) outs(%3841 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1326 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1327 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3843 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3842 : tensor<1x40x4096xf32>) outs(%cst_1327 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3844 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3845 = tosa.add %3843, %3844 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3846 = tosa.rsqrt %3845 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3847 = tosa.mul %3840, %3846 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3848 = tensor.empty() : tensor<1x40x4096xbf16>
    %3849 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3847 : tensor<1x40x4096xf32>) outs(%3848 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3850 = tosa.reshape %extracted_slice_53 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3851 = tosa.mul %3850, %3849 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3852 = tensor.empty() : tensor<4096x4096xbf16>
    %3853 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_444 : tensor<4096x4096xbf16>) outs(%3852 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3854 = tosa.reshape %3851 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1328 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3855 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3854, %3853 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1328 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3856 = tosa.reshape %3855 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3857 = tensor.empty() : tensor<4096x4096xbf16>
    %3858 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_446 : tensor<4096x4096xbf16>) outs(%3857 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3859 = tosa.reshape %3851 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1329 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3860 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3859, %3858 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1329 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3861 = tosa.reshape %3860 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3862 = tensor.empty() : tensor<4096x4096xbf16>
    %3863 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_448 : tensor<4096x4096xbf16>) outs(%3862 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3864 = tosa.reshape %3851 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1330 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3865 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3864, %3863 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1330 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3866 = tosa.reshape %3865 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3867 = tosa.reshape %3856 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3868 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3869 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3867 : tensor<1x40x32x128xbf16>) outs(%3868 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3870 = tosa.reshape %3861 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3871 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3872 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3870 : tensor<1x40x32x128xbf16>) outs(%3871 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %3873 = tosa.reshape %3866 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %3874 = tensor.empty() : tensor<1x32x40x128xbf16>
    %3875 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3873 : tensor<1x40x32x128xbf16>) outs(%3874 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1331 = tensor.extract_slice %expanded_624[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1332 = tensor.extract_slice %extracted_slice_1331[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1333 = tensor.extract_slice %extracted_slice_1332[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1334 = tensor.extract_slice %expanded_626[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1335 = tensor.extract_slice %extracted_slice_1334[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1336 = tensor.extract_slice %extracted_slice_1335[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %3876 = tensor.empty() : tensor<1x40x128xbf16>
    %3877  = tensor.collapse_shape %extracted_slice_1333 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3878 = tensor.empty() : tensor<40x128xbf16>
    %3879 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3877 : tensor<1x40x128xbf16>) outs(%3878 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3880 = tensor.empty() : tensor<1x40x128xbf16>
    %3881  = tensor.collapse_shape %extracted_slice_1336 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %3882 = tensor.empty() : tensor<40x128xbf16>
    %3883 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3881 : tensor<1x40x128xbf16>) outs(%3882 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %3884 = tensor.empty() : tensor<1x40x128xbf16>
    %3885 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3884 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3879[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3886 = tosa.reshape %3885 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3887 = tensor.empty() : tensor<1x40x128xbf16>
    %3888 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%3887 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %3883[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %3889 = tosa.reshape %3888 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %3890 = tosa.mul %3869, %3886 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1337 = tensor.extract_slice %3869[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1338 = tensor.extract_slice %3869[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3891 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3892 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1338 : tensor<1x32x40x64xbf16>) outs(%3891 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3893 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1339 = tensor.insert_slice %3892 into %3893[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1340 = tensor.insert_slice %extracted_slice_1337 into %inserted_slice_1339[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3894 = tosa.mul %inserted_slice_1340, %3889 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3895 = tosa.add %3890, %3894 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3896 = tosa.mul %3872, %3886 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1341 = tensor.extract_slice %3872[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1342 = tensor.extract_slice %3872[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %3897 = tensor.empty() : tensor<1x32x40x64xbf16>
    %3898 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1342 : tensor<1x32x40x64xbf16>) outs(%3897 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %3899 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1343 = tensor.insert_slice %3898 into %3899[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1344 = tensor.insert_slice %extracted_slice_1341 into %inserted_slice_1343[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %3900 = tosa.mul %inserted_slice_1344, %3889 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3901 = tosa.add %3896, %3900 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3902 = tensor.empty() : tensor<1x32x128x40xbf16>
    %3903 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3901 : tensor<1x32x40x128xbf16>) outs(%3902 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %3904 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3905 = tosa.add %3895, %3904 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3906 = tosa.reshape %3905 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3907 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %3908 = tosa.add %3903, %3907 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %3909 = tosa.reshape %3908 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %3910 = tosa.matmul %3906, %3909 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %3911 = tosa.reshape %3910 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3912 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3913 = tosa.reciprocal %3912 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3914 = tosa.mul %3911, %3913 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3915 = tosa.add %3914, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3916 = tensor.empty() : tensor<1x32x40x40xf32>
    %3917 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3915 : tensor<1x32x40x40xbf16>) outs(%3916 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3918 = tensor.empty() : tensor<1x32x40x1xf32>
    %3919 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3918 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3920 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3917 : tensor<1x32x40x40xf32>) outs(%3918 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3921 = tensor.empty() : tensor<1x32x40x40xf32>
    %3922 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3917, %3920 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3921 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %3923 = tensor.empty() : tensor<1x32x40x1xf32>
    %3924 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3923 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %3925 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3922 : tensor<1x32x40x40xf32>) outs(%3924 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %3926 = tensor.empty() : tensor<1x32x40x40xf32>
    %3927 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3922, %3925 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%3926 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %3928 = tensor.empty() : tensor<1x32x40x40xbf16>
    %3929 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3927 : tensor<1x32x40x40xf32>) outs(%3928 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %3930 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %3931 = tosa.add %3929, %3930 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %3932 = tosa.reshape %3931 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %3933 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %3934 = tosa.add %3875, %3933 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3935 = tosa.reshape %3934 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3936 = tosa.matmul %3932, %3935 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %3937 = tosa.reshape %3936 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %3938 = tensor.empty() : tensor<1x40x32x128xbf16>
    %3939 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3937 : tensor<1x32x40x128xbf16>) outs(%3938 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %3940 = tosa.identity %3939 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %3941 = tosa.reshape %3940 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %3942 = tensor.empty() : tensor<4096x4096xbf16>
    %3943 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_450 : tensor<4096x4096xbf16>) outs(%3942 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3944 = tosa.reshape %3941 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1345 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3945 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3944, %3943 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1345 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3946 = tosa.reshape %3945 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3947 = tosa.add %3838, %3946 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3948 = tensor.empty() : tensor<1x40x4096xf32>
    %3949 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3947 : tensor<1x40x4096xbf16>) outs(%3948 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3950 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1346 = arith.constant 2 : i32
    %3951 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3949 : tensor<1x40x4096xf32>) outs(%3950 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1346 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1347 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3952 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3951 : tensor<1x40x4096xf32>) outs(%cst_1347 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3953 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3954 = tosa.add %3952, %3953 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3955 = tosa.rsqrt %3954 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3956 = tosa.mul %3949, %3955 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3957 = tensor.empty() : tensor<1x40x4096xbf16>
    %3958 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3956 : tensor<1x40x4096xf32>) outs(%3957 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3959 = tosa.reshape %extracted_slice_54 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3960 = tosa.mul %3959, %3958 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3961 = tensor.empty() : tensor<4096x11008xbf16>
    %3962 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_452 : tensor<11008x4096xbf16>) outs(%3961 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3963 = tosa.reshape %3960 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1348 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3964 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3963, %3962 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1348 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3965 = tosa.reshape %3964 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3966 = tensor.empty() : tensor<1x40x11008xbf16>
    %3967 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3965 : tensor<1x40x11008xbf16>) outs(%3966 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %3968 = tensor.empty() : tensor<4096x11008xbf16>
    %3969 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_454 : tensor<11008x4096xbf16>) outs(%3968 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %3970 = tosa.reshape %3960 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1349 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %3971 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3970, %3969 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1349 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %3972 = tosa.reshape %3971 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3973 = tosa.mul %3967, %3972 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %3974 = tensor.empty() : tensor<11008x4096xbf16>
    %3975 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_456 : tensor<4096x11008xbf16>) outs(%3974 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %3976 = tosa.reshape %3973 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1350 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3977 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3976, %3975 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1350 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3978 = tosa.reshape %3977 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3979 = tosa.add %3947, %3978 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3980 = tensor.empty() : tensor<1x40x4096xf32>
    %3981 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3979 : tensor<1x40x4096xbf16>) outs(%3980 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %3982 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1351 = arith.constant 2 : i32
    %3983 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3981 : tensor<1x40x4096xf32>) outs(%3982 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1351 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1352 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %3984 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3983 : tensor<1x40x4096xf32>) outs(%cst_1352 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %3985 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3986 = tosa.add %3984, %3985 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3987 = tosa.rsqrt %3986 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3988 = tosa.mul %3981, %3987 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %3989 = tensor.empty() : tensor<1x40x4096xbf16>
    %3990 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3988 : tensor<1x40x4096xf32>) outs(%3989 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %3991 = tosa.reshape %extracted_slice_55 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %3992 = tosa.mul %3991, %3990 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3993 = tensor.empty() : tensor<4096x4096xbf16>
    %3994 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_458 : tensor<4096x4096xbf16>) outs(%3993 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %3995 = tosa.reshape %3992 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1353 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %3996 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3995, %3994 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1353 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %3997 = tosa.reshape %3996 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %3998 = tensor.empty() : tensor<4096x4096xbf16>
    %3999 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_460 : tensor<4096x4096xbf16>) outs(%3998 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4000 = tosa.reshape %3992 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1354 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4001 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4000, %3999 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1354 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4002 = tosa.reshape %4001 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4003 = tensor.empty() : tensor<4096x4096xbf16>
    %4004 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_462 : tensor<4096x4096xbf16>) outs(%4003 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4005 = tosa.reshape %3992 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1355 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4006 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4005, %4004 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1355 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4007 = tosa.reshape %4006 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4008 = tosa.reshape %3997 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4009 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4010 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4008 : tensor<1x40x32x128xbf16>) outs(%4009 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %4011 = tosa.reshape %4002 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4012 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4013 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4011 : tensor<1x40x32x128xbf16>) outs(%4012 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %4014 = tosa.reshape %4007 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4015 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4016 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4014 : tensor<1x40x32x128xbf16>) outs(%4015 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1356 = tensor.extract_slice %expanded_628[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1357 = tensor.extract_slice %extracted_slice_1356[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1358 = tensor.extract_slice %extracted_slice_1357[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1359 = tensor.extract_slice %expanded_630[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1360 = tensor.extract_slice %extracted_slice_1359[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1361 = tensor.extract_slice %extracted_slice_1360[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %4017 = tensor.empty() : tensor<1x40x128xbf16>
    %4018  = tensor.collapse_shape %extracted_slice_1358 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %4019 = tensor.empty() : tensor<40x128xbf16>
    %4020 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%4018 : tensor<1x40x128xbf16>) outs(%4019 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %4021 = tensor.empty() : tensor<1x40x128xbf16>
    %4022  = tensor.collapse_shape %extracted_slice_1361 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %4023 = tensor.empty() : tensor<40x128xbf16>
    %4024 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%4022 : tensor<1x40x128xbf16>) outs(%4023 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %4025 = tensor.empty() : tensor<1x40x128xbf16>
    %4026 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%4025 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %4020[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %4027 = tosa.reshape %4026 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %4028 = tensor.empty() : tensor<1x40x128xbf16>
    %4029 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%4028 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %4024[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %4030 = tosa.reshape %4029 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %4031 = tosa.mul %4010, %4027 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1362 = tensor.extract_slice %4010[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1363 = tensor.extract_slice %4010[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %4032 = tensor.empty() : tensor<1x32x40x64xbf16>
    %4033 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1363 : tensor<1x32x40x64xbf16>) outs(%4032 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %4034 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1364 = tensor.insert_slice %4033 into %4034[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1365 = tensor.insert_slice %extracted_slice_1362 into %inserted_slice_1364[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %4035 = tosa.mul %inserted_slice_1365, %4030 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4036 = tosa.add %4031, %4035 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4037 = tosa.mul %4013, %4027 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1366 = tensor.extract_slice %4013[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1367 = tensor.extract_slice %4013[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %4038 = tensor.empty() : tensor<1x32x40x64xbf16>
    %4039 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1367 : tensor<1x32x40x64xbf16>) outs(%4038 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %4040 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1368 = tensor.insert_slice %4039 into %4040[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1369 = tensor.insert_slice %extracted_slice_1366 into %inserted_slice_1368[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %4041 = tosa.mul %inserted_slice_1369, %4030 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4042 = tosa.add %4037, %4041 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4043 = tensor.empty() : tensor<1x32x128x40xbf16>
    %4044 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4042 : tensor<1x32x40x128xbf16>) outs(%4043 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %4045 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %4046 = tosa.add %4036, %4045 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4047 = tosa.reshape %4046 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4048 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %4049 = tosa.add %4044, %4048 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %4050 = tosa.reshape %4049 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %4051 = tosa.matmul %4047, %4050 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %4052 = tosa.reshape %4051 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4053 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %4054 = tosa.reciprocal %4053 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4055 = tosa.mul %4052, %4054 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4056 = tosa.add %4055, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4057 = tensor.empty() : tensor<1x32x40x40xf32>
    %4058 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4056 : tensor<1x32x40x40xbf16>) outs(%4057 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %4059 = tensor.empty() : tensor<1x32x40x1xf32>
    %4060 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4059 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %4061 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4058 : tensor<1x32x40x40xf32>) outs(%4059 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %4062 = tensor.empty() : tensor<1x32x40x40xf32>
    %4063 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4058, %4061 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%4062 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %4064 = tensor.empty() : tensor<1x32x40x1xf32>
    %4065 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4064 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %4066 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4063 : tensor<1x32x40x40xf32>) outs(%4065 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %4067 = tensor.empty() : tensor<1x32x40x40xf32>
    %4068 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4063, %4066 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%4067 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %4069 = tensor.empty() : tensor<1x32x40x40xbf16>
    %4070 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4068 : tensor<1x32x40x40xf32>) outs(%4069 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %4071 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %4072 = tosa.add %4070, %4071 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4073 = tosa.reshape %4072 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %4074 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %4075 = tosa.add %4016, %4074 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4076 = tosa.reshape %4075 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4077 = tosa.matmul %4073, %4076 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4078 = tosa.reshape %4077 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4079 = tensor.empty() : tensor<1x40x32x128xbf16>
    %4080 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4078 : tensor<1x32x40x128xbf16>) outs(%4079 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %4081 = tosa.identity %4080 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %4082 = tosa.reshape %4081 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %4083 = tensor.empty() : tensor<4096x4096xbf16>
    %4084 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_464 : tensor<4096x4096xbf16>) outs(%4083 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4085 = tosa.reshape %4082 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1370 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4086 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4085, %4084 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1370 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4087 = tosa.reshape %4086 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4088 = tosa.add %3979, %4087 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4089 = tensor.empty() : tensor<1x40x4096xf32>
    %4090 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4088 : tensor<1x40x4096xbf16>) outs(%4089 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %4091 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1371 = arith.constant 2 : i32
    %4092 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4090 : tensor<1x40x4096xf32>) outs(%4091 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1371 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1372 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %4093 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4092 : tensor<1x40x4096xf32>) outs(%cst_1372 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %4094 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4095 = tosa.add %4093, %4094 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4096 = tosa.rsqrt %4095 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4097 = tosa.mul %4090, %4096 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4098 = tensor.empty() : tensor<1x40x4096xbf16>
    %4099 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4097 : tensor<1x40x4096xf32>) outs(%4098 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %4100 = tosa.reshape %extracted_slice_56 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %4101 = tosa.mul %4100, %4099 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4102 = tensor.empty() : tensor<4096x11008xbf16>
    %4103 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_466 : tensor<11008x4096xbf16>) outs(%4102 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %4104 = tosa.reshape %4101 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1373 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %4105 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4104, %4103 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1373 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %4106 = tosa.reshape %4105 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4107 = tensor.empty() : tensor<1x40x11008xbf16>
    %4108 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4106 : tensor<1x40x11008xbf16>) outs(%4107 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %4109 = tensor.empty() : tensor<4096x11008xbf16>
    %4110 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_468 : tensor<11008x4096xbf16>) outs(%4109 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %4111 = tosa.reshape %4101 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1374 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %4112 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4111, %4110 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1374 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %4113 = tosa.reshape %4112 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4114 = tosa.mul %4108, %4113 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4115 = tensor.empty() : tensor<11008x4096xbf16>
    %4116 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_470 : tensor<4096x11008xbf16>) outs(%4115 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %4117 = tosa.reshape %4114 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1375 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4118 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4117, %4116 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1375 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4119 = tosa.reshape %4118 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4120 = tosa.add %4088, %4119 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4121 = tensor.empty() : tensor<1x40x4096xf32>
    %4122 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4120 : tensor<1x40x4096xbf16>) outs(%4121 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %4123 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1376 = arith.constant 2 : i32
    %4124 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4122 : tensor<1x40x4096xf32>) outs(%4123 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1376 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1377 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %4125 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4124 : tensor<1x40x4096xf32>) outs(%cst_1377 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %4126 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4127 = tosa.add %4125, %4126 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4128 = tosa.rsqrt %4127 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4129 = tosa.mul %4122, %4128 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4130 = tensor.empty() : tensor<1x40x4096xbf16>
    %4131 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4129 : tensor<1x40x4096xf32>) outs(%4130 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %4132 = tosa.reshape %extracted_slice_57 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %4133 = tosa.mul %4132, %4131 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4134 = tensor.empty() : tensor<4096x4096xbf16>
    %4135 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_472 : tensor<4096x4096xbf16>) outs(%4134 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4136 = tosa.reshape %4133 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1378 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4137 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4136, %4135 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1378 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4138 = tosa.reshape %4137 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4139 = tensor.empty() : tensor<4096x4096xbf16>
    %4140 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_474 : tensor<4096x4096xbf16>) outs(%4139 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4141 = tosa.reshape %4133 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1379 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4142 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4141, %4140 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1379 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4143 = tosa.reshape %4142 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4144 = tensor.empty() : tensor<4096x4096xbf16>
    %4145 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_476 : tensor<4096x4096xbf16>) outs(%4144 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4146 = tosa.reshape %4133 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1380 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4147 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4146, %4145 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1380 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4148 = tosa.reshape %4147 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4149 = tosa.reshape %4138 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4150 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4151 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4149 : tensor<1x40x32x128xbf16>) outs(%4150 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %4152 = tosa.reshape %4143 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4153 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4154 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4152 : tensor<1x40x32x128xbf16>) outs(%4153 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %4155 = tosa.reshape %4148 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4156 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4157 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4155 : tensor<1x40x32x128xbf16>) outs(%4156 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1381 = tensor.extract_slice %expanded_632[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1382 = tensor.extract_slice %extracted_slice_1381[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1383 = tensor.extract_slice %extracted_slice_1382[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1384 = tensor.extract_slice %expanded_634[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1385 = tensor.extract_slice %extracted_slice_1384[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1386 = tensor.extract_slice %extracted_slice_1385[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %4158 = tensor.empty() : tensor<1x40x128xbf16>
    %4159  = tensor.collapse_shape %extracted_slice_1383 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %4160 = tensor.empty() : tensor<40x128xbf16>
    %4161 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%4159 : tensor<1x40x128xbf16>) outs(%4160 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %4162 = tensor.empty() : tensor<1x40x128xbf16>
    %4163  = tensor.collapse_shape %extracted_slice_1386 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %4164 = tensor.empty() : tensor<40x128xbf16>
    %4165 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%4163 : tensor<1x40x128xbf16>) outs(%4164 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %4166 = tensor.empty() : tensor<1x40x128xbf16>
    %4167 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%4166 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %4161[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %4168 = tosa.reshape %4167 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %4169 = tensor.empty() : tensor<1x40x128xbf16>
    %4170 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%4169 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %4165[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %4171 = tosa.reshape %4170 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %4172 = tosa.mul %4151, %4168 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1387 = tensor.extract_slice %4151[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1388 = tensor.extract_slice %4151[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %4173 = tensor.empty() : tensor<1x32x40x64xbf16>
    %4174 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1388 : tensor<1x32x40x64xbf16>) outs(%4173 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %4175 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1389 = tensor.insert_slice %4174 into %4175[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1390 = tensor.insert_slice %extracted_slice_1387 into %inserted_slice_1389[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %4176 = tosa.mul %inserted_slice_1390, %4171 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4177 = tosa.add %4172, %4176 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4178 = tosa.mul %4154, %4168 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1391 = tensor.extract_slice %4154[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1392 = tensor.extract_slice %4154[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %4179 = tensor.empty() : tensor<1x32x40x64xbf16>
    %4180 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1392 : tensor<1x32x40x64xbf16>) outs(%4179 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %4181 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1393 = tensor.insert_slice %4180 into %4181[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1394 = tensor.insert_slice %extracted_slice_1391 into %inserted_slice_1393[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %4182 = tosa.mul %inserted_slice_1394, %4171 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4183 = tosa.add %4178, %4182 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4184 = tensor.empty() : tensor<1x32x128x40xbf16>
    %4185 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4183 : tensor<1x32x40x128xbf16>) outs(%4184 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %4186 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %4187 = tosa.add %4177, %4186 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4188 = tosa.reshape %4187 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4189 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %4190 = tosa.add %4185, %4189 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %4191 = tosa.reshape %4190 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %4192 = tosa.matmul %4188, %4191 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %4193 = tosa.reshape %4192 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4194 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %4195 = tosa.reciprocal %4194 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4196 = tosa.mul %4193, %4195 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4197 = tosa.add %4196, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4198 = tensor.empty() : tensor<1x32x40x40xf32>
    %4199 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4197 : tensor<1x32x40x40xbf16>) outs(%4198 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %4200 = tensor.empty() : tensor<1x32x40x1xf32>
    %4201 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4200 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %4202 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4199 : tensor<1x32x40x40xf32>) outs(%4200 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %4203 = tensor.empty() : tensor<1x32x40x40xf32>
    %4204 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4199, %4202 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%4203 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %4205 = tensor.empty() : tensor<1x32x40x1xf32>
    %4206 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4205 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %4207 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4204 : tensor<1x32x40x40xf32>) outs(%4206 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %4208 = tensor.empty() : tensor<1x32x40x40xf32>
    %4209 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4204, %4207 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%4208 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %4210 = tensor.empty() : tensor<1x32x40x40xbf16>
    %4211 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4209 : tensor<1x32x40x40xf32>) outs(%4210 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %4212 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %4213 = tosa.add %4211, %4212 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4214 = tosa.reshape %4213 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %4215 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %4216 = tosa.add %4157, %4215 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4217 = tosa.reshape %4216 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4218 = tosa.matmul %4214, %4217 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4219 = tosa.reshape %4218 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4220 = tensor.empty() : tensor<1x40x32x128xbf16>
    %4221 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4219 : tensor<1x32x40x128xbf16>) outs(%4220 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %4222 = tosa.identity %4221 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %4223 = tosa.reshape %4222 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %4224 = tensor.empty() : tensor<4096x4096xbf16>
    %4225 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_478 : tensor<4096x4096xbf16>) outs(%4224 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4226 = tosa.reshape %4223 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1395 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4227 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4226, %4225 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1395 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4228 = tosa.reshape %4227 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4229 = tosa.add %4120, %4228 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4230 = tensor.empty() : tensor<1x40x4096xf32>
    %4231 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4229 : tensor<1x40x4096xbf16>) outs(%4230 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %4232 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1396 = arith.constant 2 : i32
    %4233 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4231 : tensor<1x40x4096xf32>) outs(%4232 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1396 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1397 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %4234 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4233 : tensor<1x40x4096xf32>) outs(%cst_1397 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %4235 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4236 = tosa.add %4234, %4235 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4237 = tosa.rsqrt %4236 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4238 = tosa.mul %4231, %4237 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4239 = tensor.empty() : tensor<1x40x4096xbf16>
    %4240 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4238 : tensor<1x40x4096xf32>) outs(%4239 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %4241 = tosa.reshape %extracted_slice_58 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %4242 = tosa.mul %4241, %4240 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4243 = tensor.empty() : tensor<4096x11008xbf16>
    %4244 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_480 : tensor<11008x4096xbf16>) outs(%4243 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %4245 = tosa.reshape %4242 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1398 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %4246 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4245, %4244 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1398 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %4247 = tosa.reshape %4246 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4248 = tensor.empty() : tensor<1x40x11008xbf16>
    %4249 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4247 : tensor<1x40x11008xbf16>) outs(%4248 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %4250 = tensor.empty() : tensor<4096x11008xbf16>
    %4251 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_482 : tensor<11008x4096xbf16>) outs(%4250 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %4252 = tosa.reshape %4242 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1399 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %4253 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4252, %4251 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1399 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %4254 = tosa.reshape %4253 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4255 = tosa.mul %4249, %4254 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4256 = tensor.empty() : tensor<11008x4096xbf16>
    %4257 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_484 : tensor<4096x11008xbf16>) outs(%4256 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %4258 = tosa.reshape %4255 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1400 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4259 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4258, %4257 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1400 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4260 = tosa.reshape %4259 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4261 = tosa.add %4229, %4260 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4262 = tensor.empty() : tensor<1x40x4096xf32>
    %4263 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4261 : tensor<1x40x4096xbf16>) outs(%4262 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %4264 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1401 = arith.constant 2 : i32
    %4265 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4263 : tensor<1x40x4096xf32>) outs(%4264 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1401 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1402 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %4266 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4265 : tensor<1x40x4096xf32>) outs(%cst_1402 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %4267 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4268 = tosa.add %4266, %4267 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4269 = tosa.rsqrt %4268 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4270 = tosa.mul %4263, %4269 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4271 = tensor.empty() : tensor<1x40x4096xbf16>
    %4272 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4270 : tensor<1x40x4096xf32>) outs(%4271 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %4273 = tosa.reshape %extracted_slice_59 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %4274 = tosa.mul %4273, %4272 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4275 = tensor.empty() : tensor<4096x4096xbf16>
    %4276 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_486 : tensor<4096x4096xbf16>) outs(%4275 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4277 = tosa.reshape %4274 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1403 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4278 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4277, %4276 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1403 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4279 = tosa.reshape %4278 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4280 = tensor.empty() : tensor<4096x4096xbf16>
    %4281 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_488 : tensor<4096x4096xbf16>) outs(%4280 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4282 = tosa.reshape %4274 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1404 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4283 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4282, %4281 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1404 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4284 = tosa.reshape %4283 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4285 = tensor.empty() : tensor<4096x4096xbf16>
    %4286 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_490 : tensor<4096x4096xbf16>) outs(%4285 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4287 = tosa.reshape %4274 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1405 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4288 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4287, %4286 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1405 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4289 = tosa.reshape %4288 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4290 = tosa.reshape %4279 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4291 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4292 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4290 : tensor<1x40x32x128xbf16>) outs(%4291 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %4293 = tosa.reshape %4284 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4294 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4295 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4293 : tensor<1x40x32x128xbf16>) outs(%4294 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %4296 = tosa.reshape %4289 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4297 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4298 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4296 : tensor<1x40x32x128xbf16>) outs(%4297 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1406 = tensor.extract_slice %expanded_636[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1407 = tensor.extract_slice %extracted_slice_1406[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1408 = tensor.extract_slice %extracted_slice_1407[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1409 = tensor.extract_slice %expanded_638[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1410 = tensor.extract_slice %extracted_slice_1409[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1411 = tensor.extract_slice %extracted_slice_1410[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %4299 = tensor.empty() : tensor<1x40x128xbf16>
    %4300  = tensor.collapse_shape %extracted_slice_1408 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %4301 = tensor.empty() : tensor<40x128xbf16>
    %4302 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%4300 : tensor<1x40x128xbf16>) outs(%4301 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %4303 = tensor.empty() : tensor<1x40x128xbf16>
    %4304  = tensor.collapse_shape %extracted_slice_1411 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %4305 = tensor.empty() : tensor<40x128xbf16>
    %4306 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%4304 : tensor<1x40x128xbf16>) outs(%4305 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %4307 = tensor.empty() : tensor<1x40x128xbf16>
    %4308 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%4307 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %4302[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %4309 = tosa.reshape %4308 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %4310 = tensor.empty() : tensor<1x40x128xbf16>
    %4311 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%4310 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %4306[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %4312 = tosa.reshape %4311 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %4313 = tosa.mul %4292, %4309 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1412 = tensor.extract_slice %4292[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1413 = tensor.extract_slice %4292[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %4314 = tensor.empty() : tensor<1x32x40x64xbf16>
    %4315 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1413 : tensor<1x32x40x64xbf16>) outs(%4314 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %4316 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1414 = tensor.insert_slice %4315 into %4316[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1415 = tensor.insert_slice %extracted_slice_1412 into %inserted_slice_1414[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %4317 = tosa.mul %inserted_slice_1415, %4312 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4318 = tosa.add %4313, %4317 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4319 = tosa.mul %4295, %4309 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1416 = tensor.extract_slice %4295[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1417 = tensor.extract_slice %4295[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %4320 = tensor.empty() : tensor<1x32x40x64xbf16>
    %4321 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1417 : tensor<1x32x40x64xbf16>) outs(%4320 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %4322 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1418 = tensor.insert_slice %4321 into %4322[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1419 = tensor.insert_slice %extracted_slice_1416 into %inserted_slice_1418[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %4323 = tosa.mul %inserted_slice_1419, %4312 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4324 = tosa.add %4319, %4323 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4325 = tensor.empty() : tensor<1x32x128x40xbf16>
    %4326 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4324 : tensor<1x32x40x128xbf16>) outs(%4325 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %4327 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %4328 = tosa.add %4318, %4327 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4329 = tosa.reshape %4328 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4330 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %4331 = tosa.add %4326, %4330 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %4332 = tosa.reshape %4331 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %4333 = tosa.matmul %4329, %4332 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %4334 = tosa.reshape %4333 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4335 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %4336 = tosa.reciprocal %4335 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4337 = tosa.mul %4334, %4336 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4338 = tosa.add %4337, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4339 = tensor.empty() : tensor<1x32x40x40xf32>
    %4340 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4338 : tensor<1x32x40x40xbf16>) outs(%4339 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %4341 = tensor.empty() : tensor<1x32x40x1xf32>
    %4342 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4341 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %4343 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4340 : tensor<1x32x40x40xf32>) outs(%4341 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %4344 = tensor.empty() : tensor<1x32x40x40xf32>
    %4345 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4340, %4343 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%4344 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %4346 = tensor.empty() : tensor<1x32x40x1xf32>
    %4347 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4346 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %4348 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4345 : tensor<1x32x40x40xf32>) outs(%4347 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %4349 = tensor.empty() : tensor<1x32x40x40xf32>
    %4350 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4345, %4348 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%4349 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %4351 = tensor.empty() : tensor<1x32x40x40xbf16>
    %4352 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4350 : tensor<1x32x40x40xf32>) outs(%4351 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %4353 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %4354 = tosa.add %4352, %4353 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4355 = tosa.reshape %4354 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %4356 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %4357 = tosa.add %4298, %4356 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4358 = tosa.reshape %4357 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4359 = tosa.matmul %4355, %4358 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4360 = tosa.reshape %4359 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4361 = tensor.empty() : tensor<1x40x32x128xbf16>
    %4362 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4360 : tensor<1x32x40x128xbf16>) outs(%4361 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %4363 = tosa.identity %4362 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %4364 = tosa.reshape %4363 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %4365 = tensor.empty() : tensor<4096x4096xbf16>
    %4366 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_492 : tensor<4096x4096xbf16>) outs(%4365 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4367 = tosa.reshape %4364 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1420 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4368 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4367, %4366 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1420 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4369 = tosa.reshape %4368 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4370 = tosa.add %4261, %4369 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4371 = tensor.empty() : tensor<1x40x4096xf32>
    %4372 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4370 : tensor<1x40x4096xbf16>) outs(%4371 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %4373 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1421 = arith.constant 2 : i32
    %4374 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4372 : tensor<1x40x4096xf32>) outs(%4373 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1421 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1422 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %4375 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4374 : tensor<1x40x4096xf32>) outs(%cst_1422 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %4376 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4377 = tosa.add %4375, %4376 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4378 = tosa.rsqrt %4377 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4379 = tosa.mul %4372, %4378 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4380 = tensor.empty() : tensor<1x40x4096xbf16>
    %4381 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4379 : tensor<1x40x4096xf32>) outs(%4380 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %4382 = tosa.reshape %extracted_slice_60 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %4383 = tosa.mul %4382, %4381 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4384 = tensor.empty() : tensor<4096x11008xbf16>
    %4385 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_494 : tensor<11008x4096xbf16>) outs(%4384 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %4386 = tosa.reshape %4383 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1423 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %4387 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4386, %4385 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1423 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %4388 = tosa.reshape %4387 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4389 = tensor.empty() : tensor<1x40x11008xbf16>
    %4390 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4388 : tensor<1x40x11008xbf16>) outs(%4389 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %4391 = tensor.empty() : tensor<4096x11008xbf16>
    %4392 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_496 : tensor<11008x4096xbf16>) outs(%4391 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %4393 = tosa.reshape %4383 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1424 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %4394 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4393, %4392 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1424 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %4395 = tosa.reshape %4394 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4396 = tosa.mul %4390, %4395 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4397 = tensor.empty() : tensor<11008x4096xbf16>
    %4398 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_498 : tensor<4096x11008xbf16>) outs(%4397 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %4399 = tosa.reshape %4396 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1425 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4400 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4399, %4398 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1425 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4401 = tosa.reshape %4400 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4402 = tosa.add %4370, %4401 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4403 = tensor.empty() : tensor<1x40x4096xf32>
    %4404 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4402 : tensor<1x40x4096xbf16>) outs(%4403 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %4405 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1426 = arith.constant 2 : i32
    %4406 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4404 : tensor<1x40x4096xf32>) outs(%4405 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1426 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1427 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %4407 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4406 : tensor<1x40x4096xf32>) outs(%cst_1427 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %4408 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4409 = tosa.add %4407, %4408 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4410 = tosa.rsqrt %4409 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4411 = tosa.mul %4404, %4410 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4412 = tensor.empty() : tensor<1x40x4096xbf16>
    %4413 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4411 : tensor<1x40x4096xf32>) outs(%4412 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %4414 = tosa.reshape %extracted_slice_61 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %4415 = tosa.mul %4414, %4413 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4416 = tensor.empty() : tensor<4096x4096xbf16>
    %4417 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_500 : tensor<4096x4096xbf16>) outs(%4416 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4418 = tosa.reshape %4415 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1428 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4419 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4418, %4417 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1428 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4420 = tosa.reshape %4419 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4421 = tensor.empty() : tensor<4096x4096xbf16>
    %4422 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_502 : tensor<4096x4096xbf16>) outs(%4421 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4423 = tosa.reshape %4415 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1429 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4424 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4423, %4422 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1429 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4425 = tosa.reshape %4424 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4426 = tensor.empty() : tensor<4096x4096xbf16>
    %4427 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_504 : tensor<4096x4096xbf16>) outs(%4426 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4428 = tosa.reshape %4415 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1430 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4429 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4428, %4427 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1430 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4430 = tosa.reshape %4429 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4431 = tosa.reshape %4420 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4432 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4433 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4431 : tensor<1x40x32x128xbf16>) outs(%4432 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %4434 = tosa.reshape %4425 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4435 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4436 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4434 : tensor<1x40x32x128xbf16>) outs(%4435 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %4437 = tosa.reshape %4430 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xbf16>) -> tensor<1x40x32x128xbf16>
    %4438 = tensor.empty() : tensor<1x32x40x128xbf16>
    %4439 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4437 : tensor<1x40x32x128xbf16>) outs(%4438 : tensor<1x32x40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x40x128xbf16>
    %extracted_slice_1431 = tensor.extract_slice %expanded_640[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1432 = tensor.extract_slice %extracted_slice_1431[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1433 = tensor.extract_slice %extracted_slice_1432[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %extracted_slice_1434 = tensor.extract_slice %expanded_642[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1435 = tensor.extract_slice %extracted_slice_1434[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x2048x128xbf16>
    %extracted_slice_1436 = tensor.extract_slice %extracted_slice_1435[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xbf16> to tensor<1x1x40x128xbf16>
    %4440 = tensor.empty() : tensor<1x40x128xbf16>
    %4441  = tensor.collapse_shape %extracted_slice_1433 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %4442 = tensor.empty() : tensor<40x128xbf16>
    %4443 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%4441 : tensor<1x40x128xbf16>) outs(%4442 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %4444 = tensor.empty() : tensor<1x40x128xbf16>
    %4445  = tensor.collapse_shape %extracted_slice_1436 [[0,1],[2],[3]] : tensor<1x1x40x128xbf16> into tensor<1x40x128xbf16>
    %4446 = tensor.empty() : tensor<40x128xbf16>
    %4447 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%4445 : tensor<1x40x128xbf16>) outs(%4446 : tensor<40x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<40x128xbf16>
    %4448 = tensor.empty() : tensor<1x40x128xbf16>
    %4449 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%4448 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %4443[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %4450 = tosa.reshape %4449 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %4451 = tensor.empty() : tensor<1x40x128xbf16>
    %4452 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x40xi64>) outs(%4451 : tensor<1x40x128xbf16>) {
    ^bb0(%in: i64, %out: bf16):
      %4564 = arith.index_cast %in : i64 to index
      %4565 = linalg.index 2 : index
      %extracted = tensor.extract %4447[%4564, %4565] : tensor<40x128xbf16>
      linalg.yield %extracted : bf16
    } -> tensor<1x40x128xbf16>
    %4453 = tosa.reshape %4452 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xbf16>) -> tensor<1x1x40x128xbf16>
    %4454 = tosa.mul %4433, %4450 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1437 = tensor.extract_slice %4433[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1438 = tensor.extract_slice %4433[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %4455 = tensor.empty() : tensor<1x32x40x64xbf16>
    %4456 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1438 : tensor<1x32x40x64xbf16>) outs(%4455 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %4457 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1439 = tensor.insert_slice %4456 into %4457[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1440 = tensor.insert_slice %extracted_slice_1437 into %inserted_slice_1439[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %4458 = tosa.mul %inserted_slice_1440, %4453 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4459 = tosa.add %4454, %4458 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4460 = tosa.mul %4436, %4450 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %extracted_slice_1441 = tensor.extract_slice %4436[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %extracted_slice_1442 = tensor.extract_slice %4436[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xbf16> to tensor<1x32x40x64xbf16>
    %4461 = tensor.empty() : tensor<1x32x40x64xbf16>
    %4462 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1442 : tensor<1x32x40x64xbf16>) outs(%4461 : tensor<1x32x40x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x64xbf16>
    %4463 = tensor.empty() : tensor<1x32x40x128xbf16>
    %inserted_slice_1443 = tensor.insert_slice %4462 into %4463[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %inserted_slice_1444 = tensor.insert_slice %extracted_slice_1441 into %inserted_slice_1443[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xbf16> into tensor<1x32x40x128xbf16>
    %4464 = tosa.mul %inserted_slice_1444, %4453 {shift = 0 : i8} : (tensor<1x32x40x128xbf16>, tensor<1x1x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4465 = tosa.add %4460, %4464 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4466 = tensor.empty() : tensor<1x32x128x40xbf16>
    %4467 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4465 : tensor<1x32x40x128xbf16>) outs(%4466 : tensor<1x32x128x40xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x40xbf16>
    %4468 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %4469 = tosa.add %4459, %4468 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4470 = tosa.reshape %4469 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4471 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x128x40xbf16>}> : () -> tensor<1x32x128x40xbf16>
    %4472 = tosa.add %4467, %4471 : (tensor<1x32x128x40xbf16>, tensor<1x32x128x40xbf16>) -> tensor<1x32x128x40xbf16>
    %4473 = tosa.reshape %4472 {new_shape = array<i64: 32, 128, 40>} : (tensor<1x32x128x40xbf16>) -> tensor<32x128x40xbf16>
    %4474 = tosa.matmul %4470, %4473 : (tensor<32x40x128xbf16>, tensor<32x128x40xbf16>) -> tensor<32x40x40xbf16>
    %4475 = tosa.reshape %4474 {new_shape = array<i64: 1, 32, 40, 40>} : (tensor<32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4476 = "tosa.const"() <{value = dense<1.131250e+01> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %4477 = tosa.reciprocal %4476 : (tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4478 = tosa.mul %4475, %4477 {shift = 0 : i8} : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4479 = tosa.add %4478, %31 : (tensor<1x32x40x40xbf16>, tensor<1x1x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4480 = tensor.empty() : tensor<1x32x40x40xf32>
    %4481 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4479 : tensor<1x32x40x40xbf16>) outs(%4480 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %4482 = tensor.empty() : tensor<1x32x40x1xf32>
    %4483 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4482 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %4484 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4481 : tensor<1x32x40x40xf32>) outs(%4482 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.maximumf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %4485 = tensor.empty() : tensor<1x32x40x40xf32>
    %4486 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4481, %4484 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%4485 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.subf %in, %in_1454 : f32
      %4565 = math.exp %4564 : f32
      linalg.yield %4565 : f32
    } -> tensor<1x32x40x40xf32>
    %4487 = tensor.empty() : tensor<1x32x40x1xf32>
    %4488 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4487 : tensor<1x32x40x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x40x1xf32>
    %4489 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4486 : tensor<1x32x40x40xf32>) outs(%4488 : tensor<1x32x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = arith.addf %in, %out : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x1xf32>
    %4490 = tensor.empty() : tensor<1x32x40x40xf32>
    %4491 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4486, %4489 : tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) outs(%4490 : tensor<1x32x40x40xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4564 = arith.divf %in, %in_1454 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x32x40x40xf32>
    %4492 = tensor.empty() : tensor<1x32x40x40xbf16>
    %4493 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4491 : tensor<1x32x40x40xf32>) outs(%4492 : tensor<1x32x40x40xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x32x40x40xbf16>
    %4494 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xbf16>}> : () -> tensor<1x32x40x40xbf16>
    %4495 = tosa.add %4493, %4494 : (tensor<1x32x40x40xbf16>, tensor<1x32x40x40xbf16>) -> tensor<1x32x40x40xbf16>
    %4496 = tosa.reshape %4495 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xbf16>) -> tensor<32x40x40xbf16>
    %4497 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xbf16>}> : () -> tensor<1x32x40x128xbf16>
    %4498 = tosa.add %4439, %4497 : (tensor<1x32x40x128xbf16>, tensor<1x32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4499 = tosa.reshape %4498 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4500 = tosa.matmul %4496, %4499 : (tensor<32x40x40xbf16>, tensor<32x40x128xbf16>) -> tensor<32x40x128xbf16>
    %4501 = tosa.reshape %4500 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xbf16>) -> tensor<1x32x40x128xbf16>
    %4502 = tensor.empty() : tensor<1x40x32x128xbf16>
    %4503 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4501 : tensor<1x32x40x128xbf16>) outs(%4502 : tensor<1x40x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x40x32x128xbf16>
    %4504 = tosa.identity %4503 : (tensor<1x40x32x128xbf16>) -> tensor<1x40x32x128xbf16>
    %4505 = tosa.reshape %4504 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x32x128xbf16>) -> tensor<1x40x4096xbf16>
    %4506 = tensor.empty() : tensor<4096x4096xbf16>
    %4507 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_506 : tensor<4096x4096xbf16>) outs(%4506 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %4508 = tosa.reshape %4505 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1445 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4509 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4508, %4507 : tensor<40x4096xbf16>, tensor<4096x4096xbf16>) outs(%cst_1445 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4510 = tosa.reshape %4509 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4511 = tosa.add %4402, %4510 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4512 = tensor.empty() : tensor<1x40x4096xf32>
    %4513 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4511 : tensor<1x40x4096xbf16>) outs(%4512 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %4514 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1446 = arith.constant 2 : i32
    %4515 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4513 : tensor<1x40x4096xf32>) outs(%4514 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1446 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1447 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %4516 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4515 : tensor<1x40x4096xf32>) outs(%cst_1447 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %4517 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4518 = tosa.add %4516, %4517 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4519 = tosa.rsqrt %4518 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4520 = tosa.mul %4513, %4519 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4521 = tensor.empty() : tensor<1x40x4096xbf16>
    %4522 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4520 : tensor<1x40x4096xf32>) outs(%4521 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %4523 = tosa.reshape %extracted_slice_62 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %4524 = tosa.mul %4523, %4522 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4525 = tensor.empty() : tensor<4096x11008xbf16>
    %4526 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_508 : tensor<11008x4096xbf16>) outs(%4525 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %4527 = tosa.reshape %4524 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1448 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %4528 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4527, %4526 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1448 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %4529 = tosa.reshape %4528 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4530 = tensor.empty() : tensor<1x40x11008xbf16>
    %4531 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4529 : tensor<1x40x11008xbf16>) outs(%4530 : tensor<1x40x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4564 = arith.negf %in : bf16
      %4565 = math.exp %4564 : bf16
      %cst_1454 = arith.constant 1.000000e+00 : bf16
      %4566 = arith.addf %cst_1454, %4565 : bf16
      %4567 = arith.divf %in, %4566 : bf16
      linalg.yield %4567 : bf16
    } -> tensor<1x40x11008xbf16>
    %4532 = tensor.empty() : tensor<4096x11008xbf16>
    %4533 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_510 : tensor<11008x4096xbf16>) outs(%4532 : tensor<4096x11008xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x11008xbf16>
    %4534 = tosa.reshape %4524 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1449 = arith.constant dense<0.000000e+00> : tensor<40x11008xbf16>
    %4535 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4534, %4533 : tensor<40x4096xbf16>, tensor<4096x11008xbf16>) outs(%cst_1449 : tensor<40x11008xbf16>) -> tensor<40x11008xbf16>
    %4536 = tosa.reshape %4535 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4537 = tosa.mul %4531, %4536 {shift = 0 : i8} : (tensor<1x40x11008xbf16>, tensor<1x40x11008xbf16>) -> tensor<1x40x11008xbf16>
    %4538 = tensor.empty() : tensor<11008x4096xbf16>
    %4539 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_512 : tensor<4096x11008xbf16>) outs(%4538 : tensor<11008x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<11008x4096xbf16>
    %4540 = tosa.reshape %4537 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xbf16>) -> tensor<40x11008xbf16>
    %cst_1450 = arith.constant dense<0.000000e+00> : tensor<40x4096xbf16>
    %4541 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4540, %4539 : tensor<40x11008xbf16>, tensor<11008x4096xbf16>) outs(%cst_1450 : tensor<40x4096xbf16>) -> tensor<40x4096xbf16>
    %4542 = tosa.reshape %4541 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4543 = tosa.add %4511, %4542 : (tensor<1x40x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4544 = tensor.empty() : tensor<1x40x4096xf32>
    %4545 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4543 : tensor<1x40x4096xbf16>) outs(%4544 : tensor<1x40x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %4546 = tensor.empty() : tensor<1x40x4096xf32>
    %c2_i32_1451 = arith.constant 2 : i32
    %4547 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4545 : tensor<1x40x4096xf32>) outs(%4546 : tensor<1x40x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4564 = math.fpowi %in, %c2_i32_1451 : f32, i32
      linalg.yield %4564 : f32
    } -> tensor<1x40x4096xf32>
    %cst_1452 = arith.constant dense<0.000000e+00> : tensor<1x40x1xf32>
    %4548 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4547 : tensor<1x40x4096xf32>) outs(%cst_1452 : tensor<1x40x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4564 = arith.divf %in, %cst_1454 : f32
      %4565 = arith.addf %4564, %out : f32
      linalg.yield %4565 : f32
    } -> tensor<1x40x1xf32>
    %4549 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %4550 = tosa.add %4548, %4549 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4551 = tosa.rsqrt %4550 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %4552 = tosa.mul %4545, %4551 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x1xf32>) -> tensor<1x40x4096xf32>
    %4553 = tensor.empty() : tensor<1x40x4096xbf16>
    %4554 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4552 : tensor<1x40x4096xf32>) outs(%4553 : tensor<1x40x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4564 = arith.truncf %in : f32 to bf16
      linalg.yield %4564 : bf16
    } -> tensor<1x40x4096xbf16>
    %4555 = tosa.reshape %extracted_slice_63 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %4556 = tosa.mul %4555, %4554 {shift = 0 : i8} : (tensor<1x1x4096xbf16>, tensor<1x40x4096xbf16>) -> tensor<1x40x4096xbf16>
    %4557 = tensor.empty() : tensor<4096x32000xbf16>
    %4558 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_514 : tensor<32000x4096xbf16>) outs(%4557 : tensor<4096x32000xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x32000xbf16>
    %4559 = tosa.reshape %4556 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xbf16>) -> tensor<40x4096xbf16>
    %cst_1453 = arith.constant dense<0.000000e+00> : tensor<40x32000xbf16>
    %4560 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4559, %4558 : tensor<40x4096xbf16>, tensor<4096x32000xbf16>) outs(%cst_1453 : tensor<40x32000xbf16>) -> tensor<40x32000xbf16>
    %4561 = tosa.reshape %4560 {new_shape = array<i64: 1, 40, 32000>} : (tensor<40x32000xbf16>) -> tensor<1x40x32000xbf16>
    %4562 = tensor.empty() : tensor<1x40x32000xf32>
    %4563 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4561 : tensor<1x40x32000xbf16>) outs(%4562 : tensor<1x40x32000xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %4564 = arith.extf %in : bf16 to f32
      linalg.yield %4564 : f32
    } -> tensor<1x40x32000xf32>
    return %4563, %4556 : tensor<1x40x32000xf32>, tensor<1x40x4096xbf16>
  }
}

